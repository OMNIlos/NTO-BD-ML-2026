import re
import html
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
from scipy.sparse import hstack as sp_hstack
import hashlib

HERE = Path(__file__).parent

K = 20
DEPT_BONUS = 0.12

RAW_TEXT_COLUMNS = [
    "vendor_name",
    "vendor_code",
    "title",
    "description",
    "shop_category_name",
]

INVALID_LOOKUP_KEYS = {"", "-", "|||", "|||-", "-|||", "-|||-"}
LOOKUP_SPECS = (
    ("title_shop", "title_shop_key"),
    ("vendor_title", "vendor_title_key"),
    ("vendor_code", "vendor_code_clean"),
    ("title", "title_clean"),
    ("shop_category_name", "shop_cat_clean"),
    ("title_3words", "title_3words_key"),
)


def clean_text(value):
    if pd.isna(value):
        return ""
    text = str(value)
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("\\n", " ")
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[^0-9a-zA-Zа-яёА-ЯЁ%]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def clean_vendor_code(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def first_three_words(text):
    words = text.split()
    if len(words) < 3:
        return ""
    return " ".join(words[:3])


def prepare_dataframe(df):
    df = df.copy()
    for col in RAW_TEXT_COLUMNS:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("")

    df["title_clean"] = df["title"].apply(clean_text)
    df["desc_clean"] = df["description"].apply(clean_text)
    df["shop_cat_clean"] = df["shop_category_name"].apply(clean_text)
    df["vendor_name_clean"] = df["vendor_name"].apply(clean_text)
    df["vendor_code_clean"] = df["vendor_code"].apply(clean_vendor_code)

    df["title_shop_key"] = df["title_clean"] + "|||" + df["shop_cat_clean"]
    df["vendor_title_key"] = df["vendor_name_clean"] + "|||" + df["title_clean"]
    df["title_3words_key"] = df["title_clean"].apply(first_three_words)
    return df


def stable_row_key(row, row_index):
    fingerprint = "||".join(
        (
            row["title_shop_key"],
            row["vendor_code_clean"],
            row["title_clean"],
            row["shop_cat_clean"],
            row["vendor_name_clean"],
            str(row_index),
        )
    )
    return hashlib.md5(fingerprint.encode("utf-8")).hexdigest()


def build_validation_split(df):
    df = df.reset_index(drop=True).copy()
    category_rows = {}
    for row_index, row in df.iterrows():
        category_rows.setdefault(row["category_id"], []).append(
            (stable_row_key(row, row_index), row_index)
        )

    train_indices = []
    valid_indices = []
    singleton_categories = 0
    repeated_categories = 0

    for category_id in sorted(category_rows, key=str):
        grouped_rows = sorted(category_rows[category_id], key=lambda item: (item[0], item[1]))
        if len(grouped_rows) == 1:
            singleton_categories += 1
            train_indices.append(grouped_rows[0][1])
            continue

        repeated_categories += 1
        valid_indices.append(grouped_rows[0][1])
        train_indices.extend(row_index for _, row_index in grouped_rows[1:])

    split_meta = {
        "mode": "category_leave_one_repeat_out",
        "train_size": len(train_indices),
        "valid_size": len(valid_indices),
        "valid_share": len(valid_indices) / max(len(df), 1),
        "singleton_category_count": singleton_categories,
        "repeated_category_count": repeated_categories,
        "singleton_rows_in_valid": 0,
        "repeated_rows_in_valid": len(valid_indices),
        "categories_in_valid": repeated_categories,
    }

    train_df = df.iloc[train_indices].reset_index(drop=True)
    valid_df = df.iloc[valid_indices].reset_index(drop=True)
    return train_df, valid_df, split_meta


def build_consistent_lookup(df, key_col):
    keys = df[key_col].astype(str)
    valid_mask = keys.str.strip().ne("") & ~keys.isin(INVALID_LOOKUP_KEYS)
    if not valid_mask.any():
        return {}

    table = pd.DataFrame(
        {
            "key": keys[valid_mask].values,
            "category_id": df.loc[valid_mask, "category_id"].values,
        }
    )
    unique_counts = table.groupby("key")["category_id"].nunique()
    unique_keys = unique_counts[unique_counts == 1].index
    if len(unique_keys) == 0:
        return {}

    resolved = (
        table[table["key"].isin(unique_keys)]
        .groupby("key")["category_id"]
        .first()
    )
    return {str(key): int(value) for key, value in resolved.items()}


def build_lookups(df):
    return {
        lookup_name: build_consistent_lookup(df, key_col)
        for lookup_name, key_col in LOOKUP_SPECS
    }


def apply_lookups(df, lookups):
    pred_cat = np.full(len(df), -1, dtype=np.int64)
    lookup_count = 0

    for lookup_name, key_col in LOOKUP_SPECS:
        unresolved = np.flatnonzero(pred_cat == -1)
        if len(unresolved) == 0:
            break

        mapped = df.iloc[unresolved][key_col].map(lookups[lookup_name])
        hit = mapped.notna().to_numpy()
        if hit.any():
            pred_cat[unresolved[hit]] = mapped[hit].astype(np.int64).to_numpy()
            lookup_count += int(hit.sum())

    return pred_cat, lookup_count


def run_split(tr, va):
    tr = tr.reset_index(drop=True)
    va = va.reset_index(drop=True)

    lookups = build_lookups(tr)
    cat2dept = tr.groupby("category_id")["department_id"].first().to_dict()

    pred_cat, lookup_count = apply_lookups(va, lookups)
    unresolved_mask = pred_cat == -1
    remaining = int(unresolved_mask.sum())

    if remaining > 0:
        train_items = tr["title_clean"] + " " + tr["shop_cat_clean"]
        train_descs = tr["desc_clean"]

        tfidf_word = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=80000,
            min_df=1,
            max_df=0.98,
            sublinear_tf=True,
            dtype=np.float32,
        )
        tfidf_char = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            max_features=60000,
            min_df=1,
            max_df=0.995,
            sublinear_tf=True,
            dtype=np.float32,
        )
        tfidf_desc = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=40000,
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,
            dtype=np.float32,
        )

        X_tr = sp_hstack(
            [
                tfidf_word.fit_transform(train_items),
                tfidf_char.fit_transform(train_items),
                tfidf_desc.fit_transform(train_descs),
            ],
            format="csr",
        )

        svc = LinearSVC(class_weight="balanced", C=1.0, max_iter=5000)
        svc.fit(X_tr, tr["department_id"].values)

        valid_items = va["title_clean"] + " " + va["shop_cat_clean"]
        valid_descs = va["desc_clean"]
        X_va = sp_hstack(
            [
                tfidf_word.transform(valid_items),
                tfidf_char.transform(valid_items),
                tfidf_desc.transform(valid_descs),
            ],
            format="csr",
        )

        valid_indices = np.flatnonzero(unresolved_mask)
        pred_dept_svc = svc.predict(X_va[valid_indices])

        cat_texts = train_items.groupby(tr["category_id"]).agg(" ".join)
        cat_descs = train_descs.groupby(tr["category_id"]).agg(" ".join)
        unique_cats = cat_texts.index.to_numpy()

        X_cats = sp_hstack(
            [
                tfidf_word.transform(cat_texts.values),
                tfidf_char.transform(cat_texts.values),
                tfidf_desc.transform(cat_descs.values),
            ],
            format="csr",
        )
        X_valid_catdoc = sp_hstack(
            [
                tfidf_word.transform(valid_items.iloc[valid_indices]),
                tfidf_char.transform(valid_items.iloc[valid_indices]),
                tfidf_desc.transform(valid_descs.iloc[valid_indices]),
            ],
            format="csr",
        )

        n_neighbors = min(K, len(unique_cats))
        nn = NearestNeighbors(
            n_neighbors=n_neighbors,
            metric="cosine",
            algorithm="brute",
            n_jobs=-1,
        )
        nn.fit(X_cats)
        distances, indices = nn.kneighbors(X_valid_catdoc)

        resolved = np.empty(len(valid_indices), dtype=np.int64)
        for row_idx in range(len(valid_indices)):
            cat_scores = {}
            for rank_idx in range(n_neighbors):
                cat = int(unique_cats[indices[row_idx, rank_idx]])
                sim = 1.0 - distances[row_idx, rank_idx]
                bonus = DEPT_BONUS if cat2dept.get(cat, -1) == pred_dept_svc[row_idx] else 0.0
                score = sim + bonus
                best_score = cat_scores.get(cat)
                if best_score is None or score > best_score:
                    cat_scores[cat] = score
            resolved[row_idx] = max(cat_scores, key=cat_scores.get)

        pred_cat[valid_indices] = resolved

    pred_dept = np.array([cat2dept.get(int(cat), 0) for cat in pred_cat])
    return pred_cat, pred_dept, lookup_count, remaining


def main():
    train = pd.read_csv(HERE / "train.tsv", sep="\t")
    train = prepare_dataframe(train)

    tr, va, split_meta = build_validation_split(train)
    if len(va) == 0:
        raise RuntimeError("Validation split is empty. Expected repeated categories in train.tsv.")

    print("\nValidation split: category_repeat_holdout")
    print(
        "  "
        f"train={split_meta['train_size']}  "
        f"valid={split_meta['valid_size']}  "
        f"valid_share={split_meta['valid_share']:.3f}"
    )
    print(
        "  "
        f"repeated_categories={split_meta['repeated_category_count']}  "
        f"singleton_categories={split_meta['singleton_category_count']}"
    )

    pred_cat, pred_dept, n_lookup, n_ranker = run_split(tr, va)

    f1 = f1_score(va["department_id"].values, pred_dept, average="macro")
    acc = accuracy_score(va["category_id"].values, pred_cat)
    score = 30 * f1 + 70 * acc

    print(
        "  "
        f"F1_dept={f1:.4f}  "
        f"Acc_cat={acc:.4f}  "
        f"Score={score:.2f}"
    )
    print(
        "  "
        f"lookup={n_lookup} ({n_lookup / len(va):.1%})  "
        f"catdoc_knn={n_ranker} ({n_ranker / len(va):.1%})"
    )


if __name__ == "__main__":
    main()
