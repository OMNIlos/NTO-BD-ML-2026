import re
import html
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import hstack as sp_hstack

HERE = Path(__file__).parent
ROOT = HERE.parent

K = 20

def clean_text(value):
    if pd.isna(value):
        return ""
    text = str(value)
    text = html.unescape(text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = text.replace('\\n', ' ')
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'(\d+)\s*(мл|мг|гр|г|кг|л|шт|см|мм|м)\b', r'NUM\2', text, flags=re.IGNORECASE)
    text = re.sub(r'(\d+)\s*(ml|mg|g|kg|l|pcs|cm|mm|m)\b', r'NUM\2', text, flags=re.IGNORECASE)
    text = re.sub(r'[^0-9a-zA-Zа-яёА-ЯЁ%]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()


def build_lookups(df):
    lookups = {}
    for key_col, key_fn in [
        ("title_shop", lambda r: clean_text(r["title"]) + "|||" + clean_text(r["shop_category_name"])),
        ("vendor_title", lambda r: clean_text(r["vendor_name"]) + "|||" + clean_text(r["title"])),
        ("vendor_code", lambda r: str(r["vendor_code"]).strip()),
        ("title", lambda r: clean_text(r["title"])),
        ("shop_category_name", lambda r: clean_text(r["shop_category_name"])),
    ]:
        mapping = {}
        for _, row in df.iterrows():
            k = key_fn(row)
            if k in ("", "-", "|||", "|||-", "-|||", "-|||-"):
                continue
            mapping.setdefault(k, []).append(row["category_id"])

        clean = {}
        for k, cats in mapping.items():
            counter = Counter(cats)
            total = sum(counter.values())
            top_cat, top_count = counter.most_common(1)[0]
            if top_count == total:
                clean[k] = top_cat
        lookups[key_col] = clean

    mapping = {}
    for _, row in df.iterrows():
        words = clean_text(row["title"]).split()
        k = " ".join(words[:3]) if len(words) >= 3 else ""
        if not k:
            continue
        mapping.setdefault(k, []).append(row["category_id"])
    clean = {}
    for k, cats in mapping.items():
        counter = Counter(cats)
        total = sum(counter.values())
        top_cat, top_count = counter.most_common(1)[0]
        if top_count == total:
            clean[k] = top_cat
    lookups["title_3words"] = clean

    return lookups


print("Loading data...")
train = pd.read_csv(HERE / "train.tsv", sep="\t")

for col in ["vendor_name", "vendor_code", "title", "description", "shop_category_name"]:
    if col in train.columns:
        train[col] = train[col].fillna("")

train["title_clean"] = train["title"].apply(clean_text)
train["desc_clean"] = train["description"].apply(clean_text)
train["shop_cat_clean"] = train["shop_category_name"].apply(clean_text)

print("Building lookups...")
lookups = build_lookups(train)
for key, mapping in lookups.items():
    print(f"  {key}: {len(mapping)} entries")

cat2dept = train.groupby("category_id")["department_id"].first().to_dict()

print("Training TF-IDF + SVC + kNN...")
item_combined = train["title_clean"] + " " + train["shop_cat_clean"]
item_descs = train["desc_clean"]

tfidf_word = TfidfVectorizer(
    ngram_range=(1, 2), max_features=80000,
    min_df=1, max_df=0.98, sublinear_tf=True, dtype=np.float32,
)
tfidf_char = TfidfVectorizer(
    analyzer="char_wb", ngram_range=(3, 5), max_features=60000,
    min_df=1, max_df=0.995, sublinear_tf=True, dtype=np.float32,
)
tfidf_desc = TfidfVectorizer(
    ngram_range=(1, 2), max_features=40000,
    min_df=1, max_df=0.95, sublinear_tf=True, dtype=np.float32,
)

X_items = sp_hstack([
    tfidf_word.fit_transform(item_combined),
    tfidf_char.fit_transform(item_combined),
    tfidf_desc.fit_transform(item_descs),
])
print(f"  TF-IDF features: {X_items.shape[1]}")


svc = LinearSVC(C=2.0, max_iter=5000, class_weight="balanced")
svc.fit(X_items, train["department_id"].values)
print(f"  SVC trained")


cat_texts = {}
cat_descs_map = {}
for i in range(len(train)):
    cat = train.iloc[i]["category_id"]
    cat_texts.setdefault(cat, []).append(
        train.iloc[i]["title_clean"] + " " + train.iloc[i]["shop_cat_clean"]
    )
    cat_descs_map.setdefault(cat, []).append(train.iloc[i]["desc_clean"])

unique_cats = sorted(cat_texts.keys())
cat_combined_texts = pd.Series([" ".join(cat_texts[c]) for c in unique_cats])
cat_combined_descs = pd.Series([" ".join(cat_descs_map[c]) for c in unique_cats])

X_cats = sp_hstack([
    tfidf_word.transform(cat_combined_texts),
    tfidf_char.transform(cat_combined_texts),
    tfidf_desc.transform(cat_combined_descs),
])
print(f"  CatDoc matrix: {X_cats.shape}")

nn = NearestNeighbors(
    n_neighbors=min(K, len(unique_cats)),
    metric="cosine", algorithm="brute", n_jobs=-1,
)
nn.fit(X_cats)

print("Saving model...")
joblib.dump({
    "lookups": lookups,
    "cat2dept": cat2dept,
    "unique_cats": unique_cats,
    "tfidf_word": tfidf_word,
    "tfidf_char": tfidf_char,
    "tfidf_desc": tfidf_desc,
    "svc": svc,
    "nn": nn,
}, ROOT / "models.pkl", compress=3)

import os
size_mb = os.path.getsize(ROOT / "models.pkl") / 1024 / 1024
print(f"Model size: {size_mb:.1f} MB")
print("Done.")
