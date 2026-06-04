import re
import html
import joblib
import numpy as np
import pandas as pd
import warnings
from scipy.sparse import hstack as sp_hstack

warnings.filterwarnings('ignore')

K = 20
DEPT_BONUS = 0.12


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


test = pd.read_csv("test.tsv", sep="\t")
print(f"test size: {test.shape}")

for col in ["vendor_name", "vendor_code", "title", "description", "shop_category_name"]:
    if col in test.columns:
        test[col] = test[col].fillna("")

test["title_clean"] = test["title"].apply(clean_text)
test["desc_clean"] = test["description"].apply(clean_text)
test["shop_cat_clean"] = test["shop_category_name"].apply(clean_text)

mdl = joblib.load("models.pkl")
lookups = mdl["lookups"]
cat2dept = mdl["cat2dept"]
unique_cats = mdl["unique_cats"]
tfidf_word = mdl["tfidf_word"]
tfidf_char = mdl["tfidf_char"]
tfidf_desc = mdl["tfidf_desc"]
svc = mdl["svc"]
nn = mdl["nn"]

pred_cat = np.full(len(test), -1, dtype=int)

lookup_count = 0
for i, row in test.iterrows():
    for key_col, key_fn in [
        ("title_shop", lambda r: clean_text(r["title"]) + "|||" + clean_text(r["shop_category_name"])),
        ("vendor_title", lambda r: clean_text(r["vendor_name"]) + "|||" + clean_text(r["title"])),
        ("vendor_code", lambda r: str(r["vendor_code"]).strip()),
        ("title", lambda r: clean_text(r["title"])),
        ("shop_category_name", lambda r: clean_text(r["shop_category_name"])),
        ("title_3words", lambda r: " ".join(clean_text(r["title"]).split()[:3]) if len(clean_text(r["title"]).split()) >= 3 else ""),
    ]:
        k = key_fn(row)
        if k in lookups[key_col]:
            pred_cat[i] = lookups[key_col][k]
            lookup_count += 1
            break

print(f"Lookup: {lookup_count}/{len(test)}")

mask = pred_cat == -1
remaining = mask.sum()

if remaining > 0:
    test_indices = test.index[mask]

    test_combined = test["title_clean"] + " " + test["shop_cat_clean"]
    test_descs = test["desc_clean"]

    X_test = sp_hstack([
        tfidf_word.transform(test_combined),
        tfidf_char.transform(test_combined),
        tfidf_desc.transform(test_descs),
    ])

    pred_dept_svc = svc.predict(X_test[test_indices])

    test_texts_cd = pd.Series([
        test.iloc[i]["title_clean"] + " " + test.iloc[i]["shop_cat_clean"]
        for i in test_indices
    ])
    test_descs_cd = pd.Series([test.iloc[i]["desc_clean"] for i in test_indices])

    X_test_cd = sp_hstack([
        tfidf_word.transform(test_texts_cd),
        tfidf_char.transform(test_texts_cd),
        tfidf_desc.transform(test_descs_cd),
    ])

    distances, indices = nn.kneighbors(X_test_cd)

    nc = len(unique_cats)
    remaining_preds = []
    for j in range(len(test_indices)):
        cat_scores = {}
        for r in range(min(K, nc)):
            cat = unique_cats[indices[j][r]]
            sim = 1.0 - distances[j][r]
            bonus = DEPT_BONUS if cat2dept.get(cat, -1) == pred_dept_svc[j] else 0.0
            cat_scores[cat] = max(cat_scores.get(cat, 0), sim + bonus)
        remaining_preds.append(max(cat_scores, key=cat_scores.get))

    pred_cat[mask] = remaining_preds

print(f"CatDoc kNN: {remaining}/{len(test)}")

pred_dept = np.array([cat2dept.get(c, 0) for c in pred_cat])

pd.DataFrame({
    "department_id": pred_dept,
    "category_id": pred_cat,
}).to_csv("prediction.csv", index=False)

print("Saved prediction.csv")
