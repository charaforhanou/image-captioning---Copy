import pandas as pd
from difflib import SequenceMatcher

CSV_PATH = "results/test_predictions_r2gen.csv"

# ==================================================
# Chargement
# ==================================================
df = pd.read_csv(CSV_PATH)

preds = (
    df["prediction"]
    .fillna("")
    .astype(str)
    .str.lower()
    .str.strip()
)

refs = (
    df["reference"]
    .fillna("")
    .astype(str)
    .str.lower()
    .str.strip()
)

n_total = len(df)

# ==================================================
# Analyse références
# ==================================================
n_unique_refs = refs.nunique()
n_duplicates_refs = n_total - n_unique_refs
dup_rate = 100 * n_duplicates_refs / n_total

# ==================================================
# Exact Matches
# ==================================================
exact = (preds == refs)

n_exact = exact.sum()
exact_rate = 100 * n_exact / n_total

# ==================================================
# Similarité moyenne
# ==================================================
similarities = []

for pred, ref in zip(preds, refs):
    score = SequenceMatcher(None, pred, ref).ratio()
    similarities.append(score)

avg_similarity = sum(similarities) / len(similarities)

# ==================================================
# Distribution similarité
# ==================================================
sim95 = sum(s >= 0.95 for s in similarities)
sim90 = sum(s >= 0.90 for s in similarities)
sim80 = sum(s >= 0.80 for s in similarities)
sim70 = sum(s >= 0.70 for s in similarities)
sim_low = sum(s < 0.70 for s in similarities)

# ==================================================
# Analyse copie potentielle
# ==================================================
ref_set = set(refs)

copied_refs = 0

for pred in preds:
    if pred in ref_set:
        copied_refs += 1

copy_rate = 100 * copied_refs / n_total

# ==================================================
# Diversité des prédictions
# ==================================================
unique_predictions = preds.nunique()

prediction_diversity = (
    100 * unique_predictions / n_total
)

pred_duplicates = n_total - unique_predictions

# ==================================================
# Normal / Abnormal
# ==================================================
normal_patterns = [
    "lungs are clear",
    "no focal consolidation",
    "no pleural effusion",
    "no pneumothorax",
    "heart is normal",
    "cardiomediastinal silhouette is normal",
    "mediastinum is unremarkable"
]


def classify_report(text):

    text = str(text).lower()

    hits = sum(
        pattern in text
        for pattern in normal_patterns
    )

    if hits >= 2:
        return "Normal"

    return "Abnormal"


df["report_type"] = refs.apply(classify_report)

normal_count = (
    df["report_type"] == "Normal"
).sum()

abnormal_count = (
    df["report_type"] == "Abnormal"
).sum()

normal_pct = 100 * normal_count / n_total
abnormal_pct = 100 * abnormal_count / n_total

# ==================================================
# Affichage
# ==================================================
print("\n" + "=" * 65)
print("MCCAG DIAGNOSTIC REPORT")
print("=" * 65)

print(f"\nTotal studies                : {n_total}")

print("\n--- Reference Analysis ---")
print(f"Unique references            : {n_unique_refs}")
print(f"Duplicate references         : {n_duplicates_refs}")
print(f"Duplicate rate               : {dup_rate:.2f}%")

print("\n--- Prediction Analysis ---")
print(f"Exact matches                : {n_exact}")
print(f"Exact match rate             : {exact_rate:.2f}%")
print(f"Average similarity           : {avg_similarity:.4f}")

print("\n--- Similarity Distribution ---")
print(f">= 0.95                      : {sim95}")
print(f">= 0.90                      : {sim90}")
print(f">= 0.80                      : {sim80}")
print(f">= 0.70                      : {sim70}")
print(f"<  0.70                      : {sim_low}")

print("\n--- Copying Analysis ---")
print(f"Predictions matching ANY reference : {copied_refs}")
print(f"Copy rate                          : {copy_rate:.2f}%")

print("\n--- Generation Diversity ---")
print(f"Unique predictions                 : {unique_predictions}")
print(f"Duplicate predictions              : {pred_duplicates}")
print(f"Prediction diversity               : {prediction_diversity:.2f}%")

print("\n--- Clinical Distribution ---")
print(f"Normal studies               : {normal_count} ({normal_pct:.2f}%)")
print(f"Abnormal studies             : {abnormal_count} ({abnormal_pct:.2f}%)")

print("\n" + "=" * 65)