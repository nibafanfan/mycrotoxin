import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

df = pd.read_csv("mycotoxin_chemprop_comparison.csv")

# Map experimental string labels to binary
label_map = {
    "Mutagenic": 1, "Non-mutagenic": 0,
    "Genotoxic": 1, "Non-genotoxic": 0,
    "Carcinogenic": 1, "Non-carcinogenic": 0
}
df["Known_Mutagenicity"]    = df["experimental mutagenicity"].map(label_map)
df["Known_Genotoxicity"]    = df["experimental in vitro genotoxicity"].map(label_map)
df["Known_Carcinogenicity"] = df["experimental carcinogenicity"].map(label_map)

# Map paper predictions (if string)
paper_cols = [
    ("in vitro genotoxicity prediction", "Paper_Genotoxicity"),
    ("mutagenicity prediction", "Paper_Mutagenicity"),
    ("carcinogenicity prediction", "Paper_Carcinogenicity")
]
for old, new in paper_cols:
    df[new] = df[old].map(label_map) if df[old].dtype == object else df[old]

print("===== ChemProp Threshold Sweeps =====")
for endpoint in ["Mutagenicity", "Genotoxicity", "Carcinogenicity"]:
    true_col = f"Known_{endpoint}"
    pred_col = f"Pred_{endpoint}"

    print(f"\n===== {endpoint.upper()} =====")
    if true_col not in df or pred_col not in df:
        print(f"⚠️  Missing columns for {endpoint}")
        continue

    valid = df[[true_col, pred_col]].dropna()
    if len(valid) < 5 or valid[true_col].nunique() < 2:
        print("⚠️  Too few usable rows or only one class present.")
        continue

    y_true = valid[true_col]
    y_prob = valid[pred_col]

    for thresh in [0.3, 0.5, 0.7]:
        y_pred = (y_prob >= thresh).astype(int)
        print(f"\n-- Threshold: {thresh} --")
        print("Confusion Matrix:")
        print(confusion_matrix(y_true, y_pred))
        print("Classification Report:")
        print(classification_report(y_true, y_pred, digits=3))
        try:
            auc = roc_auc_score(y_true, y_prob)
            print(f"ROC AUC: {auc:.3f}")
        except:
            print("⚠️  ROC AUC failed")

# Now compare ChemProp vs Paper predictions
print("\n\n===== ChemProp vs Paper Comparisons =====")

def compare_models(model1, model2, label, threshold=0.5):
    valid = df[[model1, model2]].dropna()
    print(f"\n===== {label.upper()} – ChemProp vs Paper =====")
    print(f"Total comparisons: {len(valid)}")

    # Convert ChemProp prob to binary
    y1 = (valid[model1] >= threshold).astype(int) if valid[model1].dtype == float else valid[model1].astype(int)
    y2 = valid[model2].astype(int)

    if len(y1) < 5 or y1.nunique() < 2 or y2.nunique() < 2:
        print("⚠️  Too few usable rows or only one class present.")
        return

    print("Confusion matrix:")
    print(confusion_matrix(y1, y2))
    print("Classification report:")
    print(classification_report(y1, y2, digits=3))

compare_models("Pred_Mutagenicity",    "Paper_Mutagenicity",    "Mutagenicity", threshold=0.3)
compare_models("Pred_Genotoxicity",    "Paper_Genotoxicity",    "Genotoxicity", threshold=0.5)
compare_models("Pred_Carcinogenicity", "Paper_Carcinogenicity", "Carcinogenicity", threshold=0.7)
