# calib_fit.py
import pandas as pd, numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split
import joblib

df = pd.merge(pd.read_csv("experimental_labels.csv"),
              pd.read_csv("chemprop_canonical.csv"), on="cid")

CALIB_MODELS = {}
for ep in ["Carcinogenicity","Mutagenicity","Genotoxicity"]:
    mask = df[f"True_{ep}"].notna()
    y = df.loc[mask, f"True_{ep}"].astype(int)
    p = df.loc[mask, f"Pred_{ep}"]
    p_tr, p_val, y_tr, y_val = train_test_split(p, y, stratify=y,
                                                test_size=0.2, random_state=42)
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_tr, y_tr)
    CALIB_MODELS[ep] = iso
    print(f"{ep} –  Brier_before {np.mean((p_val-y_val)**2):.3f}   "
          f"Brier_after {np.mean((iso.predict(p_val)-y_val)**2):.3f}")

joblib.dump(CALIB_MODELS, "isotonic_models.pkl")

# calib_apply.py
import pandas as pd, joblib, numpy as np
pred = pd.read_csv("chemprop_canonical.csv")
iso = joblib.load("isotonic_models.pkl")

for ep in iso:
    pred[f"Calib_{ep}"] = iso[ep].predict(pred[f"Pred_{ep}"])
pred.to_csv("chemprop_calibrated.csv", index=False)
print("✓ wrote chemprop_calibrated.csv with calibrated probs")
