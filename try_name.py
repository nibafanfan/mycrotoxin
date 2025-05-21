import pandas as pd
import requests
import urllib.parse
from tqdm import tqdm
import time

# === Load top 100 compounds from CSV ===
df = pd.read_csv("mycotoxins_tmap_final.csv", sep=";").head(100)

# === Map experimental labels to binary ===
def binarize(value):
    if isinstance(value, str) and "mutagenic" in value.lower():
        return 1 if "non" not in value.lower() else 0
    return 0

df["Mutagenicity"] = df["experimental mutagenicity"].apply(binarize)
df["Genotoxicity"] = df["experimental in vitro genotoxicity"].apply(binarize)
df["Carcinogenicity"] = df["experimental carcinogenicity"].apply(binarize)

# === Convert names to InChI via PubChem ===
def name_to_inchi(name):
    try:
        encoded = urllib.parse.quote(str(name), safe='')
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded}/property/InChI/JSON"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json()['PropertyTable']['Properties'][0]['InChI']
    except Exception as e:
        print(f"❌ InChI conversion failed for {name} — {e}")
        return None

tqdm.pandas()
df["InChI"] = df["name"].progress_apply(name_to_inchi)

# === Query Chemprop-Transformer ===
def query_model(inchi_str):
    try:
        url = f"http://chemprop-transformer-alb-2126755060.us-east-1.elb.amazonaws.com/predict_all?inchi={inchi_str}"
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        print(f"❌ API query failed for InChI {inchi_str} — {e}")
    return None

results = []
for _, row in tqdm(df.iterrows(), total=len(df)):
    inchi = row["InChI"]
    prediction = query_model(inchi)
    time.sleep(0.5)

    result = {
        "Name": row["name"],
        "InChI": inchi,
        "Known_Mutagenicity": row["Mutagenicity"],
        "Known_Genotoxicity": row["Genotoxicity"],
        "Known_Carcinogenicity": row["Carcinogenicity"],
        "Predicted_Mutagenicity": None,
        "Predicted_Genotoxicity": None,
        "Predicted_Carcinogenicity": None
    }

    if prediction:
        for p in prediction:
            for cat in p.get("property", {}).get("categories", []):
                category = cat.get("category", "").lower()
                if "mutagenicity" in category:
                    result["Predicted_Mutagenicity"] = p["value"]
                elif "genotoxicity" in category:
                    result["Predicted_Genotoxicity"] = p["value"]
                elif "carcinogenicity" in category:
                    result["Predicted_Carcinogenicity"] = p["value"]
    results.append(result)

# === Save output ===
df_results = pd.DataFrame(results)
df_results.to_csv("mycotoxin_top100_predictions.csv", index=False)
print("✅ Saved: mycotoxin_top100_predictions.csv")
