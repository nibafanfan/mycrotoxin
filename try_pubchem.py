import pandas as pd
import requests
from tqdm import tqdm
import time

# Example subset from MicotoXilico (replace with full list as needed)
data = [
    {"Name": "Zearalenone", "SMILES": "CC1=CC=C(C=C1)C2=CC(=O)OC3=C2C=C(C=C3)O", "Mutagenicity": 0, "Genotoxicity": 1, "Carcinogenicity": 1},
    {"Name": "Aflatoxin B1", "SMILES": "CC1=C2C(=O)OC3=C(C=CC4=C3OC(=O)C5=C4C=CO5)C2=CO1", "Mutagenicity": 1, "Genotoxicity": 1, "Carcinogenicity": 1},
    {"Name": "Deoxynivalenol", "SMILES": "CC1=C2C(=C(C(=O)O2)OC3=C1C=CC(=C3)O)O", "Mutagenicity": 0, "Genotoxicity": 0, "Carcinogenicity": 0}
]

df = pd.DataFrame(data)

# Convert SMILES to InChI using PubChem REST API
def smiles_to_inchi(smiles):
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{smiles}/property/InChI/JSON"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json()['PropertyTable']['Properties'][0]['InChI']
    except Exception as e:
        print(f"❌ Failed to convert SMILES: {smiles} — {e}")
        return None

df["InChI"] = df["SMILES"].apply(smiles_to_inchi)

# Query your chemprop-transformer model API
def query_model(inchi_str):
    try:
        url = f"http://chemprop-transformer-alb-2126755060.us-east-1.elb.amazonaws.com/predict_all?inchi={inchi_str}"
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        print(f"❌ API query failed for InChI: {inchi_str} — {e}")
    return None

results = []
for _, row in tqdm(df.iterrows(), total=len(df)):
    prediction = query_model(row["InChI"])
    time.sleep(0.5)
    result = {
        "Name": row["Name"],
        "InChI": row["InChI"],
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

df_results = pd.DataFrame(results)
df_results.to_csv("mycotoxin_chemprop_comparison.csv", index=False)
print("✅ Saved: mycotoxin_chemprop_comparison.csv")
