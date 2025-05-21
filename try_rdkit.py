# run_mycotoxin_top100.py  – fully patched
import pandas as pd, requests, time, urllib.parse, json
from requests.exceptions import ReadTimeout, ConnectionError, HTTPError, RequestException
from rdkit import Chem, RDLogger
from rdkit.Chem import inchi
from tqdm import tqdm

##############################################################################
# CONFIG
##############################################################################
CSV             = "mycotoxins_tmap_final.csv"
TOP_N           = 100

PUB_TIMEOUT     = (5, 60)       # connect, read
CP_TIMEOUT      = (5, 120)      # ChemProp: allow up to 2 min per request
RETRIES         = 4             # 4 tries total
BACKOFF         = 4             # 4 s → 8 s → 16 s → 32 s

BASE_CP = "http://chemprop-transformer-alb-2126755060.us-east-1.elb.amazonaws.com"

sess = requests.Session()

# silence RDKit stereo / proton warnings
RDLogger.DisableLog("rdApp.warning")

##############################################################################
# HELPERS
##############################################################################
def retry_json(url, timeout):
    """GET a URL that returns JSON, with retries/back-off, return dict or None."""
    for i in range(RETRIES):
        try:
            r = sess.get(url, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except (ReadTimeout, ConnectionError, HTTPError, json.JSONDecodeError) as e:
            if i < RETRIES - 1:
                wait = BACKOFF * 2**i
                print(f" ↻ {url[:60]} … retry in {wait}s ({e})")
                time.sleep(wait)
            else:
                return None

def rdkit_inchi(smiles):
    if pd.isna(smiles): 
        return None
    m = Chem.MolFromSmiles(smiles)
    return inchi.MolToInchi(m) if m else None

def pubchem_name(name):
    enc = urllib.parse.quote(str(name), safe="")
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{enc}/property/InChI/JSON"
    try:
        return retry_json(url, PUB_TIMEOUT)['PropertyTable']['Properties'][0]['InChI']
    except (TypeError, KeyError):
        return None

def pubchem_cid(cid):
    if pd.isna(cid): 
        return None
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{int(cid)}/property/InChI/JSON"
    try:
        return retry_json(url, PUB_TIMEOUT)['PropertyTable']['Properties'][0]['InChI']
    except (TypeError, KeyError):
        return None

def chemprop(inchi_str):
    """Query ChemProp, retry on time-outs.  Returns JSON list or None."""
    for i in range(RETRIES):
        try:
            r = sess.get(
                f"{BASE_CP}/predict_all",
                params={"inchi": inchi_str},          # autom. URL-encoding
                timeout=CP_TIMEOUT
            )
            r.raise_for_status()
            return r.json()
        except (ReadTimeout, ConnectionError, HTTPError) as e:
            if i < RETRIES - 1:
                wait = min(BACKOFF * 2**i, 30)  # cap wait time to 30s
                print(f"   ↻ chemprop retry in {wait}s ({e})")
                time.sleep(wait)
            else:
                return None

##############################################################################
# LOAD DATA
##############################################################################
df = pd.read_csv(CSV, sep=";").rename(columns=str.strip)


##############################################################################
# MAIN LOOP
##############################################################################
succ, inchi_fail, pred_fail = [], [], []

for _, row in tqdm(df.iterrows(), total=len(df)):
    name   = row["name"]
    cid    = row.get("cid")
    smiles = row.get("SMILES")

    # ---------- 1) try SMILES → InChI via RDKit ----------
    inch = rdkit_inchi(smiles)

    # ---------- 2) fallback: PubChem name ----------
    if not inch:
        inch = pubchem_name(name)

    # ---------- 3) fallback: PubChem CID ----------
    if not inch:
        inch = pubchem_cid(cid)

    # ---------- no InChI at all ----------
    if not inch:
        inchi_fail.append(row.to_dict())
        continue

    # ---------- ChemProp ----------
    pred = chemprop(inch)

    if pred is None:                          # network or 5xx error
        r             = row.to_dict()
        r["InChI"]    = inch
        pred_fail.append(r)
        continue

    # ---------- parse ChemProp result ----------
    out = {**row.to_dict(), "InChI": inch}
    for p in pred:
        for cat in p["property"]["categories"]:
            c = cat["category"].lower()
            if   "mutagenicity"  in c: out["Pred_Mutagenicity"]    = p["value"]
            elif "genotoxicity"  in c: out["Pred_Genotoxicity"]    = p["value"]
            elif "carcinogenic"  in c: out["Pred_Carcinogenicity"] = p["value"]
    succ.append(out)

    time.sleep(0.25)          # small courtesy pause

##############################################################################
# SAVE RESULTS
##############################################################################
pd.DataFrame(succ     ).to_csv("success.csv",            index=False)
pd.DataFrame(inchi_fail).to_csv("inchi_failures.csv",     index=False)
pd.DataFrame(pred_fail ).to_csv("prediction_failures.csv",index=False)

print(f"✅ success: {len(succ)}   ❌ inchi-fail: {len(inchi_fail)}   ❌ pred-fail: {len(pred_fail)}")
