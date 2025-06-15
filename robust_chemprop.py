#!/usr/bin/env python3
"""
robust_chemprop.py  –  resumable ChemProp batch predictor
──────────────────────────────────────────────────────────
• Continues from previous partial runs (looks at index column)
• Flushes progress every BATCH_SIZE compounds
• Gracefully writes checkpoints on ^C or any un-handled exception
"""

import os, time, json, urllib.parse, signal
import pandas as pd, requests
from tqdm import tqdm
from rdkit import Chem, RDLogger
from rdkit.Chem import inchi
from requests.exceptions import ReadTimeout, ConnectionError, HTTPError

# ───────────────────────── CONFIG ──────────────────────────
RAW_CSV         = "mycotoxins_tmap_final.csv"      # original big table
HEAD_N          = None            # e.g. 500 for quick test, or None for all
BATCH_SIZE      = 25              # flush to disk every N successes / fails

PUB_TIMEOUT     = (5, 60)
CP_TIMEOUT      = (5, 120)
RETRIES         = 4
BACKOFF         = 4
BASE_CP         = "http://chemprop-transformer-alb-2126755060.us-east-1.elb.amazonaws.com"

# checkpoint files
SUCCESS_CSV     = "success copy.csv"
INCHI_FAIL_CSV  = "inchi_failures.csv"
PRED_FAIL_CSV   = "prediction_failures.csv"

# Add new file for raw API responses
RAW_API_JSONL = "chemprop_api_raw.jsonl"

# silence RDKit warnings
RDLogger.DisableLog("rdApp.warning")

sess = requests.Session()

# ────────────────────── utility helpers ─────────────────────
def retry_json(url, timeout):
    for i in range(RETRIES):
        try:
            r = sess.get(url, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except (ReadTimeout, ConnectionError, HTTPError, json.JSONDecodeError) as e:
            if i < RETRIES - 1:
                time.sleep(BACKOFF * 2**i)
            else:
                return None

def rdkit_inchi(smiles):
    if pd.isna(smiles): return None
    mol = Chem.MolFromSmiles(smiles)
    return inchi.MolToInchi(mol) if mol else None

def pubchem_inchi_by_name(name):
    enc = urllib.parse.quote(str(name), safe="")
    url = (f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
           f"{enc}/property/InChI/JSON")
    try:
        return retry_json(url, PUB_TIMEOUT)['PropertyTable']['Properties'][0]['InChI']
    except (TypeError, KeyError):
        return None

def pubchem_inchi_by_cid(cid):
    if pd.isna(cid): return None
    url = (f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
           f"{int(cid)}/property/InChI/JSON")
    try:
        return retry_json(url, PUB_TIMEOUT)['PropertyTable']['Properties'][0]['InChI']
    except (TypeError, KeyError):
        return None

def chemprop_predict(inchi_str):
    for i in range(RETRIES):
        try:
            r = sess.get(f"{BASE_CP}/predict_all",
                         params={"inchi": inchi_str},
                         timeout=CP_TIMEOUT)
            r.raise_for_status()
            return r.json()
        except (ReadTimeout, ConnectionError, HTTPError) as e:
            if i < RETRIES - 1:
                wait = min(BACKOFF * 2**i, 30)
                print(f"   ↻ chemprop retry in {wait}s ({e})")
                time.sleep(wait)
            else:
                return None

# ───────────────────── checkpoint handling ──────────────────
import pandas as pd, os
from pandas.errors import EmptyDataError   # add import

def load_ckpt(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        # file is zero-byte or has no header – treat as empty
        return pd.DataFrame()

success_df      = load_ckpt(SUCCESS_CSV)
inchi_fail_df   = load_ckpt(INCHI_FAIL_CSV)
pred_fail_df    = load_ckpt(PRED_FAIL_CSV)

done_index = set(success_df.get("index", [])) \
           | set(inchi_fail_df.get("index", [])) \
           | set(pred_fail_df.get("index", []))
print(f"[checkpoint] already finished: {len(done_index)} rows")

# convert back to plain python lists we'll append to
succ, inchi_fail, pred_fail = (success_df.to_dict("records"),
                               inchi_fail_df.to_dict("records"),
                               pred_fail_df.to_dict("records"))

def flush():
    """Write current in-memory results to disk (overwrite)."""
    pd.DataFrame(succ     ).to_csv(SUCCESS_CSV,      index=False)
    pd.DataFrame(inchi_fail).to_csv(INCHI_FAIL_CSV,  index=False)
    pd.DataFrame(pred_fail ).to_csv(PRED_FAIL_CSV,   index=False)

# ensure we save on Ctrl-C
def _graceful_exit(signum, frame):
    print("\n[signal] caught Ctrl-C – flushing progress …")
    flush()
    raise SystemExit(1)

signal.signal(signal.SIGINT, _graceful_exit)

# ───────────────────────── LOAD RAW CSV ─────────────────────
# ───────────────────────── LOAD RAW CSV ─────────────────────
raw_df = pd.read_csv(RAW_CSV, sep=";").rename(columns=str.strip)

# 1️⃣  keep only rows that have at least one of the three labels
LABEL_COLS = [
    "experimental mutagenicity",
    "experimental in vitro genotoxicity",
    "experimental carcinogenicity",
]
raw_df = raw_df[ raw_df[LABEL_COLS].notna().any(axis=1) ]

# optional: for quick tests grab HEAD_N rows **after** filtering
if HEAD_N:
    raw_df = raw_df.head(HEAD_N)

# 2️⃣  resume logic (skip anything already in a checkpoint)
todo_df = raw_df[~raw_df["index"].isin(done_index)]
print(f"processing {len(todo_df)} remaining rows …")


# ───────────────────────── MAIN LOOP ────────────────────────
since_last_flush = 0

# Open the raw API file for appending
raw_api_file = open(RAW_API_JSONL, "a")

try:
    for _, row in tqdm(todo_df.iterrows(), total=len(todo_df)):
        idx     = row["index"]          # unique integer id
        name    = row["name"]
        cid     = row.get("cid")
        smiles  = row.get("SMILES")

        # 1) RDKit
        inch = rdkit_inchi(smiles)

        # 2) PubChem by name
        if not inch:
            inch = pubchem_inchi_by_name(name)

        # 3) PubChem by CID
        if not inch:
            inch = pubchem_inchi_by_cid(cid)

        if not inch:
            inchi_fail.append(row.to_dict())
            since_last_flush += 1
            continue

        pred = chemprop_predict(inch)
        if pred is None:
            rec          = row.to_dict()
            rec["InChI"] = inch
            pred_fail.append(rec)
            since_last_flush += 1
            continue

        # Save the raw API response
        raw_api_file.write(json.dumps({"index": idx, "name": name, "cid": cid, "InChI": inch, "api_response": pred}) + "\n")

        out            = {**row.to_dict(), "InChI": inch}
        for p in pred:
            for cat in p["property"]["categories"]:
                label = cat["category"].lower()
                if   "mutagenicity"  in label: out["Pred_Mutagenicity"]    = p["value"]
                elif "genotoxicity"  in label: out["Pred_Genotoxicity"]    = p["value"]
                elif "carcinogenic"  in label: out["Pred_Carcinogenicity"] = p["value"]
        succ.append(out)
        since_last_flush += 1

        # periodic flush
        if since_last_flush >= BATCH_SIZE:
            flush()
            since_last_flush = 0

finally:
    # always flush at the very end or on unhandled error
    flush()
    raw_api_file.close()
    print(f"\n✅  success: {len(succ)}   ❌ inchi_fail: {len(inchi_fail)}   ❌ pred_fail: {len(pred_fail)}")
