#!/usr/bin/env python3
"""
fetch_pred_by_cid.py
────────────────────────────────────────────────────────
1.  Reads an input table that contains a PubChem **CID** column.
2.  Looks up each CID → InChI via the PubChem PUG REST API.
3.  Queries the public ChemProp-Transformer endpoint `/predict_all`
    once per InChI.
4.  Writes three artefacts (all resumable):

    • <out>.jsonl      – raw ChemProp JSON for every success
    • <out>.csv        – flattened wide table (one column / property)
    • <out>.ok.csv     – CID, InChI for successful rows  (checkpoint)
    • <out>.fail.csv   – CID + reason for lookup/pred failures

   Re-running the script will skip CIDs already listed in *.ok.csv
   or *.fail.csv.

Dependencies
────────────
pip install pandas requests tqdm
"""

import argparse, json, os, time, pathlib, requests, pandas as pd
from requests.exceptions import ReadTimeout, ConnectionError, HTTPError
from tqdm import tqdm

# ───────── Config ─────────
BASE_CP     = ("http://chemprop-transformer-alb-2126755060."
               "us-east-1.elb.amazonaws.com/predict_all")
PUB_URL     = ("https://pubchem.ncbi.nlm.nih.gov/rest/pug/"
               "compound/cid/{cid}/property/InChI/JSON")

CP_TIMEOUT  = (5, 120)
PUB_TIMEOUT = (5, 60)
RETRIES     = 4
BACKOFF     = 4            # seconds
BATCH       = 50           # flush cadence

sess = requests.Session()

# ───────── Helpers ─────────
def retry_json(url, timeout):
    for i in range(RETRIES):
        try:
            r = sess.get(url, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except (ReadTimeout, ConnectionError, HTTPError, json.JSONDecodeError):
            if i < RETRIES - 1:
                time.sleep(BACKOFF * 2**i)
    return None

def cid_to_inchi(cid):
    """Return InChI string or None."""
    try:
        if pd.isna(cid):  # Check for NaN values
            return None
        data = retry_json(PUB_URL.format(cid=int(cid)), PUB_TIMEOUT)
        if data is None:
            return None
        try:
            return data["PropertyTable"]["Properties"][0]["InChI"]
        except (KeyError, IndexError):
            return None
    except (ValueError, TypeError):
        return None

def chemprop_predict(inchi):
    for i in range(RETRIES):
        try:
            r = sess.get(BASE_CP, params={"inchi": inchi}, timeout=CP_TIMEOUT)
            r.raise_for_status()
            return r.json()
        except (ReadTimeout, ConnectionError, HTTPError):
            if i < RETRIES - 1:
                time.sleep(min(BACKOFF * 2**i, 60))
    return None

# ─── replace the old flatten() with this one ────────────────────────────
def slug(text, maxlen=60):
    """simple slugify helper"""
    import re, unicodedata
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()
    text = re.sub(r"[^\w\s-]", "", text).strip().lower()
    text = re.sub(r"[-\s]+", "_", text)
    return text[:maxlen]

def flatten(pred):
    """
    Build a dict {column_key → value}.
    • Use every category string as a key
    • Fallback to the assay title if no categories present
    """
    flat = {}
    for p in pred:
        val   = p["value"]
        prop  = p.get("property", {})
        cats  = prop.get("categories", [])

        if cats:
            for c in cats:
                key = slug(c["category"])
                flat.setdefault(key, val)   # keep first occurrence
        else:
            title = prop.get("title") or p.get("title") or f"token_{p.get('property_token')}"
            key   = slug(title)
            flat.setdefault(key, val)
    return flat
# ───────────────────────────────────────────────────────────────────────


def load_ckpt(path):
    if not path.exists():
        return set()
    try:
        return set(pd.read_csv(path)["cid"])
    except pd.errors.EmptyDataError:
        return set()

# ───────── Main ─────────
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="input file with CID column")
    ap.add_argument("--cid", default="cid", help="name of the CID column")
    ap.add_argument("--out", default="chemprop_preds",
                    help="basename for output files (no extension)")
    ap.add_argument("--sep", default=";", help="delimiter of the input CSV")
    return ap.parse_args()

def main():
    args = parse_args()
    base   = pathlib.Path(args.out)
    jsonl  = base.with_suffix(".jsonl")
    flat_csv = base.with_suffix(".csv")
    ok_csv   = base.with_suffix(".ok.csv")
    fail_csv = base.with_suffix(".fail.csv")

    ok_done   = load_ckpt(ok_csv)
    fail_done = load_ckpt(fail_csv)
    processed = ok_done | fail_done

    raw_df = pd.read_csv(args.csv, sep=args.sep)
    if args.cid not in raw_df.columns:
        raise SystemExit(f"❌ column '{args.cid}' not found in {args.csv}")

    # Filter out NaN CIDs
    raw_df = raw_df.dropna(subset=[args.cid])
    todo_df = raw_df[~raw_df[args.cid].isin(processed)]

    jf = open(jsonl, "a")
    ok_rows, fail_rows, flat_rows = [], [], []

    def flush():
        if ok_rows:
            pd.DataFrame(ok_rows).to_csv(
                ok_csv, mode="a", header=not ok_csv.exists(), index=False
            )
            ok_rows.clear()
        if fail_rows:
            pd.DataFrame(fail_rows).to_csv(
                fail_csv, mode="a", header=not fail_csv.exists(), index=False
            )
            fail_rows.clear()
        if flat_rows:
            pd.DataFrame(flat_rows).to_csv(
                flat_csv, mode="a", header=not flat_csv.exists(), index=False
            )
            flat_rows.clear()

    for _, row in tqdm(todo_df.iterrows(), total=len(todo_df)):
        cid = row[args.cid]
        inchi = cid_to_inchi(cid)
        if not inchi:
            fail_rows.append({"cid": cid, "reason": "inchi_lookup"})
        else:
            pred = chemprop_predict(inchi)
            if pred is None:
                fail_rows.append({"cid": cid, "inchi": inchi, "reason": "chemprop"})
            else:
                ok_rows.append({"cid": cid, "inchi": inchi})
                jf.write(json.dumps({"cid": cid, "inchi": inchi, "pred": pred}) + "\n")

                flat = flatten(pred)
                flat["cid"] = cid
                flat["inchi"] = inchi
                flat_rows.append(flat)

        if (len(ok_rows) + len(fail_rows)) >= BATCH:
            flush()

    flush()
    jf.close()

    print(f"✅ finished   ok:{len(load_ckpt(ok_csv))}   fail:{len(load_ckpt(fail_csv))}")

if __name__ == "__main__":
    main()
