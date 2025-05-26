#!/usr/bin/env python3
"""
map_properties_with_gpt.py
────────────────────────────────────────────────────────
Collapse ChemProp-Transformer’s dozens of “category” strings
into exactly three canonical endpoints:

    • carcinogenicity
    • mutagenicity
    • genotoxicity

The script is **GPT-assisted once**:

1. Reads the raw predictions JSONL (`--jsonl`) produced by
   `fetch_pred_by_cid.py`.
2. Collects every unique category string.
3. If a YAML file (`--yaml`) already exists, reuse that mapping.
   Otherwise call GPT-4-o to build the mapping, then save it.
4. Writes `--csv` with four columns:

       cid  Pred_Carcinogenicity  Pred_Mutagenicity  Pred_Genotoxicity

Dependencies
────────────
pip install pandas pyyaml tqdm python-dotenv openai
Put `OPENAI_API_KEY=sk-…` in an `.env` file or export it.

Usage
─────
python map_properties_with_gpt.py \
       --jsonl chemprop_preds.jsonl \
       --yaml  chatgpt_property_map.yaml \
       --csv   chemprop_canonical.csv
"""

# ── load API key from .env BEFORE importing openai ──────────────
from dotenv import load_dotenv
load_dotenv(".env")

import os, json, re, argparse, pathlib, textwrap, yaml
import pandas as pd
from tqdm import tqdm
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")

CANON = ["carcinogenicity", "mutagenicity", "genotoxicity"]

# ───────────────────────── helper functions ────────────────────
def collect_categories(jsonl_path):
    cats = set()
    with open(jsonl_path) as jf:
        for ln in jf:
            for p in json.loads(ln)["pred"]:
                for c in p.get("property", {}).get("categories", []):
                    cats.add(c["category"].strip())
    return sorted(cats)

def extract_json(txt):
    """Return first JSON blob in txt or None."""
    m = re.search(r'{.*}', txt, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None

def gpt_map(cats, max_retry=3):
    prompt = textwrap.dedent(f"""
        Map each toxicological category string to exactly one of
        {CANON} or "none".

        Return ONLY a valid JSON object where:
            key   = original string
            value = chosen canonical label.
    """)

    msgs = [
        {"role": "system",
         "content": "You are a helpful toxicology assistant."},
        {"role": "user",
         "content": prompt + "\n" + json.dumps(cats, indent=2)}
    ]

    for _ in range(max_retry):
        rsp = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=msgs,
            temperature=0
        )
        raw = rsp.choices[0].message.content
        obj = extract_json(raw)
        if obj:
            return obj
        # ask again if not pure JSON
        msgs.append({"role": "assistant", "content": raw})
        msgs.append({"role": "user",
                     "content": "Please reply with ONLY the JSON object."})
    raise SystemExit("❌ GPT failed to return valid JSON after 3 attempts")

def canonicalise(pred_list, mapping):
    out = {f"Pred_{c.capitalize()}": None for c in CANON}
    for entry in pred_list:
        val = entry["value"]
        for cat in entry.get("property", {}).get("categories", []):
            lab = mapping.get(cat["category"].strip())
            if lab in CANON:
                key = f"Pred_{lab.capitalize()}"
                if out[key] is None:         # keep first hit
                    out[key] = val
    return out

# ──────────────────────────── main ─────────────────────────────
def main():
    if not openai.api_key:
        raise SystemExit("❌  OPENAI_API_KEY missing (put it in .env or export).")

    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, help="chemprop_preds.jsonl")
    ap.add_argument("--yaml",  default="chatgpt_property_map.yaml")
    ap.add_argument("--csv",   default="chemprop_canonical.csv")
    ap.add_argument("--id-col", default="cid", help="identifier column to copy")
    args = ap.parse_args()

    yml_path = pathlib.Path(args.yaml)
    if yml_path.exists():
        mapping = yaml.safe_load(open(yml_path))
    else:
        cats = collect_categories(args.jsonl)
        print(f"↻ GPT-4-o classifying {len(cats)} unique category strings …")
        mapping = gpt_map(cats)
        yaml.safe_dump(mapping, open(yml_path, "w"))
        print(f"✓ mapping saved → {yml_path}")

    # canonicalise rows
    rows = []
    with open(args.jsonl) as jf:
        for ln in tqdm(jf, desc="canonicalising"):
            rec = json.loads(ln)
            flat = canonicalise(rec["pred"], mapping)
            flat[args.id_col] = rec.get(args.id_col) or rec.get("cid")
            flat["inchi"]     = rec.get("inchi")
            rows.append(flat)

    pd.DataFrame(rows).to_csv(args.csv, index=False)
    print(f"✓ wrote {args.csv}  ({len(rows)} compounds)")

# ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
