import argparse, csv, os, re, sys
import pandas as pd
import numpy as np

#not really optimized but anyway it works

BLOCKED_GROUPS = [
    "dressings and sauces",
    "cereals",
    "cheese",
    "one dish meals",
    "processed meat",
    "milk and yogurt",
    "unsweetened beverages",
    "sweetened beverages",
    "alcoholic beverages",
    "waters and flavored waters",
    "fruit juices",
    "fruit nectars",
    "plant based milk substitutes",
    "teas and herbal teas and coffees",
    "vegetables",
    "fruits",
    "fruits and vegetables",
    "cereals and potatoes",
    "potatoes",
    "bread",
    "soups",
    "sandwiches",
    "pizza pies and quiches",
    "meat other than poultry",
    "poultry",
    "fish and seafood",
    "fatty fish",
    "lean fish",
    "fish meat eggs",
    "legumes",
    "eggs",
    "fats",
    "offals",
    "pgi",
    "green dot",
    "with sulfites",
    "6.3",
    "point vert",
    "fabrique en italie",
    "triman",
    "dairy-desserts",
    "artificially-sweetened-beverages",
    "breakfast-cereals",
    "ice-cream",
    "salty-and-fatty-products",
]

DEF_IN  = os.path.join("..","data","openfoodfacts","step2.csv")
DEF_OUT = os.path.join("..","data","openfoodfacts","step3.csv")

p = argparse.ArgumentParser(description="Drop rows whose food_groups matches any blocked group.")
p.add_argument("--input", default=DEF_IN)
p.add_argument("--output", default=DEF_OUT)
p.add_argument("--col", default="food_groups")
p.add_argument("--chunksize", type=int, default=200_000)
p.add_argument("--sep", default="")
p.add_argument("--multi-sep", default=r"[|,]")
a = p.parse_args()

def autodetect_sep(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        try: return csv.Sniffer().sniff(f.read(65536), delimiters=[",","\t",";","|"]).delimiter
        except: return "\t"

def norm(s): return re.sub(r"^[a-z]{2}:", "", (s or "").strip().lower()).replace("-"," ")

blocked = {norm(g) for g in BLOCKED_GROUPS if str(g).strip()}
if not blocked:
    print("No BLOCKED_GROUPS specified; nothing to drop.")
    sys.exit(0)

if not os.path.isfile(a.input): print(f"Input not found: {a.input}", file=sys.stderr) or sys.exit(2)
sep = a.sep or autodetect_sep(a.input)
os.makedirs(os.path.dirname(a.output) or ".", exist_ok=True)
if os.path.exists(a.output): os.remove(a.output)

total_in=total_out=total_drop=0; wrote=False
split_re = re.compile(a.multi_sep)

for chunk in pd.read_csv(a.input, sep=sep, chunksize=a.chunksize, dtype=str, on_bad_lines="skip", low_memory=False):
    total_in += len(chunk)
    if a.col not in chunk.columns: print(f"Missing column: {a.col}", file=sys.stderr) or sys.exit(2)

    s = chunk[a.col].astype("string").str.strip().replace({"":pd.NA,"nan":pd.NA,"NaN":pd.NA,"[]":pd.NA,"{}":pd.NA})
    def should_drop(v):
        if v is pd.NA: return False
        toks = {norm(t) for t in split_re.split(str(v)) if t.strip()}
        return bool(toks & blocked)

    mask = s.map(should_drop)
    kept = chunk.loc[~mask]
    total_drop += int(mask.sum()); total_out += len(kept)
    if len(kept):
        kept.to_csv(a.output, index=False, mode=("w" if not wrote else "a"), header=(not wrote))
        wrote = True

print(f"Rows read: {total_in:,} | kept: {total_out:,} | dropped: {total_drop:,} ({(total_drop/total_in*100 if total_in else 0):.1f}%)")
