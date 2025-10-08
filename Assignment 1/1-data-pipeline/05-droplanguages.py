import argparse, os, sys, re
import pandas as pd

DEF_IN  = os.path.join("..","data","openfoodfacts","step4.csv")
DEF_OUT = os.path.join("..","data","openfoodfacts","step5.csv")

EN_PAT      = r'(^|,\s*)en(?:-[a-z0-9]{2,8})?:'                                  # basically a row can have en: or many
NON_EN_PAT  = r'(^|,\s*)(?!en(?:-[a-z0-9]{2,8})?:)[a-z]{2,3}(?:-[a-z0-9]{2,8})?:' # detect non-English prefixes, like different foreign character or number
SPLIT_RE    = re.compile(r"\s*,\s*")
DIGIT_RE    = re.compile(r"\d")
ASCII_LET   = re.compile(r"[A-Za-z]")

NON_ASCII_OR_DISALLOWED = re.compile(r"[^a-z:,\-\s]")


       # i was running out of idea so i asked gpt what were the most used foreign words in my dataset, and i putte them in a constant

GERMAN_COMMON = {
    "zucker","weizen","weizenmehl","milch","vollmilch","butter","sahne","ei","eier",
    "kakaobutter","kakao","schokolade","zartbitter","zitronensaft","zitronensäure",
    "zitronensaftkonzentrat","zwiebel","zwiebeln","pflanzenöl","pflanzenfett","rapsöl",
    "sonnenblumenöl","glukosesirup","dextrose","vollkorn","salz","backtriebmittel",
    "emulgator","emulgatoren","farbstoff","aroma","aromen","säurungsmittel",
    "stabilisator","stabilisatoren","verdickungsmittel","geliermittel","gerste",
    "molke","molkenpulver","magermilchpulver","butterreinfett","feuchthaltemittel",
    "gewürz","gewürze","gewürzextrakt","vanillin","zutaten","zubereitung"
}
GERMAN_NEEDLES = ("sch", "zucker", "zwiebel", "zitron")  

DUTCH_COMMON = {
    "zout","zwarte","zwarte-bes","zwarte-wortel","zonnebloem","zonnebloemolie",
    "zonnebloemlecithine","zonnebloemlecithines","zetmeel","zuurteregelaar",
    "zuurteregelaars","zuurte","zuursel","zoetstof","zoete-aardappel","zeezout",
    "zichorien-extrakt","zichorienwurzelfaser","zout-powder","zie-deksel","zie-opdruk",
    "zien","zutaten-vollmilch",
}
DUTCH_NEEDLES = ("ij", "zout", "zoet", "zonnebloem", "zuurteregela", "aardappel", "lecithine")

def looks_german_or_dutch(cell: str) -> bool:

    if not cell:
        return False
    for tok in SPLIT_RE.split(cell):
        t = tok.strip()
        if not t:
            continue
        # strip an optional language prefix like xx: or xx-YY:
        t = re.sub(r"^[a-z]{2,3}(?:-[a-z0-9]{2,8})?:", "", t)
        if not t:
            continue

        # Dutch heuristics
        if t in DUTCH_COMMON:
            return True
        if any(nd in t for nd in DUTCH_NEEDLES):
            return True

    return False

def remove_digit_tags(cell: str) -> str:
    #Remove any individual tag that contains a digit. Return cleaned, comma-joined string.
    if not isinstance(cell, str) or not cell:
        return cell
    parts = [p.strip() for p in SPLIT_RE.split(cell) if p.strip()]
    kept  = [p for p in parts if not DIGIT_RE.search(p)]
    return ",".join(kept)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  default=DEF_IN,  )
    ap.add_argument("--output", default=DEF_OUT, )
    ap.add_argument("--col",    default="ingredients_tags")
    args = ap.parse_args()

    if not os.path.isfile(args.input):
        print(f"Input not found: {args.input}", file=sys.stderr); sys.exit(2)

    df = pd.read_csv(args.input, low_memory=False, dtype=str)
    if args.col not in df.columns:
        print(f"Column '{args.col}' not found. Available: {list(df.columns)}", file=sys.stderr); sys.exit(2)

    # Normalize to lowercase and commas
    s = df[args.col].fillna("").astype(str).str.lower()
    s = (s
         .str.replace("：", ":", regex=False)
         .str.replace("；", ";", regex=False)
         .str.replace("、", ",", regex=False)
         .str.replace("|", ",", regex=False)
         .str.replace(";", ",", regex=False))

    # English-only mask (prefix-based)
    has_en     = s.str.contains(EN_PAT,     regex=True, na=False)
    has_non_en = s.str.contains(NON_EN_PAT, regex=True, na=False)
    english_only_mask = has_en & ~has_non_en

    # Remove digit-containing tags for rows that pass English-only test
    cleaned = s.where(~english_only_mask, s.map(remove_digit_tags))

    # ASCII-only after cleaning
    has_disallowed = cleaned.str.contains(NON_ASCII_OR_DISALLOWED, regex=True, na=False)

    # German/Dutch lexical/diacritic detection after cleaning
    ger_nl_mask = cleaned.map(looks_german_or_dutch)

    # Must still have at least one ASCII letter and not be empty
    has_ascii_letter_after = cleaned.str.contains(ASCII_LET, na=False)
    non_empty_after = cleaned.str.strip().ne("")

    keep_mask = english_only_mask & ~has_disallowed & ~ger_nl_mask & has_ascii_letter_after & non_empty_after

    # Stats

    print(f"Dropped                  : {len(df) - int(keep_mask.sum()):,}")

    out = df.loc[keep_mask].copy()
    out[args.col] = cleaned.loc[keep_mask]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Saved: {args.output} (rows: {len(out):,})")

if __name__ == "__main__":
    main()
