"""Turn the team's scraped FDA files into the tables the rules engine reads.

Inputs (data/fda/raw/):
  med_info.csv                                      registrations scraped for our brands
  counterfeit_drug_data_2022-03-01_to_present.csv   lots named in FDA counterfeit advisories

Outputs (data/fda/):
  registrations.csv     reg_no, product, ... (product = our model label, so every variant of a brand maps to it)
  counterfeit_lots.csv  product, lot, expiry, advisory, ... (deduplicated, one row per product + lot)
  rejected_lots.csv     rows we couldn't use (no lot, "Cannot be determined", ...)

Run: python scripts/build_fda_tables.py
"""
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "fda" / "raw"
OUT = ROOT / "data" / "fda"

# Brand (letters and digits only, lower case) -> model label. Checked in order; first prefix that matches wins.
# Products that share a brand name but are a different medicine get their own label instead.
NOT_OURS = {"flanaxace": "FlanaxACE (topical)", "flanaxplus": "FlanaxPlus (topical gel)",
            "tempracof": "Tempracof (ambroxol)", "tuserannight": "Tuseran Night (syrup)",
            "dayzincmax": "DayZincMax (food supplement)"}
OURS = [("ascozin", "Ascozin"), ("bioflu", "Bioflu"), ("biogesic", "Biogesic"), ("bonamine", "Bonamine"),
        ("buscopan", "Buscopan"), ("dayzinc", "DayZinc"), ("nodrowsedecolgen", "Decolgen"),
        ("decolgen", "Decolgen"), ("flanax", "Flanax"), ("imodium", "Imodium"), ("lactezin", "Lactezin"),
        ("ascof", "Lagundi"), ("lagundi", "Lagundi"), ("midol", "Midol"), ("myrae", "Myra_E"),
        ("neurogene", "Neurogen_E"), ("omeprazole", "Omeprazole"), ("rinityn", "Rinityn"),
        ("rogine", "Rogin_E"), ("sinecod", "Sinecod"), ("tempra", "Tempra"), ("tuseran", "Tuseran_Forte")]
# Spelling variants of brands outside our 20, so lots of one brand are matched together.
OTHER_ALIASES = {"alxanfr": "alaxanfr", "alaxan": "alaxanfr"}

# Our suggested "classic" registration per product (the pack we expect people to photograph).
# These are guesses from the product names; check each against the Capsure pack photos before relying on them.
CLASSIC = {"Ascozin": "DR-XY46665", "Bioflu": "DR-XY34482", "Biogesic": "DR-153", "Bonamine": "DR-XY22028",
           "Buscopan": "DR-XY41205", "Decolgen": "DR-XY41124", "Flanax": "DR-XY46696", "Imodium": "DR-XY14916",
           "Lactezin": "DR-XY44068", "Lagundi": "HMRP-21", "Midol": "DR-XY30477", "Rinityn": "DRP-6907",
           "Sinecod": "DRP-8129", "Tempra": "DR-4396", "Tuseran_Forte": "DR-XY29720"}

PLACEHOLDER_LOTS = {"", "CANNOTBEDETERMINED", "NOTINDICATED"}
MIN_FUZZY_LOT_LEN = 6   # same rule as rules/surecure_rules.py


def key(s):
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def label_for(brand):
    k = key(brand)
    for prefix, lab in NOT_OURS.items():
        if k.startswith(prefix):
            return lab, False
    for prefix, lab in OURS:
        if k.startswith(prefix):
            return lab, True
    return None, False


def normalize_lot(lot):
    return "".join(ch for ch in (lot or "").upper() if ch.isalnum())


def expiry_iso(s):
    m = re.fullmatch(r"\s*(\d{1,2})\s*/\s*(\d{4})\s*", s or "")
    return f"{m.group(2)}-{int(m.group(1)):02d}" if m else ""


def build_registrations():
    rows = []
    with open(RAW / "med_info.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            lab, ours = label_for(r["Brand Name"])
            reg = r["Registration Number"].strip().upper()
            rows.append({
                "reg_no": reg,
                "product": lab or r["Brand Name"].strip(),
                "brand_name": r["Brand Name"].strip(),
                "generic_name": r["Generic Name / Product Name"].strip(),
                "strength_form": r["Dosage Strength and Form"].strip(),
                "classification": r["Classification"].strip(),
                "reg_type": re.match(r"[A-Z]+", reg).group(0) if reg else "",
                "photo_supported": "yes" if ours else "no",
                "classic": "suggested" if CLASSIC.get(lab) == reg else "",
                "valid_from": "", "valid_to": "",
                "source": "FDA Philippines Verification Portal, scraped by the team",
            })
    with open(OUT / "registrations.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows


def display_names(path):
    # For brands outside our 20, show the most common spelling (with spaces) rather than the matching key.
    spell = defaultdict(Counter)
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            b = r["brand name"].strip()
            spell[OTHER_ALIASES.get(key(b), key(b))][b] += 1
    return {k: max(c, key=lambda b: (" " in b, c[b])) for k, c in spell.items()}


def build_counterfeit_lots():
    groups, rejected = defaultdict(list), []
    src = RAW / "counterfeit_drug_data_2022-03-01_to_present.csv"
    names = display_names(src)
    with open(src, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            brand = r["brand name"].strip()
            lot = normalize_lot(r["lot number"])
            if lot in PLACEHOLDER_LOTS:
                rejected.append({**r, "reason": "no usable lot number"})
                continue
            lab, ours = label_for(brand)
            if lab is None:
                k = OTHER_ALIASES.get(key(brand), key(brand))
                lab = names.get(k) if k else f"(no brand) {r['generic name'].strip()}"
            groups[(lab, lot)].append((r, ours))
    rows = []
    for (lab, lot), items in sorted(groups.items()):
        raws = [r for r, _ in items]
        brands = Counter(r["brand name"].strip() for r in raws)
        rows.append({
            "product": lab,
            "lot": raws[0]["lot number"].strip(),
            "lot_normalized": lot,
            "expiry": ";".join(sorted({expiry_iso(r["expiry date"]) for r in raws} - {""})),
            "advisory": "",
            "brand_as_listed": brands.most_common(1)[0][0],
            "generic_name": raws[0]["generic name"].strip(),
            "strength_form": raws[0]["dosage and strength"].strip(),
            "photo_supported": "yes" if items[0][1] else "no",
            "fuzzy_match": "yes" if len(lot) >= MIN_FUZZY_LOT_LEN else "exact only",
            "times_listed": len(raws),
        })
    with open(OUT / "counterfeit_lots.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(OUT / "rejected_lots.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rejected[0]))
        w.writeheader()
        w.writerows(rejected)
    return rows, rejected


if __name__ == "__main__":
    regs = build_registrations()
    lots, rej = build_counterfeit_lots()
    print(f"registrations: {len(regs)} rows, {sum(r['photo_supported'] == 'yes' for r in regs)} for our products")
    print(f"counterfeit lots: {len(lots)} unique product+lot, "
          f"{sum(r['photo_supported'] == 'yes' for r in lots)} for our products; {len(rej)} rows rejected")
