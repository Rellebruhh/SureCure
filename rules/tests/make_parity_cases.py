"""Write verdicts from the Python reference engine for RulesParityTest (Kotlin).

Usage: python rules/tests/make_parity_cases.py
Reads data/fda/*.csv, writes App/src/test/resources/rules/parity_cases.tsv and
parity_barcodes.csv. Rerun after rebuilding the FDA tables or changing surecure_rules.py.

Columns (tab separated, \\N = None): tables, stage1_product, barcode, lot, reg_no, mfg_date,
exp_date, level, product_checked, reasons (joined with " || ").
tables = "real" (bundled tables, no barcodes) or "real+barcodes" (plus made-up EAN-13s, to
exercise the barcode rules until real ones are collected).
"""
import csv
import random
import shutil
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rules"))
from surecure_rules import FdaTables, ScanInput, verify  # noqa: E402

OUT = ROOT / "App/src/test/resources/rules"
TODAY = date(2026, 10, 24)
NONE = "\\N"


def ean(first12):
    total = sum(int(d) * (3 if i % 2 else 1) for i, d in enumerate(first12))
    return first12 + str((10 - total % 10) % 10)


FAKE_BARCODES = {ean("480000000001"): "Biogesic", ean("480000000002"): "Bioflu",
                 ean("480000000003"): "Flanax", ean("480000000004"): "Myra_E"}


def load(with_barcodes):
    tmp = Path(tempfile.mkdtemp())
    for f in ["registrations.csv", "counterfeit_lots.csv"]:
        shutil.copy(ROOT / "data/fda" / f, tmp / f)
    rows = ["ean13,product,pack_size,source_url"]
    if with_barcodes:
        rows += [f"{k},{v},TEST,made-up test barcode" for k, v in FAKE_BARCODES.items()]
    (tmp / "barcodes.csv").write_text("\n".join(rows) + "\n")
    return FdaTables.from_csv(tmp)


def variants(lot, rng):
    """Near misses a user or OCR might produce."""
    out = [lot, lot.lower(), " " + lot + " ", "-".join(lot), lot[:-1], lot + "7"]
    if len(lot) > 1:
        i = rng.randrange(len(lot))
        out.append(lot[:i] + ("0" if lot[i] != "0" else "1") + lot[i + 1:])
        out.append(lot[:i] + lot[i + 1:])
    out.append(lot.replace("0", "O").replace("1", "I"))
    return out


def main():
    rng = random.Random(20261024)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "parity_barcodes.csv").write_text(
        "ean13,product,pack_size,source_url\n"
        + "".join(f"{k},{v},TEST,made-up test barcode\n" for k, v in FAKE_BARCODES.items()))
    real = load(False)
    labels = ["Ascozin", "Bioflu", "Biogesic", "Bonamine", "Buscopan", "DayZinc", "Decolgen", "Flanax",
              "Imodium", "Lactezin", "Lagundi", "Midol", "Myra_E", "Neurogen_E", "Omeprazole", "Rinityn",
              "Rogin_E", "Sinecod", "Tempra", "Tuseran_Forte"]
    products = labels + sorted({r.product for r in real.registrations.values()} - set(labels)) \
        + sorted({c.product for c in real.counterfeit_lots} - set(labels))[:8] + [None, ""]
    regs_by_product = {}
    for r in real.registrations.values():
        regs_by_product.setdefault(r.product, []).append(r.reg_no)
    all_regs = sorted(real.registrations)
    lots_by_product = {}
    for c in real.counterfeit_lots:
        lots_by_product.setdefault(c.product, []).append(c.lot)
    all_lots = [c.lot for c in real.counterfeit_lots]
    barcodes = [None, "", "12345", ean("480000000099"), ean("480000000001")[:-1] + "0"] + list(FAKE_BARCODES)

    def date_pair():
        kind = rng.choice(["none", "none", "ok", "expired", "inverted", "exp_only", "today"])
        if kind == "none":
            return None, None
        if kind == "ok":
            return date(2025, rng.randint(1, 12), 1), date(2028, rng.randint(1, 12), 28)
        if kind == "expired":
            return date(2023, 1, 1), TODAY - timedelta(days=rng.randint(1, 900))
        if kind == "inverted":
            return date(2027, 5, 1), date(2027, 5, 1) - timedelta(days=rng.randint(0, 60))
        if kind == "exp_only":
            return None, TODAY + timedelta(days=rng.randint(-3, 3))
        return TODAY, TODAY

    cases = []
    # Every listed lot, exact, for its own product, no other codes.
    for c in real.counterfeit_lots:
        cases.append(("real", ScanInput(c.product, lot=c.lot)))
    # Random combinations.
    for _ in range(700):
        p = rng.choice(products)
        own_lots = lots_by_product.get(p or "", [])
        lot_pool = [None, "", "ZZ99999", "A1", rng.choice(all_lots)]
        if own_lots:
            lot_pool += variants(rng.choice(own_lots), rng) * 2
        own_regs = regs_by_product.get(p or "", [])
        reg_pool = [None, "", "DR-NOT-IN-TABLE", rng.choice(all_regs)]
        if own_regs:
            r = rng.choice(own_regs)
            reg_pool += [r, r, r.lower(), " " + r + " "]
        mfg, exp = date_pair()
        scan = ScanInput(p, barcode=rng.choice(barcodes), lot=rng.choice(lot_pool),
                         reg_no=rng.choice(reg_pool), mfg_date=mfg, exp_date=exp)
        cases.append((rng.choice(["real", "real+barcodes"]), scan))

    tables = {"real": real, "real+barcodes": load(True)}

    def cell(v):
        return NONE if v is None else str(v)

    with open(OUT / "parity_cases.tsv", "w", encoding="utf-8", newline="\n") as f:
        for name, s in cases:
            v = verify(s, tables[name], TODAY)
            fields = [name, s.stage1_product, s.barcode, s.lot, s.reg_no, s.mfg_date, s.exp_date,
                      v.level, v.product_checked, " || ".join(v.reasons)]
            assert not any("\t" in cell(x) or "\n" in cell(x) for x in fields)
            f.write("\t".join(cell(x) for x in fields) + "\n")
    levels = {}
    for name, s in cases:
        lv = verify(s, tables[name], TODAY).level
        levels[lv] = levels.get(lv, 0) + 1
    print(f"{len(cases)} cases written, levels {levels}")


if __name__ == "__main__":
    main()
