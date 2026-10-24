"""Build the app's drugstore asset from the checked branch list.

Usage: python scripts/build_drugstore_asset.py
Reads  data/drugstores/ph_drugstores_checked.csv (built by scripts/check_drugstores.py)
Writes App/src/main/assets/drugstores/branches.csv

Keeps only rows with coord_precision = exact: the only rows that may be shown as a location.
municipality_centroid rows (city known, point is the city centre) and rows with no location
are left out. Source order is kept, so the output is the same on every run.

PSGC codes are written as 10-digit strings. check_drugstores.py currently writes regions 01 to 09
without the leading zero (Bacarra: 102802000 instead of 0102802000) because the boundary file
stores adm3_psgc as a number; this pads them back. The upstream fix is to pad psgc_final in
check_drugstores.py, after which this padding does nothing.
"""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/drugstores/ph_drugstores_checked.csv"
OUT = ROOT / "App/src/main/assets/drugstores/branches.csv"
COLUMNS = ["franchise", "branch", "address", "lat", "lon", "psgc", "municipality", "province"]


def main():
    with open(SRC, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    kept = []
    for r in rows:
        if r["coord_precision"] != "exact":
            continue
        psgc = r["psgc_final"].strip()
        if not psgc.isdigit() or len(psgc) > 10:
            sys.exit(f"unexpected PSGC code {psgc!r} for {r['Franchise Name']} {r['Branch Name']}")
        lat, lon = float(r["lat_final"]), float(r["lon_final"])
        if not (4.0 <= lat <= 21.5 and 116.0 <= lon <= 127.0):
            sys.exit(f"coordinate outside the Philippines for {r['Franchise Name']} {r['Branch Name']}")
        kept.append([r["Franchise Name"].strip(), r["Branch Name"].strip(), r["Exact Address"].strip(),
                     r["lat_final"].strip(), r["lon_final"].strip(), psgc.zfill(10),
                     r["municity_final"].strip(), r["province_final"].strip()])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(COLUMNS)
        w.writerows(kept)
    padded = sum(1 for r in rows if r["coord_precision"] == "exact" and len(r["psgc_final"].strip()) < 10)
    print(f"{len(kept)} of {len(rows)} branches written (coord_precision = exact); "
          f"{padded} PSGC codes padded to 10 digits; {OUT.stat().st_size / 1024:.0f} KB -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
