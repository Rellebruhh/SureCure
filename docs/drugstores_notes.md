# Drugstore data in the app

Chain drugstore branches (Mercury Drug, Watsons, Generika, TGP) for "Find a nearby drugstore".
**Not an FDA license list.** Any screen showing it must say: "Chain drugstore branches. Not a
complete list of licensed pharmacies." (`DRUGSTORE_DISCLAIMER`). Never call them licensed or verified.

## Data

- `data/drugstores/raw/ph_drugstores.csv`: the scraped list (4,252 branches), unchanged.
- `data/drugstores/ph_drugstores_checked.csv`, `drugstores_per_municity.csv`, `needs_review.csv`,
  `README.md`: built by `scripts/check_drugstores.py` against the 2023 PSGC boundaries. Never edit by hand.
- `App/src/main/assets/drugstores/branches.csv`: the app asset, built by
  `python scripts/build_drugstore_asset.py`. **684 KB** (about 168 KB compressed in the APK).
  - Only `coord_precision = exact` rows: 3,651 branches. The 568 `municipality_centroid` rows (city
    known, point is the city centre) and 33 rows with no location are not bundled, so nothing in the
    app can show a city centre as a branch location. None of the 66 `needs_review.csv` rows is exact.
  - Columns: `franchise, branch, address, lat, lon, psgc, municipality, province`.
  - Southstar has no exact locations (all 551 lack coordinates), so it never appears.

## PSGC codes: upstream fix needed

`check_drugstores.py` writes regions 01 to 09 without the leading zero (Bacarra `102802000`; the
PSGC is `0102802000`): the boundary files store `adm3_psgc` as a number. 2,075 of the 3,651 bundled
branches are affected. Inside our own files the codes still match, but a join with an official PSGC
list or a future `psgc`-keyed table would miss them.

- The asset script pads them to 10 digits, so the app is correct now.
- **Upstream fix (for the data person):** write `psgc_final` (and `psgc_municity`) as
  `str(code).zfill(10)` in `check_drugstores.py`, then rebuild. After that the padding in
  `build_drugstore_asset.py` does nothing. `drugstores_per_municity.csv` has the same issue.

## Code (`com.mlbb.surecure.drugstores`, pure Kotlin)

- `DrugstoreDirectory.nearestBranches(lat, lon, k = 10)`: haversine distance over the bundled list,
  nearest first. Computed on the phone; the user's location is never stored or sent.
- For the picker when location permission is refused: `provinces()`, `municipalities(province)`,
  `findMunicipalities(query)`, `branchesIn(psgc)`.
  Highly urbanized cities are their own "province" in the PSGC (Cebu City is filed under "Cebu City",
  not "Cebu"; same for Davao, Lapu-Lapu and others), so the picker should offer a city search, not
  only province then city. NCR names are long ("NCR, Second District (Not a Province)") and shown as is.
- `DrugstoreLoader.load(context)`: reads the asset once (off the main thread).

## Tests (2026-10-24)

`DrugstoresTest`, 8/8 (JVM). Expected nearest branches were computed separately in Python on the
same file. A point at Quezon Memorial Circle returns 10 Quezon City branches, nearest `TGP #189` at
0.425 km; Cebu City and Davao City points return their own city's branches; haversine checks;
k edge cases; asset has 3,651 rows, all inside the Philippines, all PSGC codes 10 digits, no
`needs_review.csv` row; picker helpers including the Cebu City case.

## Not done yet

- **"Find a nearby drugstore" on amber and red results:** done in `ui/result/NearbyDrugstores.kt`.
  One approximate reading through Android's `LocationManager` (`ACCESS_COARSE_LOCATION` only, network
  provider, 15 s limit, else the last known reading); nothing is stored or sent. If permission is refused
  or no location comes back, the user searches a city. A list with the disclaimer; no map library.
- **Step 5, reports by municipality:** skipped, no backend in the project.
