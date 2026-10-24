# FDA tables used by the rules engine

`raw/` holds the two files our team scraped, unchanged. `scripts/build_fda_tables.py` turns them into the tables
`rules/surecure_rules.py` (and later the Kotlin port) reads:

- `registrations.csv`: 66 registration numbers for our brands. `product` is the model label (for example every
  Decolgen variant maps to `Decolgen`), so a Forte or No-Drowse pack isn't flagged as "registration belongs to
  another product". `classic = suggested` marks the variant we think the Capsure photos show; check it against the
  pack photos. No validity dates were scraped, so the date check is skipped for now.
- `counterfeit_lots.csv`: 327 unique product + lot pairs from FDA counterfeit advisories (March 2022 to present),
  122 of them for our products. Duplicates are merged (`times_listed`), spellings are unified (AlaxanFR, Alxan FR,
  Alaxan FR → Alaxan FR), expiry is `YYYY-MM` (several per lot when advisories disagree). Lots under 6 characters
  (`015`, `260`, `670`, ...) only match exactly.
- `rejected_lots.csv`: 5 rows without a usable lot ("Cannot be determined", "Not Indicated", blank).

Still missing: the advisory number and URL for each lot (`advisory` is blank, so the app says "an FDA counterfeit
advisory" instead of naming it), registrations for Myra E, Neurogen E, Rogin E, Omeprazole and DayZinc (only
DayZincMax, a food supplement, was found), and barcodes for Ascozin, Lactezin, Neurogen E,
Rinityn, Rogin E, Omeprazole, plain Buscopan 10 mg and the regular Imodium pack.

Rebuild after editing the raw files: `python scripts/build_fda_tables.py`.

`barcodes.csv` is collected by hand, not built by the script: 18 EAN-13 codes for 14 of our 20 products,
from online pharmacy listings (2026-10-24), all passing the check digit. Most are box or promo packs; a retail
strip may carry a different code. The engine reads only `ean13` and `product`; `pack_size`, `source_url`,
`retrieved` and `notes` are for people. Add codes only from real packs or listings, never guessed.
