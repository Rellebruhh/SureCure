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
DayZincMax, a food supplement, was found), and the EAN-13 barcodes (`barcodes.csv`).

Rebuild after editing the raw files: `python scripts/build_fda_tables.py`.
