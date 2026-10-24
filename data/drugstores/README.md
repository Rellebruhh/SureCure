# Drugstore branches: checked against PSGC boundaries

`raw/ph_drugstores.csv` is the scraped list of 4,252 chain branches (Mercury Drug, Watsons, Generika, Southstar,
TGP). It is **not** an FDA license list; the FDA establishment portal couldn't be reached, so label it in the app
as "chain drugstore branches".

`scripts/check_drugstores.py` checks every branch against the 2023 PSGC city/municipality boundaries
(faeldon/philippines-json-maps, MIT licence):

| qa_status | Meaning | Count |
|---|---|---|
| `ok` | point is inside the city named in the address | 3,568 |
| `ok_border` | point is in a neighbouring city, within 3 km of the named one | 67 |
| `ok_unverified` | point is on land, but no city name was recognized in the address | 16 |
| `no_coordinates` | no coordinate (all 551 Southstar branches, plus 14 others) | 565 |
| `city_mismatch` | point is more than 3 km from the city in the address (wrong geocode) | 25 |
| `not_on_land` | point is more than 1.5 km from any boundary (water or wrong place) | 7 |
| `shared_coordinate` | 3+ branches on one exact point and the city can't be confirmed | 4 |

Outputs:
- `ph_drugstores_checked.csv`: every original column plus the QA columns and a final placement:
  `psgc_final`, `municity_final`, `province_final`, `lat_final`, `lon_final`, `psgc_source`
  (`point` = trusted coordinate, `address` = city taken from the address when the point was missing or wrong,
  `unresolved`) and `coord_precision` (`exact` for 3,651, `municipality_centroid` for 568, `none` for 33).
- `drugstores_per_municity.csv`: branch counts per city/municipality (PSGC code) and chain, 545 cities and
  municipalities, 4,219 branches placed. This is the table for municipal-level maps and the access-gap model.
- `needs_review.csv`: 66 rows for a person to fix (25 mismatches, 7 off-land points, 4 shared points, 30 unresolved
  addresses). One known false alarm: the Plaridel branch at "Banga I" (Banga I is a barangay of Plaridel).

Rules of use:
- Pins on a map: only `coord_precision = exact`.
- Counts per municipality: everything with a `psgc_final`.
- Several Watsons outlets in one SM mall share a point; that's real, not an error.

Rebuild: `python scripts/check_drugstores.py data/drugstores/raw/ph_drugstores.csv <path to philippines-json-maps>/2023/geojson data/drugstores`
(needs `git clone --depth 1 --filter=blob:none --sparse https://github.com/faeldon/philippines-json-maps` with
`2023/geojson/regions/medres`, `provdists/medres` and `municities/medres` checked out).
