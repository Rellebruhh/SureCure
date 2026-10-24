"""Quality check of ph_drugstores.csv against 2023 PSGC boundaries (faeldon/philippines-json-maps).

For every branch: does it have a coordinate, is it in the Philippines, which city/municipality contains it, and does
that agree with the city written in the address? Writes ph_drugstores_checked.csv (original columns + QA columns).

Usage: python check_drugstores.py <ph_drugstores.csv> <.../2023/geojson> <out_dir>
"""
import glob
import json
import math
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.path import Path as MPath

SRC, MAPS, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])

# Highly urbanized cities and the BARMM special area are not in the province files; names from the PSGC.
EXTRA_NAMES = {330100000: "Angeles City", 331400000: "Olongapo City", 431200000: "Lucena City",
               630200000: "Bacolod City", 631000000: "Iloilo City", 730600000: "Cebu City",
               731100000: "Lapu-Lapu City", 731300000: "Mandaue City", 831600000: "Tacloban City",
               931700000: "Zamboanga City", 1030500000: "Cagayan de Oro City", 1030900000: "Iligan City",
               1230800000: "General Santos City", 1430300000: "Baguio City", 1630400000: "Butuan City",
               1731500000: "Puerto Princesa City"}
ALIASES = {"baliuag": "baliwag", "kalookan": "caloocan", "dasmarinas cavite": "dasmarinas", "gensan": "general santos",
           "cdo": "cagayan de oro", "sto tomas": "santo tomas", "lapulapu": "lapu lapu",
           "martirez": "martires", "zaragosa": "zaragoza", "borobo": "barobo", "mlang": "m lang",
           "montalban": "rodriguez", "fernado": "fernando", "governor mariano alvarez": "general mariano alvarez",
           "mataas na kahoy": "mataasnakahoy", "q c": "quezon city", "capaz": "capas"}
STREET_WORDS = re.compile(r"\b(st|street|ave|avenue|road|rd|hwy|highway|blvd|boulevard|drive|dr|cor|corner|ext|"
                          r"extension|bldg|building|mall|plaza|lot|blk|block|purok|sitio|national)\b")
COASTAL_KM = 1.5     # points this close to a boundary count as on land (polygons are simplified at the coast)
MISMATCH_KM = 3.0    # beyond this distance from the address city the point is a real discrepancy


def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\bsto\.?\s", "santo ", s)
    s = re.sub(r"\bsta\.?\s", "santa ", s)
    s = re.sub(r"\bgen\.?\s", "general ", s)
    s = re.sub(r"\(.*?\)", " ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\bcity of\b", " ", s)
    s = re.sub(r"\b(?<!quezon )city\b", " ", s)       # "Quezon City" keeps "city" so it isn't the province
    s = " ".join(s.split())
    for a, b in ALIASES.items():
        s = re.sub(rf"\b{a}\b", b, s)
    return s


def km(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(a))


# --- boundaries: one entry per city/municipality, built from its barangay polygons -----------------
prov_name, mun_name = {}, dict(EXTRA_NAMES)
for f in glob.glob(str(MAPS / "regions/medres/*.json")):
    for ft in json.load(open(f))["features"]:
        prov_name[ft["properties"]["adm2_psgc"]] = ft["properties"]["adm2_en"]
for f in glob.glob(str(MAPS / "provdists/medres/*.json")):
    for ft in json.load(open(f)).get("features", []):
        mun_name[ft["properties"]["adm3_psgc"]] = ft["properties"]["adm3_en"]

munis = {}
# City/municipality outlines from the province files first (includes the City of Manila, which has no barangay file)
for f in glob.glob(str(MAPS / "provdists/medres/*.json")):
    for ft in json.load(open(f)).get("features", []):
        p, g = ft["properties"], ft["geometry"]
        if g is None:
            continue
        m = munis.setdefault(p["adm3_psgc"], {"psgc": p["adm3_psgc"], "prov": p["adm2_psgc"], "rings": []})
        polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        m["rings"] += [np.asarray(poly[0]) for poly in polys]
from_provdists = set(munis)
# Then the cities missing there (highly urbanized cities, BARMM special area), from their barangay polygons
for f in glob.glob(str(MAPS / "municities/medres/*.json")):
    for ft in json.load(open(f)).get("features", []):
        p, g = ft["properties"], ft["geometry"]
        if g is None or p["adm3_psgc"] in from_provdists:
            continue
        m = munis.setdefault(p["adm3_psgc"], {"psgc": p["adm3_psgc"], "prov": p["adm2_psgc"], "rings": []})
        polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        m["rings"] += [np.asarray(poly[0]) for poly in polys]
for m in munis.values():
    allpts = np.vstack(m["rings"])
    m["paths"] = [MPath(r) for r in m["rings"]]
    m["bbox"] = (*allpts.min(0), *allpts.max(0))
    m["cen"] = allpts.mean(0)
    m["name"] = mun_name.get(m["psgc"], str(m["psgc"]))
    m["province"] = prov_name.get(m["prov"]) or m["name"]
    m["radius_km"] = 0.5 * km(m["bbox"][1], m["bbox"][0], m["bbox"][3], m["bbox"][2]) / math.sqrt(2)
munis = list(munis.values())
verts = np.vstack([np.c_[np.vstack(m["rings"]), np.full(sum(len(r) for r in m["rings"]), i)]
                   for i, m in enumerate(munis)])
by_norm = {}
for m in munis:
    by_norm.setdefault(norm(m["name"]), []).append(m)
province_norms = {norm(v) for v in prov_name.values() if v}


def locate(lon, lat):
    for m in munis:
        x0, y0, x1, y1 = m["bbox"]
        if x0 <= lon <= x1 and y0 <= lat <= y1 and any(pa.contains_point((lon, lat)) for pa in m["paths"]):
            return m, 0.0
    # Not inside any polygon: nearest boundary vertex (simplified coastlines cut off seaside branches)
    dlat = (verts[:, 1] - lat) * 111.0
    dlon = (verts[:, 0] - lon) * 111.0 * math.cos(math.radians(lat))
    j = int(np.argmin(dlat ** 2 + dlon ** 2))
    return munis[int(verts[j, 2])], float(math.hypot(dlat[j], dlon[j]))


def address_cities(addr):
    parts = [p for p in str(addr).split(",") if p.strip()]
    if len(parts) > 2:
        parts = parts[1:]                                   # first part is the building or street
    parts = [p for p in parts if not STREET_WORDS.search(norm(p) + " ") or "city" in p.lower()]
    text = " " + " ".join(norm(p) for p in parts) + " "
    hits = [n for n in by_norm if len(n) >= 3 and f" {n} " in text]
    hits = [h for h in hits if not any(h != o and f" {h} " in f" {o} " for o in hits)]
    city_hits = [h for h in hits if h not in province_norms or len(by_norm[h]) == 1 and False]
    return city_hits or hits, text


MANILA = {norm(n) for n in ["Tondo I / II", "Binondo", "Quiapo", "San Nicolas", "Santa Cruz", "Sampaloc", "San Miguel",
                            "Ermita", "Intramuros", "Malate", "Paco", "Pandacan", "Port Area", "Santa Ana",
                            "San Andres", "Santa Mesa", "Manila", "City of Manila"]}

# --- check every branch ---------------------------------------------------------------------------
d = pd.read_csv(SRC)
shared = Counter(zip(d["Latitude"].round(5), d["Longitude"].round(5)))
out = []
for _, r in d.iterrows():
    o = {"qa_status": "", "qa_detail": "", "psgc_municity": "", "municity": "", "province": "",
         "address_city_found": "", "km_off": ""}
    lat, lon = r["Latitude"], r["Longitude"]
    guesses, text = address_cities(r["Exact Address"])
    o["address_city_found"] = "; ".join(guesses)
    if pd.isna(lat) or pd.isna(lon):
        o["qa_status"], o["qa_detail"] = "no_coordinates", "needs geocoding from the address"
        out.append(o)
        continue
    if not (4.5 <= lat <= 21.5 and 116 <= lon <= 127):
        swapped = 4.5 <= lon <= 21.5 and 116 <= lat <= 127
        o["qa_status"] = "outside_philippines"
        o["qa_detail"] = "latitude and longitude look swapped" if swapped else "coordinate outside the Philippines"
        out.append(o)
        continue
    m, off = locate(lon, lat)
    o.update(psgc_municity=m["psgc"], municity=m["name"], province=m["province"])
    if off > COASTAL_KM:
        o["qa_status"], o["km_off"] = "not_on_land", round(off, 1)
        o["qa_detail"] = f"{off:.1f} km from the nearest boundary (in water or outside the map)"
        out.append(o)
        continue
    mn = norm(m["name"])
    ok = {mn} | (MANILA if mn in MANILA else set())
    full = " " + norm(r["Exact Address"]) + " "
    in_address = any(f" {n} " in full for n in ok)
    if in_address:
        o["qa_status"] = "ok"
    elif not guesses:
        o["qa_status"], o["qa_detail"] = "ok_unverified", "no city name recognized in the address"
    else:
        dist = min(max(0.0, km(lat, lon, c["cen"][1], c["cen"][0]) - c["radius_km"])
                   for g in guesses for c in by_norm[g])
        o["km_off"] = round(dist, 1)
        if dist > MISMATCH_KM:
            o["qa_status"] = "city_mismatch"
            o["qa_detail"] = f"point is in {m['name']} ({m['province']}); address names {', '.join(guesses)}"
        else:
            o["qa_status"], o["qa_detail"] = "ok_border", f"point is in {m['name']}, next to {', '.join(guesses)}"
    if off > 0 and o["qa_status"] == "ok":
        o["qa_detail"] = f"just off the simplified coastline ({off:.1f} km)"
    n_same = shared[(round(lat, 5), round(lon, 5))]
    if n_same >= 3:
        # Usually several outlets in one mall (SM, Robinsons). Only a problem if the address can't confirm the city.
        note = f"{n_same} branches share this exact point"
        if o["qa_status"] in ("ok_unverified", "ok_border"):
            o["qa_status"], o["qa_detail"] = "shared_coordinate", note + " (possibly a city-centre geocode)"
        else:
            o["qa_detail"] = (o["qa_detail"] + "; " if o["qa_detail"] else "") + note
    out.append(o)

res = pd.concat([d, pd.DataFrame(out)], axis=1)


# --- final city/municipality per branch, for mapping and per-municipality counts --------------------
def resolve_from_address(addr):
    full = " " + norm(addr) + " "
    guesses, _ = address_cities(addr)
    if not guesses:
        return None
    # A city whose province is also written in the address wins (Pantukan, Davao de Oro beats "Davao")
    in_prov_all = {c["psgc"]: c for g in guesses for c in by_norm[g] if f" {norm(c['province'])} " in full}
    if len(in_prov_all) == 1:
        return next(iter(in_prov_all.values()))
    order = sorted(guesses, key=lambda g: full.rfind(f" {g} "), reverse=True)   # city usually comes last
    for g in order:
        cands = by_norm[g]
        in_prov = [c for c in cands if f" {norm(c['province'])} " in full]
        if len(cands) == 1:
            return cands[0]
        if len(in_prov) == 1:
            return in_prov[0]
    return None


final = []
for _, r in res.iterrows():
    good_point = r["qa_status"] in ("ok", "ok_border", "ok_unverified")
    f = {"psgc_final": "", "municity_final": "", "province_final": "", "psgc_source": "unresolved",
         "lat_final": "", "lon_final": "", "coord_precision": "none"}
    if good_point:
        f.update(psgc_final=r["psgc_municity"], municity_final=r["municity"], province_final=r["province"],
                 psgc_source="point", lat_final=r["Latitude"], lon_final=r["Longitude"], coord_precision="exact")
    else:
        m = resolve_from_address(r["Exact Address"])
        if m is not None:
            f.update(psgc_final=m["psgc"], municity_final=m["name"], province_final=m["province"],
                     psgc_source="address", lat_final=round(m["cen"][1], 6), lon_final=round(m["cen"][0], 6),
                     coord_precision="municipality_centroid")
    final.append(f)
res = pd.concat([res, pd.DataFrame(final)], axis=1)
counts = (res[res["psgc_final"] != ""].groupby(["psgc_final", "municity_final", "province_final", "Franchise Name"])
          .size().unstack(fill_value=0))
counts["total"] = counts.sum(axis=1)
counts.reset_index().to_csv(OUT / "drugstores_per_municity.csv", index=False)
print("final placement:", res["psgc_source"].value_counts().to_dict(), "| municipalities with a branch:", len(counts))
res.to_csv(OUT / "ph_drugstores_checked.csv", index=False)
print(res["qa_status"].value_counts().to_string())
print()
print(pd.crosstab(res["Franchise Name"], res["qa_status"]).to_string())
prior = res[res["Note"].notna()]
print("\nrows the scrape had already flagged:", len(prior), prior["qa_status"].value_counts().to_dict())
