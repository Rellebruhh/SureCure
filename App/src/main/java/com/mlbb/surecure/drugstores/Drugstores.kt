package com.mlbb.surecure.drugstores

import com.mlbb.surecure.rules.Csv
import kotlin.math.asin
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

/*
 * Chain drugstore branches bundled with the app (assets/drugstores/branches.csv, built by
 * scripts/build_drugstore_asset.py). Only branches with an exact, QA-checked location are bundled,
 * so every branch here may be shown as a location.
 *
 * This is a list of chain branches (Mercury Drug, Watsons, Generika, TGP), not an FDA license list:
 * any screen showing it must say so (DRUGSTORE_DISCLAIMER). Never call them licensed or verified.
 * Distances are computed on the phone; the user's location is never stored or sent.
 * Pure Kotlin so it runs in JVM unit tests.
 */

const val DRUGSTORE_DISCLAIMER = "Chain drugstore branches. Not a complete list of licensed pharmacies."

data class DrugstoreBranch(
    val franchise: String,
    val branch: String,
    val address: String,
    val lat: Double,
    val lon: Double,
    /** 10-digit PSGC city/municipality code; the join key for anything per municipality. */
    val psgc: String,
    val municipality: String,
    val province: String,
)

data class NearbyBranch(val branch: DrugstoreBranch, val distanceKm: Double)

/** A city or municipality for the manual picker. */
data class Municipality(val psgc: String, val name: String, val province: String)

/** Great-circle distance in km (haversine, mean Earth radius 6371.0088 km). */
fun haversineKm(lat1: Double, lon1: Double, lat2: Double, lon2: Double): Double {
    val p1 = Math.toRadians(lat1)
    val p2 = Math.toRadians(lat2)
    val dp = p2 - p1
    val dl = Math.toRadians(lon2 - lon1)
    val h = sin(dp / 2) * sin(dp / 2) + cos(p1) * cos(p2) * sin(dl / 2) * sin(dl / 2)
    return 2 * 6371.0088 * asin(sqrt(h.coerceIn(0.0, 1.0)))
}

class DrugstoreDirectory(val branches: List<DrugstoreBranch>) {

    /** The [k] branches closest to (lat, lon), nearest first. Ties keep file order. */
    fun nearestBranches(lat: Double, lon: Double, k: Int = 10): List<NearbyBranch> {
        if (k <= 0) return emptyList()
        return branches
            .map { NearbyBranch(it, haversineKm(lat, lon, it.lat, it.lon)) }
            .sortedBy { it.distanceKm }
            .take(k)
    }

    /** Province names for the picker (used when location permission is refused), sorted. */
    fun provinces(): List<String> = branches.map { it.province }.distinct().sorted()

    /** Cities and municipalities in [province] that have at least one bundled branch, sorted by name. */
    fun municipalities(province: String): List<Municipality> =
        branches.filter { it.province == province }
            .map { Municipality(it.psgc, it.municipality, it.province) }
            .distinct()
            .sortedBy { it.name }

    /**
     * Cities and municipalities whose name contains [query] (any province), sorted by name.
     * Needed because highly urbanized cities are their own "province" in the PSGC: Cebu City is
     * filed under "Cebu City", not "Cebu", so a province-first picker alone would hide it.
     */
    fun findMunicipalities(query: String): List<Municipality> {
        val q = query.trim()
        if (q.isEmpty()) return emptyList()
        return branches.filter { it.municipality.contains(q, ignoreCase = true) }
            .map { Municipality(it.psgc, it.municipality, it.province) }
            .distinct()
            .sortedBy { it.name }
    }

    /** Branches in one city or municipality, by PSGC code, sorted by chain then branch name. */
    fun branchesIn(psgc: String): List<DrugstoreBranch> =
        branches.filter { it.psgc == psgc }.sortedWith(compareBy({ it.franchise }, { it.branch }))

    companion object {
        fun fromCsv(text: String): DrugstoreDirectory = DrugstoreDirectory(
            Csv.readDicts(text).map { r ->
                DrugstoreBranch(
                    franchise = r.getValue("franchise"),
                    branch = r.getValue("branch"),
                    address = r.getValue("address"),
                    lat = r.getValue("lat").toDouble(),
                    lon = r.getValue("lon").toDouble(),
                    psgc = r.getValue("psgc"),
                    municipality = r.getValue("municipality"),
                    province = r.getValue("province"),
                )
            }
        )
    }
}
