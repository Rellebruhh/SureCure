package com.mlbb.surecure.drugstores

import com.mlbb.surecure.rules.Csv
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

/**
 * nearestBranches and the bundled asset. Expected nearest branches were computed independently
 * in Python on the same file (2026-10-24).
 */
class DrugstoresTest {

    private val directory = DrugstoreDirectory.fromCsv(
        File("src/main/assets/drugstores/branches.csv").readText()
    )

    @Test
    fun haversineKnownDistances() {
        assertEquals(111.195, haversineKm(14.0, 121.0, 15.0, 121.0), 0.001)   // 1 degree of latitude
        assertEquals(571.0, haversineKm(14.5995, 120.9842, 10.3157, 123.8854), 0.5)  // Manila to Cebu
        assertEquals(0.0, haversineKm(14.6, 121.0, 14.6, 121.0), 0.0)
    }

    @Test
    fun quezonCityPointReturnsQuezonCityBranchesFirst() {
        val near = directory.nearestBranches(14.6516, 121.0493)  // Quezon Memorial Circle
        assertEquals(10, near.size)
        assertTrue(near.all { it.branch.municipality == "Quezon City" })
        assertEquals("TGP #189", near[0].branch.branch)
        assertEquals(0.425, near[0].distanceKm, 0.001)
        assertEquals(1.573, near[9].distanceKm, 0.001)
        assertTrue("nearest first", near.zipWithNext().all { (a, b) -> a.distanceKm <= b.distanceKm })
    }

    @Test
    fun otherCitiesToo() {
        val cebu = directory.nearestBranches(10.3110, 123.8930)
        assertTrue(cebu.all { it.branch.municipality == "Cebu City" })
        val davao = directory.nearestBranches(7.0731, 125.6128, k = 5)
        assertEquals(5, davao.size)
        assertTrue(davao.all { it.branch.municipality == "City of Davao" })
    }

    @Test
    fun kEdgeCases() {
        assertTrue(directory.nearestBranches(14.65, 121.05, k = 0).isEmpty())
        assertEquals(directory.branches.size, directory.nearestBranches(14.65, 121.05, k = 100_000).size)
        assertTrue(DrugstoreDirectory(emptyList()).nearestBranches(14.65, 121.05).isEmpty())
    }

    @Test
    fun assetHoldsOnlyExactLocations() {
        // 3,651 branches with coord_precision = exact; centroid and no-location rows are not bundled.
        assertEquals(3651, directory.branches.size)
        assertTrue(directory.branches.all { it.lat in 4.0..21.5 && it.lon in 116.0..127.0 })
        assertTrue("Southstar has no exact locations", directory.branches.none { it.franchise.startsWith("Southstar") })
    }

    @Test
    fun psgcCodesAreTenDigits() {
        assertTrue(directory.branches.all { it.psgc.length == 10 && it.psgc.all(Char::isDigit) })
        assertEquals("0102802000", directory.branches.first { it.municipality == "Bacarra" }.psgc)
    }

    @Test
    fun noBranchFromTheReviewList() {
        val review = Csv.readDicts(File("../data/drugstores/needs_review.csv").readText())
            .map { it.getValue("Franchise Name") to it.getValue("Branch Name") }
            .toSet()
        assertEquals(66, review.size)
        val bundled = directory.branches.map { it.franchise to it.branch }
        assertTrue(bundled.none { it in review })
    }

    @Test
    fun pickerFallback() {
        val provinces = directory.provinces()
        assertTrue("Cebu" in provinces)
        // Highly urbanized cities are their own "province" in the PSGC: Cebu City is not under Cebu.
        assertTrue(directory.municipalities("Cebu").none { it.name == "Cebu City" })
        assertTrue(directory.municipalities("Cebu").isNotEmpty())
        // City search finds it regardless of province.
        val found = directory.findMunicipalities("cebu city")
        assertEquals(listOf("Cebu City"), found.map { it.name })
        val inCebu = directory.branchesIn(found[0].psgc)
        assertTrue(inCebu.isNotEmpty() && inCebu.all { it.municipality == "Cebu City" })
        assertTrue(directory.findMunicipalities("  ").isEmpty())
    }
}
