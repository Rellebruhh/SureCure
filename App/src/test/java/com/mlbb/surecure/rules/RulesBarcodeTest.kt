package com.mlbb.surecure.rules

import com.mlbb.surecure.rules.VerdictLevel.AMBER
import com.mlbb.surecure.rules.VerdictLevel.GREEN
import com.mlbb.surecure.rules.VerdictLevel.RED
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

/**
 * Barcode rules on the bundled tables, including assets/fda/barcodes.csv.
 * Expected reasons are the Python engine's output for the same cases (2026-10-24).
 */
class RulesBarcodeTest {

    private val today = PackDate(2026, 10, 24)
    private fun asset(name: String) = File("src/main/assets/fda/$name").readText()
    private val tables = FdaTables.fromCsv(
        asset("registrations.csv"), asset("counterfeit_lots.csv"), asset("barcodes.csv")
    )

    private val biofluBox = "4807788526762"
    private val validUnknown = "4807788000019"
    private val badCheckDigit = "4807788526763"

    private fun check(scan: ScanInput) = verify(scan, tables, today)

    @Test
    fun barcodeTableLoads() {
        assertEquals(18, tables.barcodes.size)
        assertTrue("every code passes the check digit", tables.barcodes.keys.all { ean13Valid(it) })
        assertEquals("Bioflu", tables.barcodes[biofluBox])
    }

    @Test
    fun ownBarcodeGivesNoBarcodeComplaint() {
        val v = check(ScanInput("Bioflu", barcode = biofluBox, lot = "ZZ99999"))
        assertFalse(v.reasons.any { it.contains("Barcode", ignoreCase = true) })
        assertEquals(listOf("No registration number entered."), v.reasons)
        // With Bioflu's own registration too, nothing is left to flag.
        assertEquals(GREEN, check(ScanInput("Bioflu", barcode = biofluBox, lot = "ZZ99999", regNo = "DR-XY34482")).level)
    }

    @Test
    fun barcodeOfAnotherProductIsRed() {
        val v = check(ScanInput("Biogesic", barcode = biofluBox, lot = "ZZ99999"))
        assertEquals(RED, v.level)
        assertEquals("Barcode belongs to Bioflu, not Biogesic.", v.reasons[0])
    }

    @Test
    fun validBarcodeNotInTableIsAmber() {
        assertTrue(ean13Valid(validUnknown))
        val v = check(ScanInput("Biogesic", barcode = validUnknown, lot = "ZZ99999", regNo = "DR-153"))
        assertEquals(AMBER, v.level)
        assertEquals(listOf("Barcode $validUnknown is not in the barcode table."), v.reasons)
    }

    @Test
    fun wrongCheckDigitIsAmberNeverRed() {
        assertFalse(ean13Valid(badCheckDigit))
        val v = check(ScanInput("Biogesic", barcode = badCheckDigit, lot = "ZZ99999", regNo = "DR-153"))
        assertEquals(AMBER, v.level)
        assertEquals(listOf("Barcode could not be read reliably (EAN-13 check digit fails)."), v.reasons)
        // Even with Bioflu's digits, a bad check digit never names a product, so no red.
        assertNotEquals(RED, check(ScanInput("Biogesic", barcode = badCheckDigit, lot = "ZZ99999")).level)
    }

    @Test
    fun barcodeAndRegistrationOfDifferentProductsIsRed() {
        val v = check(ScanInput("Biogesic", barcode = "4807788561688", lot = "ZZ99999", regNo = "DR-XY34482"))
        assertEquals(RED, v.level)
        assertEquals(
            listOf(
                "Barcode belongs to Biogesic but registration DR-XY34482 belongs to Bioflu.",
                "Registration DR-XY34482 belongs to Bioflu, not Biogesic.",
            ),
            v.reasons
        )
    }

    @Test
    fun barcodeFirstLookup() {
        assertEquals("Bioflu", tables.productForBarcode(biofluBox))
        assertEquals("Lagundi", tables.productForBarcode("4800204012412"))  // Ascof Forte pack
        assertNull("valid but not in the table: run the photo model", tables.productForBarcode(validUnknown))
        assertNull("bad check digit: run the photo model", tables.productForBarcode(badCheckDigit))
        assertNull(tables.productForBarcode(null))
        assertNull(tables.productForBarcode(""))
    }
}
