package com.mlbb.surecure.rules

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class PackDateTest {

    private fun exp(s: String?) = PackDate.parsePrinted(s, endOfMonth = true)
    private fun mfg(s: String) = PackDate.parsePrinted(s, endOfMonth = false)

    @Test
    fun monthOnlyExpiryIsLastDayOfMonth() {
        assertEquals(PackDate(2028, 3, 31), exp("MAR 2028"))
        assertEquals(PackDate(2028, 3, 31), exp("EXP: MAR2028"))
        assertEquals(PackDate(2028, 3, 31), exp("March 2028"))
        assertEquals(PackDate(2027, 5, 31), exp("05/2027"))
        assertEquals(PackDate(2027, 5, 31), exp("05-27"))
        assertEquals(PackDate(2026, 4, 30), exp("2026-04"))
        assertEquals(PackDate(2028, 2, 29), exp("FEB 2028"))
        assertEquals(PackDate(2027, 2, 28), exp("02/2027"))
    }

    @Test
    fun monthOnlyManufactureIsFirstDayOfMonth() {
        assertEquals(PackDate(2025, 3, 1), mfg("MFG MAR 2025"))
        assertEquals(PackDate(2025, 3, 1), mfg("03/2025"))
    }

    @Test
    fun fullDates() {
        assertEquals(PackDate(2028, 3, 15), exp("2028-03-15"))
        assertEquals(PackDate(2028, 3, 15), exp("15 MAR 2028"))
        assertEquals(PackDate(2028, 3, 15), exp("15-Mar-28"))
    }

    @Test
    fun unclearDatesAreNull() {
        assertNull(exp("03/15/2028"))   // day/month order is ambiguous
        assertNull(exp("13/2028"))
        assertNull(exp("FOO 2028"))
        assertNull(exp("31 FEB 2028"))
        assertNull(exp(""))
        assertNull(exp(null))
    }

    @Test
    fun packIsNotExpiredDuringItsLastMonth() {
        val v = verify(
            ScanInput("Biogesic", regNo = "DR-153", lot = "ZZ99999", expDate = exp("OCT 2026")),
            FdaTables(registrations = mapOf("DR-153" to Registration("DR-153", "Biogesic"))),
            today = PackDate(2026, 10, 24)
        )
        assertEquals(VerdictLevel.GREEN, v.level)
    }

    @Test
    fun isoFormatAndOrder() {
        assertEquals("2026-01-05", PackDate(2026, 1, 5).toString())
        assertTrue(PackDate(2026, 1, 5) < PackDate(2026, 2, 1))
        assertEquals(PackDate(2026, 1, 5), PackDate.parseIso("2026-01-05"))
    }
}
