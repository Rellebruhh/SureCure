package com.mlbb.surecure.ui.result

import com.mlbb.surecure.ml.ClassifierResult
import com.mlbb.surecure.ml.ProductScore
import com.mlbb.surecure.ml.Recognition
import com.mlbb.surecure.rules.FdaTables
import com.mlbb.surecure.rules.PackDate
import com.mlbb.surecure.rules.VerdictLevel
import com.mlbb.surecure.rules.verify
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

class ScanFlowTest {

    private val labels = listOf(
        "Ascozin", "Bioflu", "Biogesic", "Bonamine", "Buscopan", "DayZinc", "Decolgen", "Flanax",
        "Imodium", "Lactezin", "Lagundi", "Midol", "Myra_E", "Neurogen_E", "Omeprazole", "Rinityn",
        "Rogin_E", "Sinecod", "Tempra", "Tuseran_Forte"
    )
    private val tables = FdaTables.fromCsv(
        File("src/main/assets/fda/registrations.csv").readText(),
        File("src/main/assets/fda/counterfeit_lots.csv").readText(),
    )
    private val today = PackDate(2026, 10, 24)

    private fun result(status: Recognition) = ClassifierResult(
        status,
        listOf(ProductScore("Biogesic", 0.6f), ProductScore("Bioflu", 0.2f), ProductScore("Tempra", 0.1f)),
        energy = -4.5f,
        logits = FloatArray(20),
        runName = "test",
    )

    @Test
    fun stage1FollowsFigure1() {
        assertEquals(ScanFlow.ProductStep.Confirm("Biogesic"), ScanFlow.productStep(result(Recognition.ACCEPTED)))
        assertEquals(
            ScanFlow.ProductStep.Pick(listOf("Biogesic", "Bioflu", "Tempra")),
            ScanFlow.productStep(result(Recognition.LOW_CONFIDENCE))
        )
        assertEquals(ScanFlow.ProductStep.ChooseFromList, ScanFlow.productStep(result(Recognition.NOT_RECOGNIZED)))
        assertEquals(ScanFlow.ProductStep.ChooseFromList, ScanFlow.productStep(null))
    }

    @Test
    fun manualSelectionOffersModelProductsThenOtherBrands() {
        val choices = ScanFlow.productChoices(labels, tables)
        assertEquals(labels, choices.take(20))
        assertTrue("Alaxan FR" in choices)
        assertEquals(choices.size, choices.toSet().size)
    }

    @Test
    fun displayNames() {
        assertEquals("Myra E", ScanFlow.displayName("Myra_E"))
        assertEquals("Tuseran Forte", ScanFlow.displayName("Tuseran_Forte"))
    }

    @Test
    fun blankFieldsAreNotEnteredAndDatesParse() {
        val parsed = ScanFlow.parse("Biogesic", ScanFlow.CodeEntry(lot = " 23651568 ", expiry = "MAR 2028"))
        parsed as ScanFlow.Parsed.Ok
        assertEquals("23651568", parsed.scan.lot)
        assertNull(parsed.scan.regNo)
        assertNull(parsed.scan.mfgDate)
        assertEquals(PackDate(2028, 3, 31), parsed.scan.expDate)
    }

    @Test
    fun unreadableDateIsAnErrorNotIgnored() {
        val parsed = ScanFlow.parse("Biogesic", ScanFlow.CodeEntry(expiry = "03/15/2028"))
        parsed as ScanFlow.Parsed.Errors
        assertEquals(setOf("expiry"), parsed.fields.keys)
    }

    @Test
    fun redHeadlineSaysAdvisoryOnlyForLotMatches() {
        val lot = verify(scanFor("Biogesic", ScanFlow.CodeEntry(lot = "23651568")), tables, today)
        assertEquals(VerdictLevel.RED, lot.level)
        assertEquals("Matches an FDA counterfeit advisory", ScanFlow.headline(lot))

        val mismatch = verify(
            scanFor("Biogesic", ScanFlow.CodeEntry(lot = "ZZ99999", regNo = "DR-XY34482")), tables, today
        )
        assertEquals(VerdictLevel.RED, mismatch.level)
        assertEquals(ScanFlow.RED_MISMATCH_HEADLINE, ScanFlow.headline(mismatch))
    }

    @Test
    fun amberAndGreenHeadlines() {
        val amber = verify(scanFor("Myra_E", ScanFlow.CodeEntry(lot = "ZZ99999")), tables, today)
        assertEquals("Couldn't fully verify", ScanFlow.headline(amber))
        val green = verify(scanFor("Biogesic", ScanFlow.CodeEntry(lot = "ZZ99999", regNo = "DR-153")), tables, today)
        assertEquals("No match with FDA counterfeit records", ScanFlow.headline(green))
    }

    private fun scanFor(product: String, entry: ScanFlow.CodeEntry) =
        (ScanFlow.parse(product, entry) as ScanFlow.Parsed.Ok).scan
}
