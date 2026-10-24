package com.mlbb.surecure.rules

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

/**
 * The Kotlin engine against the Python reference on the bundled FDA tables: same level, same
 * product checked and the same reasons, word for word, in the same order.
 * Cases and Python verdicts: src/test/resources/rules/parity_cases.tsv, written by
 * rules/tests/make_parity_cases.py (every listed lot, plus random near misses, wrong and unknown
 * registrations, bad barcodes and dates).
 */
class RulesParityTest {

    private val today = PackDate(2026, 10, 24)

    private fun resource(name: String): String =
        javaClass.classLoader!!.getResourceAsStream("rules/$name")!!.bufferedReader().use { it.readText() }

    private fun asset(name: String) = File("src/main/assets/fda/$name").readText()

    private fun cell(s: String): String? = if (s == "\\N") null else s

    private fun date(s: String): PackDate? = cell(s)?.let { PackDate.parseIso(it) }

    @Test
    fun kotlinVerdictsMatchPython() {
        val real = FdaTables.fromCsv(asset("registrations.csv"), asset("counterfeit_lots.csv"))
        val withBarcodes = FdaTables.fromCsv(
            asset("registrations.csv"), asset("counterfeit_lots.csv"), resource("parity_barcodes.csv")
        )
        val lines = resource("parity_cases.tsv").lines().filter { it.isNotEmpty() }
        assertTrue("expected at least 1000 cases, got ${lines.size}", lines.size >= 1000)
        val failures = mutableListOf<String>()
        for ((n, line) in lines.withIndex()) {
            val f = line.split("\t")
            val tables = if (f[0] == "real") real else withBarcodes
            val scan = ScanInput(cell(f[1]), cell(f[2]), cell(f[3]), cell(f[4]), date(f[5]), date(f[6]))
            val v = verify(scan, tables, today)
            val expectedReasons = f[9].split(" || ")
            if (v.level.name.lowercase() != f[7] || v.productChecked != cell(f[8]) || v.reasons != expectedReasons) {
                failures += "case ${n + 1} $scan\n  python: ${f[7]} ${f[8]} $expectedReasons\n" +
                    "  kotlin: ${v.level} ${v.productChecked} ${v.reasons}"
            }
        }
        assertEquals(
            "${failures.size} of ${lines.size} cases differ from Python:\n" + failures.take(10).joinToString("\n"),
            0, failures.size
        )
    }
}
