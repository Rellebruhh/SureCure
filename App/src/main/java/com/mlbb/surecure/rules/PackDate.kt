package com.mlbb.surecure.rules

import java.util.Calendar
import java.util.Locale

/**
 * Calendar date for the rules engine (year, month 1..12, day).
 *
 * Not java.time.LocalDate on purpose: minSdk is 24 and java.time needs API 26 without
 * desugaring. toString() prints yyyy-MM-dd, like Python's date, so reason texts match
 * rules/surecure_rules.py exactly.
 */
data class PackDate(val year: Int, val month: Int, val day: Int) : Comparable<PackDate> {

    init {
        require(month in 1..12) { "month $month out of range" }
        require(day in 1..lastDay(year, month)) { "day $day out of range for $year-$month" }
    }

    override fun compareTo(other: PackDate): Int =
        compareValuesBy(this, other, { it.year }, { it.month }, { it.day })

    override fun toString(): String = String.format(Locale.ROOT, "%04d-%02d-%02d", year, month, day)

    companion object {
        private val MONTHS = listOf(
            "JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"
        )

        fun today(): PackDate {
            val c = Calendar.getInstance()
            return PackDate(c.get(Calendar.YEAR), c.get(Calendar.MONTH) + 1, c.get(Calendar.DAY_OF_MONTH))
        }

        fun lastDay(year: Int, month: Int): Int = when (month) {
            2 -> if ((year % 4 == 0 && year % 100 != 0) || year % 400 == 0) 29 else 28
            4, 6, 9, 11 -> 30
            else -> 31
        }

        /** yyyy-MM-dd, as in the CSV tables. Throws on anything else. */
        fun parseIso(text: String): PackDate {
            val m = Regex("""(\d{4})-(\d{2})-(\d{2})""").matchEntire(text.trim())
                ?: throw IllegalArgumentException("not an ISO date: '$text'")
            return PackDate(m.groupValues[1].toInt(), m.groupValues[2].toInt(), m.groupValues[3].toInt())
        }

        /**
         * A date as printed on a pack, or null if it isn't understood (ask the user instead).
         *
         * Month-only dates ("MAR 2028", "03/2028", "2028-03", "03/28") become the last day of
         * the month when [endOfMonth] is true (use for expiry, so a pack isn't called expired
         * during its final month) and the first day otherwise (use for manufacture).
         * Full dates: "2028-03-15", "15 MAR 2028". Numeric day/month/year orders such as
         * "03/15/2028" are ambiguous and return null.
         */
        fun parsePrinted(text: String?, endOfMonth: Boolean): PackDate? {
            if (text == null) return null
            val t = text.trim().uppercase(Locale.ROOT)
                .removePrefix("EXP").removePrefix("MFG").removePrefix("MFD")
                .trim(' ', '.', ':')
            fun year(s: String) = if (s.length == 2) 2000 + s.toInt() else s.toInt()
            fun month(name: String) = MONTHS.indexOf(name.take(3)).takeIf { it >= 0 }?.plus(1)
            fun monthOnly(y: Int, m: Int?): PackDate? {
                if (m == null || m !in 1..12) return null
                return PackDate(y, m, if (endOfMonth) lastDay(y, m) else 1)
            }
            fun full(y: Int, m: Int?, d: Int): PackDate? {
                if (m == null || m !in 1..12 || d !in 1..lastDay(y, m)) return null
                return PackDate(y, m, d)
            }
            Regex("""(\d{4})-(\d{1,2})-(\d{1,2})""").matchEntire(t)?.let {
                return full(it.groupValues[1].toInt(), it.groupValues[2].toInt(), it.groupValues[3].toInt())
            }
            Regex("""(\d{1,2})[\s./-]*([A-Z]{3,9})\.?[\s./-]*(\d{4}|\d{2})""").matchEntire(t)?.let {
                return full(year(it.groupValues[3]), month(it.groupValues[2]), it.groupValues[1].toInt())
            }
            Regex("""([A-Z]{3,9})\.?[\s./-]*(\d{4}|\d{2})""").matchEntire(t)?.let {
                return monthOnly(year(it.groupValues[2]), month(it.groupValues[1]))
            }
            Regex("""(\d{4})[./-](\d{1,2})""").matchEntire(t)?.let {
                return monthOnly(it.groupValues[1].toInt(), it.groupValues[2].toInt())
            }
            Regex("""(\d{1,2})[./-](\d{4}|\d{2})""").matchEntire(t)?.let {
                return monthOnly(year(it.groupValues[2]), it.groupValues[1].toInt())
            }
            return null
        }
    }
}
