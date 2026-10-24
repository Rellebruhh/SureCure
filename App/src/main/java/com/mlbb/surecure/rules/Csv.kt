package com.mlbb.surecure.rules

/**
 * Minimal CSV reader for the FDA tables, matching Python's csv.DictReader on them:
 * comma separated, double-quoted fields (with "" as an escaped quote), \n or \r\n rows.
 */
internal object Csv {

    /** Rows as header -> value maps. Missing trailing fields read as "". */
    fun readDicts(text: String): List<Map<String, String>> {
        val rows = readRows(text)
        if (rows.isEmpty()) return emptyList()
        val header = rows[0]
        return rows.drop(1)
            .filter { row -> row.any { it.isNotEmpty() } }
            .map { row -> header.indices.associate { header[it] to row.getOrElse(it) { "" } } }
    }

    fun readRows(text: String): List<List<String>> {
        val rows = mutableListOf<List<String>>()
        var row = mutableListOf<String>()
        val field = StringBuilder()
        var inQuotes = false
        var i = 0
        while (i < text.length) {
            val c = text[i]
            if (inQuotes) {
                if (c == '"') {
                    if (i + 1 < text.length && text[i + 1] == '"') {
                        field.append('"')
                        i++
                    } else {
                        inQuotes = false
                    }
                } else {
                    field.append(c)
                }
            } else when (c) {
                '"' -> inQuotes = true
                ',' -> {
                    row.add(field.toString())
                    field.setLength(0)
                }
                '\r', '\n' -> {
                    if (c == '\r' && i + 1 < text.length && text[i + 1] == '\n') i++
                    row.add(field.toString())
                    field.setLength(0)
                    rows.add(row)
                    row = mutableListOf()
                }
                else -> field.append(c)
            }
            i++
        }
        if (field.isNotEmpty() || row.isNotEmpty()) {
            row.add(field.toString())
            rows.add(row)
        }
        return rows
    }
}
