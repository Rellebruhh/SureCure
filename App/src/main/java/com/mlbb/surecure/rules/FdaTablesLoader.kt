package com.mlbb.surecure.rules

import android.content.Context
import java.io.FileNotFoundException

/**
 * Loads the FDA tables from assets/fda/ once and caches them.
 *
 * registrations.csv and counterfeit_lots.csv are required. barcodes.csv is optional (it covers
 * 14 of the 20 products so far); a scanned barcode not in it gives amber "not in the barcode table".
 * The tables are built by scripts/build_fda_tables.py from data/fda/raw/; don't edit them by hand.
 */
object FdaTablesLoader {

    const val ASSET_DIR = "fda"

    @Volatile
    private var cached: FdaTables? = null

    /** Reads ~50 KB of CSV on first call; call off the main thread. */
    fun load(context: Context): FdaTables {
        cached?.let { return it }
        synchronized(this) {
            cached?.let { return it }
            val assets = context.applicationContext.assets
            fun read(name: String) = assets.open("$ASSET_DIR/$name").bufferedReader().use { it.readText() }
            val barcodes = try {
                read("barcodes.csv")
            } catch (e: FileNotFoundException) {
                null
            }
            val tables = FdaTables.fromCsv(read("registrations.csv"), read("counterfeit_lots.csv"), barcodes)
            cached = tables
            return tables
        }
    }
}
