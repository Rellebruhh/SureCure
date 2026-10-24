package com.mlbb.surecure.drugstores

import android.content.Context

/** Loads assets/drugstores/branches.csv once and caches it. About 700 KB; call off the main thread. */
object DrugstoreLoader {

    const val ASSET = "drugstores/branches.csv"

    @Volatile
    private var cached: DrugstoreDirectory? = null

    fun load(context: Context): DrugstoreDirectory {
        cached?.let { return it }
        synchronized(this) {
            cached?.let { return it }
            val text = context.applicationContext.assets.open(ASSET).bufferedReader().use { it.readText() }
            return DrugstoreDirectory.fromCsv(text).also { cached = it }
        }
    }
}
