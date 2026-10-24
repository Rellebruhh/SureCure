package com.mlbb.surecure.ui.result

import android.content.Context
import com.mlbb.surecure.ml.ProductClassifier
import com.mlbb.surecure.rules.FdaTables
import com.mlbb.surecure.rules.FdaTablesLoader
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext

/**
 * App-wide photo model and FDA tables, loaded once on first use and kept for the life of the
 * process (the model is ~46 MB; reloading it per scan would be slow).
 */
object ScanEngines {

    private val lock = Mutex()
    private var classifier: ProductClassifier? = null

    suspend fun classifier(context: Context): ProductClassifier = withContext(Dispatchers.Default) {
        lock.withLock {
            classifier ?: ProductClassifier.create(context.applicationContext).also { classifier = it }
        }
    }

    suspend fun tables(context: Context): FdaTables = withContext(Dispatchers.IO) {
        FdaTablesLoader.load(context.applicationContext)
    }
}
