package com.mlbb.surecure.ui.dashboard

import android.os.Build
import androidx.annotation.DrawableRes
import androidx.annotation.RequiresApi
import com.mlbb.surecure.R
import java.time.Instant

/**
 * Demo data models for the dashboard's scan history.
 * Replace the sample entries with records from your repository/database later.
 */
enum class ScanSafety {
    SAFE,
    UNSAFE
}

data class ScanHistoryEntryPreview(
    val id: String,
    val safety: ScanSafety,
    val scannedOn: Instant,
    @DrawableRes val thumbnailResId: Int = R.drawable.test_thumbnail
)

/**
 * Helper class that allows the segmenting of scan history previews into time periods.
 * Scan history is separated by major time periods, the same way photos are in gallery apps
 * (This Month, August 2026, July 2026, June 2026, ..., Over A Year Ago, etc.)
 */
data class ScanHistorySection(
    val timePeriod: String,
    val entries: List<ScanHistoryEntryPreview>
)

/** Text and grouping preserved from the Figma HTML export. */
@RequiresApi(Build.VERSION_CODES.O)
fun sampleScanHistory(): List<ScanHistorySection> = listOf(
    ScanHistorySection(
        timePeriod = "This Month",
        entries = listOf(
            ScanHistoryEntryPreview("scan-1", ScanSafety.SAFE, Instant.now()),
            ScanHistoryEntryPreview("scan-2", ScanSafety.SAFE, Instant.now()),
            ScanHistoryEntryPreview("scan-3", ScanSafety.UNSAFE, Instant.now())
        )
    ),
    ScanHistorySection(
        timePeriod = "September",
        entries = listOf(
            ScanHistoryEntryPreview("scan-4", ScanSafety.SAFE, Instant.now()),
            ScanHistoryEntryPreview("scan-5", ScanSafety.SAFE, Instant.now()),
            ScanHistoryEntryPreview("scan-6", ScanSafety.SAFE, Instant.now()),
            ScanHistoryEntryPreview("scan-6", ScanSafety.SAFE, Instant.now()),
            ScanHistoryEntryPreview("scan-6", ScanSafety.SAFE, Instant.now()),
            ScanHistoryEntryPreview("scan-6", ScanSafety.SAFE, Instant.now())
        )
    )
)
