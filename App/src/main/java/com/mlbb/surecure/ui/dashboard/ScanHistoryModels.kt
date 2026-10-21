package com.mlbb.surecure.ui.dashboard

import androidx.annotation.DrawableRes
import com.mlbb.surecure.R

/**
 * Demo data models for the dashboard's scan history.
 * Replace the sample entries with records from your repository/database later.
 */
enum class ScanSafety {
    SAFE,
    UNSAFE
}

data class ScanHistoryEntry(
    val id: String,
    val safety: ScanSafety,
    val scannedOn: String,
    @DrawableRes val thumbnailResId: Int = R.drawable.test_thumbnail
)

data class ScanHistorySection(
    val title: String,
    val entries: List<ScanHistoryEntry>
)

/** Text and grouping preserved from the Figma HTML export. */
fun sampleScanHistory(): List<ScanHistorySection> = listOf(
    ScanHistorySection(
        title = "This Month",
        entries = listOf(
            ScanHistoryEntry("scan-1", ScanSafety.SAFE, "Scanned on October 19, 2026"),
            ScanHistoryEntry("scan-2", ScanSafety.SAFE, "Scanned on October 19, 2026"),
            ScanHistoryEntry("scan-3", ScanSafety.UNSAFE, "Scanned on October 19, 2026")
        )
    ),
    ScanHistorySection(
        title = "September",
        entries = listOf(
            ScanHistoryEntry("scan-4", ScanSafety.SAFE, "Scanned on October 19, 2026"),
            ScanHistoryEntry("scan-5", ScanSafety.SAFE, "Scanned on October 19, 2026"),
            ScanHistoryEntry("scan-6", ScanSafety.SAFE, "Scanned on October 19, 2026")
        )
    )
)
