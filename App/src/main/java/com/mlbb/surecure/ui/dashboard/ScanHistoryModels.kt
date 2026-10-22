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

data class ScanHistoryEntryPreview(
    val id: String,
    val safety: ScanSafety,
    val scannedOn: String,
    @DrawableRes val thumbnailResId: Int = R.drawable.test_thumbnail
)

data class ScanHistorySection(
    val title: String,
    val entries: List<ScanHistoryEntryPreview>
)

/** Text and grouping preserved from the Figma HTML export. */
fun sampleScanHistory(): List<ScanHistorySection> = listOf(
    ScanHistorySection(
        title = "This Month",
        entries = listOf(
            ScanHistoryEntryPreview("scan-1", ScanSafety.SAFE, "Scanned on October 19, 2026"),
            ScanHistoryEntryPreview("scan-2", ScanSafety.SAFE, "Scanned on October 19, 2026"),
            ScanHistoryEntryPreview("scan-3", ScanSafety.UNSAFE, "Scanned on October 19, 2026")
        )
    ),
    ScanHistorySection(
        title = "September",
        entries = listOf(
            ScanHistoryEntryPreview("scan-4", ScanSafety.SAFE, "Scanned on October 19, 2026"),
            ScanHistoryEntryPreview("scan-5", ScanSafety.SAFE, "Scanned on October 19, 2026"),
            ScanHistoryEntryPreview("scan-6", ScanSafety.SAFE, "Scanned on October 19, 2026")
        )
    )
)
