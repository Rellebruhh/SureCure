package com.mlbb.surecure.util

import android.os.Build
import androidx.annotation.RequiresApi
import java.time.Instant
import java.time.ZoneId
import java.time.format.DateTimeFormatter
import java.time.format.FormatStyle
import java.util.Locale

@RequiresApi(Build.VERSION_CODES.O)
fun formatScanDate(
    instant: Instant,
    zoneId: ZoneId = ZoneId.systemDefault()
): String {
    val formatter = DateTimeFormatter
        .ofPattern("MMMM d, yyyy", Locale.getDefault())
        .withLocale(Locale.getDefault())


    return formatter.format(instant.atZone(zoneId))
}