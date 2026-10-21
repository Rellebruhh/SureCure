package com.mlbb.surecure.ui.dashboard

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

private val SureCureMint = Color(0xFFDEFFE8)
private val SureCureDarkGreen = Color(0xFF02542D)
private val SureCureGreen = Color(0xFF14AE5C)
private val SureCureGray = Color(0xFF757575)
private val SureCureRed = Color(0xFFC00F0C)
private val UnsafeRowBackground = Color(0xFFFFE8E8)

/**
 * Native Jetpack Compose version of the dashboard exported from Figma.
 *
 * @param onCameraClick Called when the user taps the green camera button. Connect this
 * to your navigation flow (for example, navigate to CaptureScreen).
 * @param onScanClick Optional callback for a scan-history row.
 * @param sections Data to display. Defaults to sample content matching the HTML export.
 */
@Composable
fun DashboardScreen(
    onCameraClick: () -> Unit,
    onScanClick: (ScanHistoryEntry) -> Unit = {},
    sections: List<ScanHistorySection> = sampleScanHistory()
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(Color.White),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        BrandHeader()

        Column(
            modifier = Modifier
                .fillMaxWidth()
                .weight(1f)
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 30.dp)
        ) {
            Text(
                text = "Scan History",
                color = Color.Black,
                fontSize = 24.sp,
                fontWeight = FontWeight.Bold,
                lineHeight = 29.sp,
                modifier = Modifier.padding(top = 18.dp, bottom = 10.dp)
            )

            sections.forEach { section ->
                Text(
                    text = section.title,
                    color = Color.Black,
                    fontSize = 16.sp,
                    lineHeight = 20.sp,
                    modifier = Modifier.padding(top = 2.dp, bottom = 10.dp)
                )

                section.entries.forEach { entry ->
                    ScanHistoryRow(
                        entry = entry,
                        onClick = { onScanClick(entry) }
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                }
                Spacer(modifier = Modifier.height(4.dp))
            }
        }

        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(top = 8.dp, bottom = 22.dp),
            horizontalArrangement = Arrangement.Center,
            verticalAlignment = Alignment.CenterVertically
        ) {
            CameraButton(onClick = onCameraClick)
        }
    }
}

@Composable
private fun BrandHeader() {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .background(SureCureMint)
            .padding(horizontal = 20.dp, vertical = 25.dp),
        horizontalArrangement = Arrangement.Center,
        verticalAlignment = Alignment.CenterVertically
    ) {
        // The exported logo SVGs were not included with the uploaded HTML/CSS,
        // so this small capsule mark is an editable placeholder.
        BrandMark()
        Spacer(modifier = Modifier.width(7.dp))
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text(
                text = "Sure",
                color = SureCureDarkGreen,
                fontSize = 39.sp,
                lineHeight = 47.sp,
                fontWeight = FontWeight.Bold,
                letterSpacing = (-1.2).sp
            )
            Text(
                text = "Cure",
                color = SureCureGreen,
                fontSize = 39.sp,
                lineHeight = 47.sp,
                fontWeight = FontWeight.Bold,
                letterSpacing = (-1.2).sp
            )
        }
    }
}

@Composable
private fun BrandMark() {
    Canvas(modifier = Modifier.size(30.dp)) {
        val pillWidth = size.width * 0.76f
        val pillHeight = size.height * 0.34f
        drawRoundRect(
            color = SureCureDarkGreen,
            topLeft = Offset(size.width * 0.10f, size.height * 0.34f),
            size = Size(pillWidth, pillHeight),
            cornerRadius = androidx.compose.ui.geometry.CornerRadius(pillHeight / 2),
            style = Stroke(width = size.width * 0.10f)
        )
        drawLine(
            color = SureCureGreen,
            start = Offset(size.width * 0.50f, size.height * 0.37f),
            end = Offset(size.width * 0.50f, size.height * 0.66f),
            strokeWidth = size.width * 0.08f,
            cap = StrokeCap.Round
        )
    }
}

@Composable
private fun ScanHistoryRow(
    entry: ScanHistoryEntry,
    onClick: () -> Unit
) {
    val isSafe = entry.safety == ScanSafety.SAFE
    val statusColor = if (isSafe) SureCureGreen else SureCureRed
    val rowBackground = if (isSafe) Color.Transparent else UnsafeRowBackground

    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(8.dp))
            .background(rowBackground)
            .clickable(onClick = onClick)
            .padding(horizontal = 7.dp, vertical = 7.dp),
        horizontalArrangement = Arrangement.Start,
        verticalAlignment = Alignment.CenterVertically
    ) {
        MedicineThumbnail(isSafe = isSafe)
        Spacer(modifier = Modifier.width(10.dp))
        Column(
            modifier = Modifier.weight(1f),
            verticalArrangement = Arrangement.Center
        ) {
            Text(
                text = if (isSafe) "Marked safe." else "Marked unsafe.",
                color = statusColor,
                fontSize = 16.sp,
                lineHeight = 20.sp,
                fontWeight = FontWeight.Bold,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )
            Spacer(modifier = Modifier.height(7.dp))
            Text(
                text = entry.scannedOn,
                color = SureCureGray,
                fontSize = 11.sp,
                lineHeight = 14.sp,
                maxLines = 2,
                overflow = TextOverflow.Ellipsis
            )
        }
    }
}

/** Visual stand-in for the scan thumbnails, which were referenced but not uploaded. */
@Composable
private fun MedicineThumbnail(isSafe: Boolean) {
    val backdrop = if (isSafe) SureCureMint else Color(0xFFFFD6D6)
    Box(
        modifier = Modifier
            .size(width = 58.dp, height = 48.dp)
            .clip(RoundedCornerShape(7.dp))
            .background(backdrop),
        contentAlignment = Alignment.Center
    ) {
        Canvas(modifier = Modifier.size(31.dp)) {
            val pillWidth = size.width * 0.80f
            val pillHeight = size.height * 0.40f
            drawRoundRect(
                color = if (isSafe) SureCureDarkGreen else SureCureRed,
                topLeft = Offset(size.width * 0.10f, size.height * 0.30f),
                size = Size(pillWidth, pillHeight),
                cornerRadius = androidx.compose.ui.geometry.CornerRadius(pillHeight / 2)
            )
            drawLine(
                color = Color.White,
                start = Offset(size.width * 0.50f, size.height * 0.31f),
                end = Offset(size.width * 0.50f, size.height * 0.69f),
                strokeWidth = size.width * 0.045f,
                cap = StrokeCap.Round
            )
        }
    }
}

@Composable
private fun CameraButton(onClick: () -> Unit) {
    Box(
        modifier = Modifier
            .size(64.dp)
            .clip(CircleShape)
            .background(SureCureGreen)
            .border(width = 1.dp, color = Color.Black, shape = CircleShape)
            .clickable(onClick = onClick),
        contentAlignment = Alignment.Center
    ) {
        CameraGlyph()
    }
}

@Composable
private fun CameraGlyph() {
    Canvas(modifier = Modifier.size(29.dp)) {
        val stroke = size.width * 0.085f
        drawRoundRect(
            color = Color.White,
            topLeft = Offset(size.width * 0.08f, size.height * 0.28f),
            size = Size(size.width * 0.84f, size.height * 0.58f),
            cornerRadius = androidx.compose.ui.geometry.CornerRadius(size.width * 0.09f),
            style = Stroke(width = stroke)
        )
        drawRoundRect(
            color = Color.White,
            topLeft = Offset(size.width * 0.26f, size.height * 0.15f),
            size = Size(size.width * 0.28f, size.height * 0.18f),
            cornerRadius = androidx.compose.ui.geometry.CornerRadius(size.width * 0.04f)
        )
        drawCircle(
            color = Color.White,
            radius = size.width * 0.15f,
            center = Offset(size.width * 0.50f, size.height * 0.57f),
            style = Stroke(width = stroke)
        )
    }
}
