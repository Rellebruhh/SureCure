package com.mlbb.surecure.ui.camera

import android.net.Uri
import android.widget.ImageView
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView

@Composable
fun PhotoReviewScreen(
    frontPhoto: Uri,
    codePanelPhoto: Uri,
    onRetake: () -> Unit,
    onDone: () -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(20.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        Text(
            text = "Review your photos",
            style = MaterialTheme.typography.headlineSmall
        )

        Text(
            text = "Medicine front",
            style = MaterialTheme.typography.titleMedium
        )

        CapturedPhoto(
            uri = frontPhoto,
            contentDescription = "Medicine front photo"
        )

        Text(
            text = "Barcode, lot number and expiry panel",
            style = MaterialTheme.typography.titleMedium
        )

        CapturedPhoto(
            uri = codePanelPhoto,
            contentDescription = "Medicine code panel photo"
        )

        Button(
            onClick = onDone,
            modifier = Modifier.fillMaxWidth()
        ) {
            Text("Check this medicine")
        }

        OutlinedButton(
            onClick = onRetake,
            modifier = Modifier.fillMaxWidth()
        ) {
            Text("Retake photos")
        }

        Text(
            text = "The images are saved locally.",
            style = MaterialTheme.typography.bodySmall
        )
    }
}

@Composable
private fun CapturedPhoto(
    uri: Uri,
    contentDescription: String
) {
    AndroidView(
        factory = { context ->
            ImageView(context).apply {
                scaleType = ImageView.ScaleType.FIT_CENTER
                setImageURI(uri)
                this.contentDescription = contentDescription
            }
        },
        update = { imageView ->
            imageView.setImageURI(uri)
            imageView.contentDescription = contentDescription
        },
        modifier = Modifier
            .fillMaxWidth()
            .height(240.dp)
            .clip(RoundedCornerShape(12.dp))
    )
}
