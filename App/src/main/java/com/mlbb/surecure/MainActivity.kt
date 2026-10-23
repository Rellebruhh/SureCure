package com.mlbb.surecure

import android.os.Build
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.activity.enableEdgeToEdge
import androidx.annotation.RequiresApi
import com.mlbb.surecure.ui.dashboard.DashboardScreen
import com.mlbb.surecure.ui.theme.SureCureTheme

class MainActivity : ComponentActivity() {
    @RequiresApi(Build.VERSION_CODES.O)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            SureCureTheme {
                DashboardScreen(onCameraClick = {
                    // Navigate to your camera/capture screen here.
                })
            }
        }
    }
}