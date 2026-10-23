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
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.ui.platform.LocalContext
import com.mlbb.surecure.data.repository.ScanHistoryRepository
import com.mlbb.surecure.ui.dashboard.DashboardScreen
import com.mlbb.surecure.ui.dashboard.ScanHistorySection
import com.mlbb.surecure.ui.dashboard.groupScanHistory
import com.mlbb.surecure.ui.theme.SureCureTheme
import androidx.compose.runtime.getValue
import androidx.compose.runtime.setValue
import kotlinx.coroutines.launch

class MainActivity : ComponentActivity() {
    @RequiresApi(Build.VERSION_CODES.O)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            SureCureTheme {
                val context = LocalContext.current
                val repository = remember {
                    ScanHistoryRepository(context)
                }
                val scope = rememberCoroutineScope()

                var sections by remember {
                    mutableStateOf(emptyList<ScanHistorySection>())
                }
                var isRefreshing by remember {
                    mutableStateOf(false)
                }

                // Initial load. The repository caches the result.
                LaunchedEffect(repository) {
                    sections = groupScanHistory(repository.loadEntries())
                }
                DashboardScreen(
                    sections = sections,
                    isRefreshing = isRefreshing,
                    onRefreshClick = {
                        if (!isRefreshing) {
                            scope.launch {
                                isRefreshing = true
                                try {
                                    val entries = repository.loadEntries(
                                        forceRefresh = true
                                    )
                                    sections = groupScanHistory(entries)
                                } finally {
                                    isRefreshing = false
                                }
                            }
                        }
                    }
                )
            }
        }
    }
}