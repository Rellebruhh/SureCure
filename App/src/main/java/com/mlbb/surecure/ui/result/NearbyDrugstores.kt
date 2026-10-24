package com.mlbb.surecure.ui.result

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.location.Location
import android.location.LocationManager
import android.os.CancellationSignal
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.core.location.LocationManagerCompat
import com.mlbb.surecure.drugstores.DRUGSTORE_DISCLAIMER
import com.mlbb.surecure.drugstores.DrugstoreBranch
import com.mlbb.surecure.drugstores.DrugstoreDirectory
import com.mlbb.surecure.drugstores.DrugstoreLoader
import com.mlbb.surecure.drugstores.Municipality
import com.mlbb.surecure.drugstores.NearbyBranch
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeoutOrNull
import kotlin.coroutines.resume

private sealed interface NearbyMode {
    data object Start : NearbyMode
    data object Locating : NearbyMode
    data class Near(val branches: List<NearbyBranch>) : NearbyMode
    data class Pick(val reason: String?) : NearbyMode
    data class InCity(val city: Municipality, val branches: List<DrugstoreBranch>) : NearbyMode
}

/**
 * "Find a nearby drugstore", offered on amber and red results.
 *
 * Takes one approximate (coarse) location reading when the user taps the button, sorts the
 * bundled chain branches by distance on the phone, and keeps nothing: the location is not stored
 * or sent anywhere. If permission is refused or no location comes back, the user searches a city.
 */
@Composable
fun NearbyDrugstores(onBack: () -> Unit) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    var directory by remember { mutableStateOf<DrugstoreDirectory?>(null) }
    var mode by remember { mutableStateOf<NearbyMode>(NearbyMode.Start) }

    LaunchedEffect(Unit) {
        directory = withContext(Dispatchers.IO) { DrugstoreLoader.load(context) }
    }

    fun locate() {
        mode = NearbyMode.Locating
        scope.launch {
            val location = currentCoarseLocation(context)
            val dir = directory
            mode = if (location != null && dir != null) {
                NearbyMode.Near(dir.nearestBranches(location.latitude, location.longitude))
            } else {
                NearbyMode.Pick("Your location isn't available right now. Search for your city instead.")
            }
        }
    }

    val permission = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) locate() else mode = NearbyMode.Pick("Without location, search for your city instead.")
    }

    Text("Nearby drugstores", style = MaterialTheme.typography.titleLarge)
    Text(DRUGSTORE_DISCLAIMER, style = MaterialTheme.typography.bodySmall)

    when (val m = mode) {
        NearbyMode.Start -> {
            Text(
                "SureCure can use your approximate location once to sort the list. " +
                    "It isn't saved or sent anywhere.",
                style = MaterialTheme.typography.bodyMedium
            )
            Button(
                onClick = {
                    if (hasCoarsePermission(context)) locate()
                    else permission.launch(Manifest.permission.ACCESS_COARSE_LOCATION)
                },
                enabled = directory != null,
                modifier = Modifier.fillMaxWidth()
            ) { Text("Use my approximate location") }
            OutlinedButton(onClick = { mode = NearbyMode.Pick(null) }, modifier = Modifier.fillMaxWidth()) {
                Text("Search for a city instead")
            }
        }

        NearbyMode.Locating -> Row(verticalAlignment = Alignment.CenterVertically) {
            CircularProgressIndicator(modifier = Modifier.size(24.dp))
            Spacer(Modifier.padding(start = 12.dp))
            Text("Finding your approximate location...")
        }

        is NearbyMode.Near -> {
            if (m.branches.isEmpty()) Text("No branches in the list.")
            m.branches.forEach { BranchRow(it.branch, ScanFlow.formatDistance(it.distanceKm)) }
            TextButton(onClick = { mode = NearbyMode.Pick(null) }) { Text("Search for a city instead") }
        }

        is NearbyMode.Pick -> CityPicker(
            directory = directory,
            reason = m.reason,
            onCity = { city -> mode = NearbyMode.InCity(city, directory?.branchesIn(city.psgc).orEmpty()) },
        )

        is NearbyMode.InCity -> {
            Text("${m.city.name}, ${m.city.province}", style = MaterialTheme.typography.titleMedium)
            m.branches.forEach { BranchRow(it, null) }
            TextButton(onClick = { mode = NearbyMode.Pick(null) }) { Text("Choose another city") }
        }
    }

    OutlinedButton(onClick = onBack, modifier = Modifier.fillMaxWidth()) { Text("Back to the result") }
}

@Composable
private fun CityPicker(directory: DrugstoreDirectory?, reason: String?, onCity: (Municipality) -> Unit) {
    var query by remember { mutableStateOf("") }
    reason?.let { Text(it, style = MaterialTheme.typography.bodyMedium) }
    OutlinedTextField(
        value = query,
        onValueChange = { query = it },
        label = { Text("City or municipality") },
        singleLine = true,
        modifier = Modifier.fillMaxWidth()
    )
    val matches = directory?.findMunicipalities(query).orEmpty().take(20)
    if (query.isNotBlank() && matches.isEmpty()) {
        Text("No chain branches listed for that name.", style = MaterialTheme.typography.bodySmall)
    }
    matches.forEach { city ->
        OutlinedButton(onClick = { onCity(city) }, modifier = Modifier.fillMaxWidth()) {
            Text("${city.name}, ${city.province}")
        }
    }
}

@Composable
private fun BranchRow(branch: DrugstoreBranch, distance: String?) {
    Column(verticalArrangement = Arrangement.spacedBy(2.dp), modifier = Modifier.fillMaxWidth()) {
        Text(
            branch.branch + (distance?.let { " · $it" } ?: ""),
            style = MaterialTheme.typography.bodyLarge,
            fontWeight = FontWeight.SemiBold
        )
        Text(branch.address, style = MaterialTheme.typography.bodySmall)
    }
}

private fun hasCoarsePermission(context: Context): Boolean =
    ContextCompat.checkSelfPermission(context, Manifest.permission.ACCESS_COARSE_LOCATION) ==
        PackageManager.PERMISSION_GRANTED

/** One coarse reading (network provider), at most 15 s; falls back to the last known one. Null if none. */
private suspend fun currentCoarseLocation(context: Context): Location? {
    if (!hasCoarsePermission(context)) return null
    val lm = context.getSystemService(Context.LOCATION_SERVICE) as? LocationManager ?: return null
    val provider = when {
        LocationManagerCompat.hasProvider(lm, LocationManager.NETWORK_PROVIDER) &&
            lm.isProviderEnabled(LocationManager.NETWORK_PROVIDER) -> LocationManager.NETWORK_PROVIDER
        else -> LocationManager.PASSIVE_PROVIDER
    }
    val fresh = try {
        withTimeoutOrNull(15_000) {
            suspendCancellableCoroutine<Location?> { cont ->
                val signal = CancellationSignal()
                cont.invokeOnCancellation { signal.cancel() }
                LocationManagerCompat.getCurrentLocation(
                    lm, provider, signal, ContextCompat.getMainExecutor(context)
                ) { location -> if (cont.isActive) cont.resume(location) }
            }
        }
    } catch (e: SecurityException) {
        null
    }
    if (fresh != null) return fresh
    return try {
        lm.getLastKnownLocation(provider)
    } catch (e: SecurityException) {
        null
    }
}
