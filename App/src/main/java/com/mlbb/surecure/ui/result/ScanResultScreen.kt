package com.mlbb.surecure.ui.result

import android.net.Uri
import android.widget.ImageView
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
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
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import com.mlbb.surecure.ml.ClassifierResult
import com.mlbb.surecure.rules.FdaTables
import com.mlbb.surecure.rules.Verdict
import com.mlbb.surecure.rules.VerdictLevel
import com.mlbb.surecure.rules.verify
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

// Same values as the dashboard's brand colors (private there).
private val SureCureGreen = Color(0xFF14AE5C)
private val SureCureDarkGreen = Color(0xFF02542D)
private val SureCureRed = Color(0xFFC00F0C)
private val SureCureAmber = Color(0xFFB26A00)
private val GreenBackground = Color(0xFFDEFFE8)
private val RedBackground = Color(0xFFFFE8E8)
private val AmberBackground = Color(0xFFFFF3DC)

private sealed interface Phase {
    data object Loading : Phase
    data class Product(val step: ScanFlow.ProductStep) : Phase
    data class Codes(val product: String?) : Phase
    data class Result(val verdict: Verdict, val product: String?) : Phase
    data class Nearby(val result: Result) : Phase
}

/**
 * Scan result flow after photo review (Figure 1):
 * Stage 1 product identification + manual selection, Stage 2 code entry (typed for now; ML Kit
 * reading will fill the same fields), Stage 3 verification rules and verdict.
 */
@Composable
fun ScanResultScreen(
    frontPhoto: Uri,
    codePanelPhoto: Uri,
    onDone: () -> Unit,
) {
    val context = LocalContext.current
    var phase by remember { mutableStateOf<Phase>(Phase.Loading) }
    var tables by remember { mutableStateOf<FdaTables?>(null) }
    var modelLabels by remember { mutableStateOf<List<String>>(emptyList()) }
    var notice by remember { mutableStateOf<String?>(null) }
    var entry by remember { mutableStateOf(ScanFlow.CodeEntry()) }

    LaunchedEffect(frontPhoto) {
        var result: ClassifierResult? = null
        try {
            tables = ScanEngines.tables(context)
            val classifier = ScanEngines.classifier(context)
            modelLabels = classifier.meta.labels
            result = withContext(Dispatchers.Default) { classifier.classify(context, frontPhoto) }
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            notice = "The photo couldn't be checked (${e.localizedMessage ?: "error"}). Choose the product below."
        }
        phase = Phase.Product(ScanFlow.productStep(result))
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(20.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        Text("Check this medicine", style = MaterialTheme.typography.headlineSmall)
        notice?.let { Text(it, color = SureCureRed, style = MaterialTheme.typography.bodyMedium) }

        when (val p = phase) {
            Phase.Loading -> Row(verticalAlignment = Alignment.CenterVertically) {
                CircularProgressIndicator(modifier = Modifier.size(24.dp))
                Spacer(Modifier.padding(start = 12.dp))
                Text("Identifying the product...")
            }

            is Phase.Product -> ProductStage(
                step = p.step,
                choices = tables?.let { ScanFlow.productChoices(modelLabels, it) } ?: modelLabels,
                onChosen = { phase = Phase.Codes(it) },
                onShowList = { phase = Phase.Product(ScanFlow.ProductStep.ChooseFromList) },
            )

            is Phase.Codes -> CodeStage(
                product = p.product,
                codePanelPhoto = codePanelPhoto,
                entry = entry,
                onEntryChange = { entry = it },
                onChangeProduct = { phase = Phase.Product(ScanFlow.ProductStep.ChooseFromList) },
                onCheck = { scan ->
                    val t = tables
                    if (t != null) phase = Phase.Result(verify(scan, t), p.product)
                    else notice = "The FDA tables couldn't be loaded."
                },
            )

            is Phase.Result -> VerdictStage(
                verdict = p.verdict,
                onEditCodes = { phase = Phase.Codes(p.product) },
                onFindDrugstore = if (ScanFlow.offersNearbyDrugstores(p.verdict)) {
                    { phase = Phase.Nearby(p) }
                } else null,
                onDone = onDone,
            )

            is Phase.Nearby -> NearbyDrugstores(onBack = { phase = p.result })
        }
    }
}

// --------------------------------------------------------------------------- Stage 1

@Composable
private fun ProductStage(
    step: ScanFlow.ProductStep,
    choices: List<String>,
    onChosen: (String?) -> Unit,
    onShowList: () -> Unit,
) {
    when (step) {
        is ScanFlow.ProductStep.Confirm -> {
            Text("This looks like", style = MaterialTheme.typography.titleMedium)
            Text(
                ScanFlow.displayName(step.label),
                style = MaterialTheme.typography.headlineMedium,
                fontWeight = FontWeight.Bold,
                color = SureCureDarkGreen
            )
            Button(
                onClick = { onChosen(step.label) },
                modifier = Modifier.fillMaxWidth(),
                colors = ButtonDefaults.buttonColors(containerColor = SureCureGreen)
            ) { Text("Yes, that's it") }
            TextButton(onClick = onShowList) { Text("Not this?") }
        }

        is ScanFlow.ProductStep.Pick -> {
            Text("Which one is it?", style = MaterialTheme.typography.titleMedium)
            step.labels.forEach { label ->
                OutlinedButton(onClick = { onChosen(label) }, modifier = Modifier.fillMaxWidth()) {
                    Text(ScanFlow.displayName(label))
                }
            }
            TextButton(onClick = onShowList) { Text("None of these") }
        }

        ScanFlow.ProductStep.ChooseFromList -> ProductList(choices, onChosen)
    }
}

@Composable
private fun ProductList(choices: List<String>, onChosen: (String?) -> Unit) {
    var query by remember { mutableStateOf("") }
    Text("Which medicine is it?", style = MaterialTheme.typography.titleMedium)
    OutlinedTextField(
        value = query,
        onValueChange = { query = it },
        label = { Text("Search") },
        singleLine = true,
        modifier = Modifier.fillMaxWidth()
    )
    val shown = choices.filter { ScanFlow.displayName(it).contains(query.trim(), ignoreCase = true) }
    shown.forEach { label ->
        OutlinedButton(onClick = { onChosen(label) }, modifier = Modifier.fillMaxWidth()) {
            Text(ScanFlow.displayName(label))
        }
    }
    TextButton(onClick = { onChosen(null) }) {
        Text("It's not listed / I don't know")
    }
}

// --------------------------------------------------------------------------- Stage 2

@Composable
private fun CodeStage(
    product: String?,
    codePanelPhoto: Uri,
    entry: ScanFlow.CodeEntry,
    onEntryChange: (ScanFlow.CodeEntry) -> Unit,
    onChangeProduct: () -> Unit,
    onCheck: (com.mlbb.surecure.rules.ScanInput) -> Unit,
) {
    var errors by remember { mutableStateOf<Map<String, String>>(emptyMap()) }

    Text(
        "Product: " + (product?.let { ScanFlow.displayName(it) } ?: "not identified"),
        style = MaterialTheme.typography.titleMedium
    )
    TextButton(onClick = onChangeProduct) { Text("Change product") }

    Text("Type the codes printed on the pack", style = MaterialTheme.typography.titleMedium)
    Photo(codePanelPhoto, "Code panel photo")
    Text(
        "Leave a field blank if it isn't printed or you can't read it.",
        style = MaterialTheme.typography.bodySmall
    )
    OutlinedTextField(
        value = entry.barcode,
        onValueChange = { onEntryChange(entry.copy(barcode = it)) },
        label = { Text("Barcode number (13 digits under the bars)") },
        singleLine = true,
        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
        modifier = Modifier.fillMaxWidth()
    )
    OutlinedTextField(
        value = entry.lot,
        onValueChange = { onEntryChange(entry.copy(lot = it)) },
        label = { Text("Lot / batch number") },
        singleLine = true,
        modifier = Modifier.fillMaxWidth()
    )
    OutlinedTextField(
        value = entry.regNo,
        onValueChange = { onEntryChange(entry.copy(regNo = it)) },
        label = { Text("FDA registration number (e.g. DR-XY12345)") },
        singleLine = true,
        modifier = Modifier.fillMaxWidth()
    )
    OutlinedTextField(
        value = entry.expiry,
        onValueChange = { onEntryChange(entry.copy(expiry = it)); errors = errors - "expiry" },
        label = { Text("Expiry date (e.g. MAR 2028)") },
        singleLine = true,
        isError = "expiry" in errors,
        supportingText = errors["expiry"]?.let { msg -> { Text(msg) } },
        modifier = Modifier.fillMaxWidth()
    )
    OutlinedTextField(
        value = entry.manufactured,
        onValueChange = { onEntryChange(entry.copy(manufactured = it)); errors = errors - "manufactured" },
        label = { Text("Manufacture date (optional)") },
        singleLine = true,
        isError = "manufactured" in errors,
        supportingText = errors["manufactured"]?.let { msg -> { Text(msg) } },
        modifier = Modifier.fillMaxWidth()
    )
    Button(
        onClick = {
            when (val parsed = ScanFlow.parse(product, entry)) {
                is ScanFlow.Parsed.Ok -> onCheck(parsed.scan)
                is ScanFlow.Parsed.Errors -> errors = parsed.fields
            }
        },
        modifier = Modifier.fillMaxWidth(),
        colors = ButtonDefaults.buttonColors(containerColor = SureCureGreen)
    ) { Text("Check against FDA records") }
}

// --------------------------------------------------------------------------- Stage 3

@Composable
private fun VerdictStage(
    verdict: Verdict,
    onEditCodes: () -> Unit,
    onFindDrugstore: (() -> Unit)?,
    onDone: () -> Unit,
) {
    val (color, background) = when (verdict.level) {
        VerdictLevel.RED -> SureCureRed to RedBackground
        VerdictLevel.AMBER -> SureCureAmber to AmberBackground
        VerdictLevel.GREEN -> SureCureGreen to GreenBackground
    }
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(12.dp))
            .background(background)
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        Text(
            ScanFlow.headline(verdict),
            style = MaterialTheme.typography.headlineSmall,
            fontWeight = FontWeight.Bold,
            color = color
        )
        Text(
            "Checked as: " + (verdict.productChecked?.let { ScanFlow.displayName(it) } ?: "unknown product"),
            style = MaterialTheme.typography.bodyMedium
        )
    }
    Text("Why", style = MaterialTheme.typography.titleMedium)
    verdict.reasons.forEach { Text("• $it", style = MaterialTheme.typography.bodyMedium) }
    if (verdict.level == VerdictLevel.GREEN) {
        Text(
            "This means the codes don't match any FDA counterfeit record we have. " +
                "It doesn't prove the medicine is genuine.",
            style = MaterialTheme.typography.bodySmall
        )
    }
    if (onFindDrugstore != null) {
        OutlinedButton(onClick = onFindDrugstore, modifier = Modifier.fillMaxWidth()) {
            Text("Find a nearby drugstore")
        }
    }
    OutlinedButton(onClick = onEditCodes, modifier = Modifier.fillMaxWidth()) { Text("Edit the codes") }
    Button(
        onClick = onDone,
        modifier = Modifier.fillMaxWidth(),
        colors = ButtonDefaults.buttonColors(containerColor = SureCureGreen)
    ) { Text("Done") }
}

@Composable
private fun Photo(uri: Uri, description: String) {
    AndroidView(
        factory = { ctx ->
            ImageView(ctx).apply {
                scaleType = ImageView.ScaleType.FIT_CENTER
                setImageURI(uri)
                contentDescription = description
            }
        },
        update = { it.setImageURI(uri) },
        modifier = Modifier
            .fillMaxWidth()
            .height(220.dp)
            .clip(RoundedCornerShape(12.dp))
    )
}
