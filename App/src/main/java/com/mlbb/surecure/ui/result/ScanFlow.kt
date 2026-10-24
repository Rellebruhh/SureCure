package com.mlbb.surecure.ui.result

import com.mlbb.surecure.ml.ClassifierResult
import com.mlbb.surecure.ml.Recognition
import com.mlbb.surecure.rules.FdaTables
import com.mlbb.surecure.rules.PackDate
import com.mlbb.surecure.rules.ScanInput
import com.mlbb.surecure.rules.Verdict
import com.mlbb.surecure.rules.VerdictLevel

/**
 * Logic behind the scan-result flow (Figure 1, stages 1 to 3), kept free of Compose so it can
 * be unit tested. The screen is ScanResultScreen.kt.
 */
object ScanFlow {

    /** Stage 1: what to ask the user after the photo model runs. */
    sealed interface ProductStep {
        /** In scope and confident: show the product with "Yes, that's it" and "Not this?". */
        data class Confirm(val label: String) : ProductStep

        /** In scope but not confident: "Which one is it?" with up to three choices + "None of these". */
        data class Pick(val labels: List<String>) : ProductStep

        /** Not recognized (or the model failed): manual selection from every known product. */
        data object ChooseFromList : ProductStep
    }

    fun productStep(result: ClassifierResult?): ProductStep = when (result?.status) {
        Recognition.ACCEPTED -> ProductStep.Confirm(result.top1.label)
        Recognition.LOW_CONFIDENCE -> ProductStep.Pick(result.top3.map { it.label })
        Recognition.NOT_RECOGNIZED, null -> ProductStep.ChooseFromList
    }

    /**
     * Products for manual selection: the photo model's products first (labels.json order), then
     * every other product in the FDA tables, so the counterfeit lot check also runs for brands
     * the model doesn't know (Alaxan FR, Neozep Forte, ...).
     */
    fun productChoices(modelLabels: List<String>, tables: FdaTables): List<String> =
        modelLabels + tables.knownProducts.filter { it !in modelLabels }

    /** "Myra_E" -> "Myra E", "Tuseran_Forte" -> "Tuseran Forte". The label itself is what the engine uses. */
    fun displayName(label: String): String = label.replace('_', ' ')

    /** Stage 2 input as typed by the user, before parsing. */
    data class CodeEntry(
        val lot: String = "",
        val regNo: String = "",
        val expiry: String = "",
        val manufactured: String = "",
    )

    sealed interface Parsed {
        data class Ok(val scan: ScanInput) : Parsed

        /** Field name -> message, for dates that were typed but couldn't be read. */
        data class Errors(val fields: Map<String, String>) : Parsed
    }

    /** Builds the engine input. Blank fields count as "not entered"; unreadable dates are errors. */
    fun parse(product: String?, entry: CodeEntry): Parsed {
        val errors = mutableMapOf<String, String>()
        val exp = entry.expiry.trim().takeIf { it.isNotEmpty() }?.let {
            PackDate.parsePrinted(it, endOfMonth = true).also { d ->
                if (d == null) errors["expiry"] = DATE_HELP
            }
        }
        val mfg = entry.manufactured.trim().takeIf { it.isNotEmpty() }?.let {
            PackDate.parsePrinted(it, endOfMonth = false).also { d ->
                if (d == null) errors["manufactured"] = DATE_HELP
            }
        }
        if (errors.isNotEmpty()) return Parsed.Errors(errors)
        return Parsed.Ok(
            ScanInput(
                stage1Product = product,
                lot = entry.lot.trim().takeIf { it.isNotEmpty() },
                regNo = entry.regNo.trim().takeIf { it.isNotEmpty() },
                mfgDate = mfg,
                expDate = exp,
            )
        )
    }

    const val DATE_HELP = "Couldn't read this date. Type it as printed, e.g. MAR 2028 or 03/2028."

    /**
     * Stage 3 headline. Red has two causes: a lot listed in an FDA counterfeit advisory, or codes
     * that belong to a different product. Only the first may say it matches an advisory.
     */
    fun headline(verdict: Verdict): String =
        if (verdict.level == VerdictLevel.RED && !verdict.counterfeitLotMatch) {
            RED_MISMATCH_HEADLINE
        } else {
            verdict.level.headline
        }

    const val RED_MISMATCH_HEADLINE = "Codes belong to a different product"
}
