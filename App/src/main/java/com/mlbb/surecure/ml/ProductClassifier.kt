package com.mlbb.surecure.ml

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.media.ExifInterface
import android.net.Uri
import org.json.JSONArray
import org.json.JSONObject
import java.io.ByteArrayInputStream
import java.io.Closeable
import java.nio.FloatBuffer
import kotlin.math.exp

/**
 * Photo model: says which of the 20 supported products a pack photo shows, or that it
 * doesn't recognize it. It is not a counterfeit detector; the verdict comes from the rules
 * engine after the user confirms the product.
 *
 * Covers steps 2 to 4 of meta.json's decision_order. Step 1 (a decoded barcode found in the
 * product table skips the photo model) belongs to the caller.
 *
 * Loading reads a ~46 MB model, and classify() takes a few hundred ms on a phone, so call
 * both off the main thread (e.g. withContext(Dispatchers.Default)). One instance can be
 * shared; ONNX Runtime sessions are thread-safe. Close it when the app no longer needs it.
 */
class ProductClassifier private constructor(
    private val env: OrtEnvironment,
    private val session: OrtSession,
    val meta: ModelMeta,
) : Closeable {

    /** Classifies an image from the camera or gallery; applies its EXIF orientation. */
    fun classify(context: Context, uri: Uri): ClassifierResult {
        val bytes = context.contentResolver.openInputStream(uri)?.use { it.readBytes() }
            ?: error("Cannot open $uri")
        return classifyEncoded(bytes)
    }

    /** Classifies an encoded image file (JPEG, PNG, ...); applies its EXIF orientation. */
    fun classifyEncoded(bytes: ByteArray): ClassifierResult {
        val orientation = runCatching {
            ExifInterface(ByteArrayInputStream(bytes)).getAttributeInt(
                ExifInterface.TAG_ORIENTATION,
                ExifInterface.ORIENTATION_NORMAL
            )
        }.getOrDefault(ExifInterface.ORIENTATION_NORMAL)
        val options = BitmapFactory.Options().apply {
            inPreferredConfig = Bitmap.Config.ARGB_8888
        }
        val bitmap = BitmapFactory.decodeByteArray(bytes, 0, bytes.size, options)
            ?: error("Cannot decode image")
        return try {
            classify(bitmap, orientation)
        } finally {
            bitmap.recycle()
        }
    }

    /**
     * Classifies a decoded bitmap.
     * @param exifOrientation pass the photo's EXIF orientation if the bitmap was decoded
     * without applying it; pass ORIENTATION_NORMAL if the bitmap is already upright.
     */
    fun classify(
        bitmap: Bitmap,
        exifOrientation: Int = ExifInterface.ORIENTATION_NORMAL
    ): ClassifierResult {
        val argbBitmap = if (bitmap.config == Bitmap.Config.ARGB_8888) {
            bitmap
        } else {
            bitmap.copy(Bitmap.Config.ARGB_8888, false)
        }
        val pixels = IntArray(argbBitmap.width * argbBitmap.height)
        argbBitmap.getPixels(pixels, 0, argbBitmap.width, 0, 0, argbBitmap.width, argbBitmap.height)
        if (argbBitmap !== bitmap) argbBitmap.recycle()
        val input = ImagePreprocessor.toInputTensor(
            pixels, bitmap.width, bitmap.height, exifOrientation, meta.fillRgb
        )
        return classifyTensor(input)
    }

    /** Runs an already preprocessed 1 x 3 x 224 x 224 input (used by the golden tests). */
    fun classifyTensor(input: FloatArray): ClassifierResult {
        val size = ImagePreprocessor.INPUT_SIZE.toLong()
        require(input.size.toLong() == 3 * size * size) { "input must be 3 x 224 x 224" }
        val logits: FloatArray
        val energy: Float
        OnnxTensor.createTensor(env, FloatBuffer.wrap(input), longArrayOf(1, 3, size, size))
            .use { tensor ->
                session.run(mapOf(INPUT_NAME to tensor)).use { out ->
                    @Suppress("UNCHECKED_CAST")
                    logits = (out.get(LOGITS_NAME).get().value as Array<FloatArray>)[0]
                    energy = (out.get(meta.guardOutput).get().value as FloatArray)[0]
                }
            }
        return decide(logits, energy)
    }

    /** decision_order steps 2 to 4. */
    private fun decide(logits: FloatArray, energy: Float): ClassifierResult {
        check(logits.size == meta.labels.size) { "model has ${logits.size} outputs, labels.json has ${meta.labels.size}" }
        val probs = softmax(logits, meta.temperature)
        val order = probs.indices.sortedByDescending { probs[it] }
        val top3 = order.take(3).map { ProductScore(meta.labels[it], probs[it].toFloat()) }
        val status = when {
            energy > meta.guardThreshold -> Recognition.NOT_RECOGNIZED
            probs[order[0]] < meta.confidenceThreshold -> Recognition.LOW_CONFIDENCE
            else -> Recognition.ACCEPTED
        }
        return ClassifierResult(status, top3, energy, logits, meta.runName)
    }

    override fun close() {
        session.close()
    }

    companion object {
        const val ASSET_DIR = "model"
        private const val INPUT_NAME = "image"
        private const val LOGITS_NAME = "logits"

        /** Loads surecure.onnx, meta.json and labels.json from assets/model/ (same run). */
        fun create(context: Context, assetDir: String = ASSET_DIR): ProductClassifier {
            val assets = context.applicationContext.assets
            val meta = ModelMeta.parse(
                assets.open("$assetDir/meta.json").bufferedReader().use { it.readText() },
                assets.open("$assetDir/labels.json").bufferedReader().use { it.readText() },
            )
            val modelBytes = assets.open("$assetDir/surecure.onnx").use { it.readBytes() }
            val env = OrtEnvironment.getEnvironment()
            val session = OrtSession.SessionOptions().use { options ->
                env.createSession(modelBytes, options)
            }
            val outputs = session.outputNames
            check(LOGITS_NAME in outputs && meta.guardOutput in outputs) {
                "surecure.onnx outputs $outputs do not match meta.json (run ${meta.runName}); " +
                    "model files from different runs?"
            }
            return ProductClassifier(env, session, meta)
        }

        internal fun softmax(logits: FloatArray, temperature: Double): DoubleArray {
            val z = DoubleArray(logits.size) { logits[it] / temperature }
            val max = z.max()
            val e = DoubleArray(z.size) { exp(z[it] - max) }
            val sum = e.sum()
            return DoubleArray(z.size) { e[it] / sum }
        }
    }
}

/** Recognition outcome; tells the UI which screen to show. */
enum class Recognition {
    /** Show the top-1 product with a "Not this?" link. */
    ACCEPTED,

    /** Show the top-3 picker: "Which one is it?" + "None of these". */
    LOW_CONFIDENCE,

    /** Out-of-scope guard rejected the photo: product list + manual code entry. */
    NOT_RECOGNIZED,
}

/** @param label product name exactly as in labels.json, e.g. "Myra_E", "Tuseran_Forte". */
data class ProductScore(val label: String, val confidence: Float)

data class ClassifierResult(
    val status: Recognition,
    /** Three most likely products, highest first, confidences after temperature scaling. */
    val top3: List<ProductScore>,
    /** Out-of-scope score; higher means less like the training products. */
    val energy: Float,
    val logits: FloatArray,
    val runName: String,
) {
    val top1: ProductScore get() = top3[0]

    override fun equals(other: Any?): Boolean =
        other is ClassifierResult && status == other.status && top3 == other.top3 &&
            energy == other.energy && logits.contentEquals(other.logits) && runName == other.runName

    override fun hashCode(): Int = 31 * (31 * status.hashCode() + top3.hashCode()) + logits.contentHashCode()
}

/** The parts of meta.json the app needs. Values are calibrated per run; never edit by hand. */
data class ModelMeta(
    val runName: String,
    val labels: List<String>,
    val fillRgb: IntArray,
    val temperature: Double,
    val confidenceThreshold: Double,
    val guardOutput: String,
    val guardThreshold: Double,
) {
    companion object {
        fun parse(metaJson: String, labelsJson: String): ModelMeta {
            val meta = JSONObject(metaJson)
            val guard = meta.optJSONObject("guard")
                ?: error("meta.json has no 'guard' block: this is not an energy-guard run")
            check(guard.getString("score") == "energy") {
                "meta.json guard score is '${guard.getString("score")}', the app expects 'energy'"
            }
            val labels = JSONArray(labelsJson).let { a -> List(a.length()) { a.getString(it) } }
            val fill = meta.getJSONArray("fill_rgb")
            return ModelMeta(
                runName = meta.getString("run_name"),
                labels = labels,
                fillRgb = IntArray(3) { fill.getInt(it) },
                temperature = meta.getDouble("temperature"),
                confidenceThreshold = meta.getDouble("confidence_threshold"),
                guardOutput = guard.getString("onnx_output"),
                guardThreshold = guard.getDouble("threshold"),
            )
        }
    }

    override fun equals(other: Any?): Boolean =
        other is ModelMeta && runName == other.runName && labels == other.labels &&
            fillRgb.contentEquals(other.fillRgb) && temperature == other.temperature &&
            confidenceThreshold == other.confidenceThreshold && guardOutput == other.guardOutput &&
            guardThreshold == other.guardThreshold

    override fun hashCode(): Int = runName.hashCode()
}
