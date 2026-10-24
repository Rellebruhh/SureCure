package com.mlbb.surecure.ml

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONObject
import org.junit.AfterClass
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.BeforeClass
import org.junit.Test
import org.junit.runner.RunWith
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs

/**
 * Acceptance test from the model contract, run on device with the app's bundled model.
 * Golden files (androidTest/assets/golden) come from the same training run as
 * main/assets/model; expected.json holds the PyTorch/ONNX reference outputs.
 *
 * 1. Each tensor_XX.bin straight into ONNX Runtime: logits within 1e-3, same top-1 and
 *    same accept decision.
 * 2. Each raw/XX_*.jpg through the Kotlin preprocessing: same top-1, logits within 0.1.
 * If 1 passes and 2 fails, the bug is in preprocessing (or JPEG decoding).
 */
@RunWith(AndroidJUnit4::class)
class ProductClassifierGoldenTest {

    companion object {
        private lateinit var classifier: ProductClassifier
        private lateinit var expected: JSONObject

        @BeforeClass
        @JvmStatic
        fun setUp() {
            val instrumentation = InstrumentationRegistry.getInstrumentation()
            classifier = ProductClassifier.create(instrumentation.targetContext)
            expected = JSONObject(
                instrumentation.context.assets.open("golden/expected.json")
                    .bufferedReader().use { it.readText() }
            )
        }

        @AfterClass
        @JvmStatic
        fun tearDown() {
            classifier.close()
        }
    }

    private val testAssets get() = InstrumentationRegistry.getInstrumentation().context.assets

    private fun images() = expected.getJSONArray("images").let { a ->
        List(a.length()) { a.getJSONObject(it) }
    }

    private fun expectedLogits(image: JSONObject) = image.getJSONArray("expected_logits").let { a ->
        FloatArray(a.length()) { a.getDouble(it).toFloat() }
    }

    private fun maxDiff(a: FloatArray, b: FloatArray): Float =
        a.indices.maxOf { abs(a[it] - b[it]) }

    @Test
    fun bundledModelIsTheGoldenRun() {
        val labels = expected.getJSONArray("labels").let { a -> List(a.length()) { a.getString(it) } }
        assertEquals("labels.json vs expected.json", labels, classifier.meta.labels)
        val guard = expected.getJSONObject("guard")
        assertEquals(guard.getDouble("threshold"), classifier.meta.guardThreshold, 0.0)
        assertEquals(expected.getDouble("temperature"), classifier.meta.temperature, 0.0)
        assertEquals(
            expected.getDouble("confidence_threshold"), classifier.meta.confidenceThreshold, 0.0
        )
    }

    @Test
    fun goldenTensorsMatchExpected() {
        val tolerance = expected.getDouble("tolerance_runtime_logits").toFloat()
        val failures = mutableListOf<String>()
        for (image in images()) {
            val bytes = testAssets.open("golden/" + image.getString("tensor_file")).use { it.readBytes() }
            val input = FloatArray(bytes.size / 4)
            ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(input)

            val result = classifier.classifyTensor(input)
            val diff = maxDiff(result.logits, expectedLogits(image))
            val energyDiff = abs(result.energy - image.getDouble("expected_energy").toFloat())
            val accepted = result.status == Recognition.ACCEPTED
            val line = "${image.getString("tensor_file")}: top1=${result.top1.label} " +
                "status=${result.status} logitDiff=$diff energyDiff=$energyDiff"
            println(line)
            if (diff > tolerance || energyDiff > tolerance ||
                result.top1.label != image.getString("expected_top1") ||
                accepted != image.getBoolean("expected_accept")
            ) failures += line
        }
        assertTrue("Golden tensor mismatches:\n" + failures.joinToString("\n"), failures.isEmpty())
    }

    @Test
    fun goldenRawImagesMatchExpected() {
        val tolerance = expected.getDouble("tolerance_preprocess_logits").toFloat()
        val failures = mutableListOf<String>()
        for (image in images()) {
            val bytes = testAssets.open("golden/" + image.getString("raw_file")).use { it.readBytes() }
            val result = classifier.classifyEncoded(bytes)
            val diff = maxDiff(result.logits, expectedLogits(image))
            val line = "${image.getString("raw_file")}: top1=${result.top1.label} " +
                "expected=${image.getString("expected_top1")} status=${result.status} logitDiff=$diff"
            println(line)
            if (result.top1.label != image.getString("expected_top1") || diff > tolerance) {
                failures += line
            }
        }
        assertTrue("Golden raw image mismatches:\n" + failures.joinToString("\n"), failures.isEmpty())
    }
}
