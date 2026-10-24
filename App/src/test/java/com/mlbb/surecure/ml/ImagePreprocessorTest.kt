package com.mlbb.surecure.ml

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Checks ImagePreprocessor against outputs made by PIL (the training pipeline's library):
 * EXIF orientation via ImageOps.exif_transpose, letterbox + Image.resize(BILINEAR).
 * Fixtures: src/test/resources/preprocess, written by scripts/make_preprocess_fixtures.py.
 * Expected: identical bytes, not just close.
 */
class ImagePreprocessorTest {

    private val fill = intArrayOf(124, 116, 104)

    private class Case(val name: String, val w: Int, val h: Int, val orientation: Int, val kind: String)

    private fun resource(name: String): ByteArray =
        javaClass.classLoader!!.getResourceAsStream("preprocess/$name")!!.use { it.readBytes() }

    private fun cases(kind: String): List<Case> =
        String(resource("index.txt")).lines().filter { it.isNotBlank() }.map {
            val p = it.split(" ")
            Case(p[0], p[1].toInt(), p[2].toInt(), p[3].toInt(), p[4])
        }.filter { it.kind == kind }

    private fun toArgb(rgb: ByteArray): IntArray = IntArray(rgb.size / 3) { i ->
        (0xFF shl 24) or ((rgb[i * 3].toInt() and 0xFF) shl 16) or
            ((rgb[i * 3 + 1].toInt() and 0xFF) shl 8) or (rgb[i * 3 + 2].toInt() and 0xFF)
    }

    @Test
    fun letterboxResizeMatchesPil() {
        val all = cases("letterbox")
        assertTrue(all.isNotEmpty())
        for (c in all) {
            val actual = ImagePreprocessor.letterboxResize(
                toArgb(resource("${c.name}.in")), c.w, c.h, c.orientation, fill
            )
            val expected = resource("${c.name}.out")
            assertEquals("${c.name}: size", expected.size, actual.size)
            val diffs = expected.indices.count { expected[it] != actual[it] }
            val first = expected.indices.firstOrNull { expected[it] != actual[it] }
            assertEquals(
                "${c.name}: $diffs bytes differ from PIL, first at byte $first", 0, diffs
            )
        }
    }

    @Test
    fun exifOrientationMatchesPil() {
        for (c in cases("orient")) {
            val argb = toArgb(resource("${c.name}.in"))
            val view = ImagePreprocessor.Oriented(c.w, c.h, c.orientation)
            val expected = toArgb(resource("${c.name}.out"))
            assertEquals("${c.name}: pixel count", expected.size, view.width * view.height)
            for (y in 0 until view.height) for (x in 0 until view.width) {
                assertEquals(
                    "${c.name}: pixel ($x, $y)",
                    expected[y * view.width + x],
                    argb[view.sourceIndex(x, y)]
                )
            }
        }
    }

    @Test
    fun tensorIsChannelFirstAndScaledTo01() {
        // 1 x 1 white pixel: letterbox is the pixel itself, resized to 224 x 224 white.
        val t = ImagePreprocessor.toInputTensor(intArrayOf(0xFFFFFFFF.toInt()), 1, 1, 1, fill)
        assertEquals(3 * 224 * 224, t.size)
        assertTrue(t.all { it == 1f })
        // Red-only pixel: channel 0 is 1, channels 1 and 2 are 0.
        val r = ImagePreprocessor.toInputTensor(intArrayOf(0xFFFF0000.toInt()), 1, 1, 1, fill)
        val plane = 224 * 224
        assertEquals(1f, r[0])
        assertEquals(0f, r[plane])
        assertEquals(0f, r[2 * plane])
    }
}
