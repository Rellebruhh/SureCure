package com.mlbb.surecure.ml

/**
 * Turns decoded pixels into the model input, following the model contract in meta.json:
 * EXIF rotate upright, RGB, letterbox to a centered square filled with fill_rgb,
 * resize to 224 x 224 bilinear, divide by 255, channel-first (1 x 3 x 224 x 224).
 *
 * The resize reproduces Pillow's Image.resize(BILINEAR) bit for bit (same filter support,
 * same 22-bit fixed-point rounding, horizontal pass first). Android's own scaler does not
 * widen the filter when shrinking, so a large photo would come out sharper than the images
 * the model was calibrated on.
 *
 * Pure Kotlin on purpose: no Android imports, so it runs in plain JVM unit tests.
 */
object ImagePreprocessor {

    const val INPUT_SIZE = 224

    /** Bit shift of Pillow's 8-bit fixed-point resampling (32 - 8 - 2). */
    private const val PRECISION_BITS = 22

    /**
     * @param argb pixels as returned by Bitmap.getPixels, row by row, before any rotation.
     * @param exifOrientation EXIF orientation tag value 1..8 (ExifInterface.ORIENTATION_*).
     * @return float32 values in 0..1, layout [channel][y][x], RGB order.
     */
    fun toInputTensor(
        argb: IntArray,
        width: Int,
        height: Int,
        exifOrientation: Int,
        fillRgb: IntArray,
        size: Int = INPUT_SIZE,
    ): FloatArray {
        val rgb = letterboxResize(argb, width, height, exifOrientation, fillRgb, size)
        val plane = size * size
        val out = FloatArray(3 * plane)
        for (i in 0 until plane) {
            out[i] = (rgb[i * 3].toInt() and 0xFF) / 255f
            out[plane + i] = (rgb[i * 3 + 1].toInt() and 0xFF) / 255f
            out[2 * plane + i] = (rgb[i * 3 + 2].toInt() and 0xFF) / 255f
        }
        return out
    }

    /**
     * Upright, letterboxed and resized image as 8-bit RGB, layout [y][x][channel].
     * The square canvas is never allocated: pixels outside the photo read as fill_rgb.
     */
    fun letterboxResize(
        argb: IntArray,
        width: Int,
        height: Int,
        exifOrientation: Int,
        fillRgb: IntArray,
        size: Int = INPUT_SIZE,
    ): ByteArray {
        require(argb.size >= width * height) { "pixel array is smaller than width x height" }
        require(fillRgb.size == 3) { "fill_rgb needs 3 values" }
        val oriented = Oriented(width, height, exifOrientation)
        val side = maxOf(oriented.width, oriented.height)
        val offsetX = (side - oriented.width) / 2
        val offsetY = (side - oriented.height) / 2
        val fill = (fillRgb[0] shl 16) or (fillRgb[1] shl 8) or fillRgb[2]

        // Canvas pixel (x, y) of the side x side letterboxed square, as 0xRRGGBB.
        fun canvas(x: Int, y: Int): Int {
            val ox = x - offsetX
            val oy = y - offsetY
            if (ox < 0 || oy < 0 || ox >= oriented.width || oy >= oriented.height) return fill
            return argb[oriented.sourceIndex(ox, oy)] and 0xFFFFFF
        }

        // Horizontal pass: side rows x size columns, rounded to 8 bits like Pillow.
        val horizontal = ByteArray(side * size * 3)
        if (side == size) {
            for (y in 0 until side) for (x in 0 until size) {
                putRgb(horizontal, (y * size + x) * 3, canvas(x, y))
            }
        } else {
            val k = Coefficients(side, size)
            for (y in 0 until side) {
                for (x in 0 until size) {
                    var r = 1 shl (PRECISION_BITS - 1)
                    var g = r
                    var b = r
                    val start = k.start[x]
                    for (t in 0 until k.count[x]) {
                        val w = k.weights[x * k.taps + t]
                        val p = canvas(start + t, y)
                        r += ((p shr 16) and 0xFF) * w
                        g += ((p shr 8) and 0xFF) * w
                        b += (p and 0xFF) * w
                    }
                    val o = (y * size + x) * 3
                    horizontal[o] = clip8(r)
                    horizontal[o + 1] = clip8(g)
                    horizontal[o + 2] = clip8(b)
                }
            }
        }
        if (side == size) return horizontal

        // Vertical pass: size x size.
        val out = ByteArray(size * size * 3)
        val k = Coefficients(side, size)
        for (y in 0 until size) {
            val start = k.start[y]
            for (x in 0 until size) {
                var r = 1 shl (PRECISION_BITS - 1)
                var g = r
                var b = r
                for (t in 0 until k.count[y]) {
                    val w = k.weights[y * k.taps + t]
                    val i = ((start + t) * size + x) * 3
                    r += (horizontal[i].toInt() and 0xFF) * w
                    g += (horizontal[i + 1].toInt() and 0xFF) * w
                    b += (horizontal[i + 2].toInt() and 0xFF) * w
                }
                val o = (y * size + x) * 3
                out[o] = clip8(r)
                out[o + 1] = clip8(g)
                out[o + 2] = clip8(b)
            }
        }
        return out
    }

    /** Upright view of a width x height image, matching PIL.ImageOps.exif_transpose. */
    internal class Oriented(private val srcW: Int, private val srcH: Int, orientation: Int) {
        private val o = if (orientation in 1..8) orientation else 1
        val width = if (o >= 5) srcH else srcW
        val height = if (o >= 5) srcW else srcH

        fun sourceIndex(x: Int, y: Int): Int {
            val sx: Int
            val sy: Int
            when (o) {
                2 -> { sx = srcW - 1 - x; sy = y }
                3 -> { sx = srcW - 1 - x; sy = srcH - 1 - y }
                4 -> { sx = x; sy = srcH - 1 - y }
                5 -> { sx = y; sy = x }
                6 -> { sx = y; sy = srcH - 1 - x }
                7 -> { sx = srcW - 1 - y; sy = srcH - 1 - x }
                8 -> { sx = srcW - 1 - y; sy = x }
                else -> { sx = x; sy = y }
            }
            return sy * srcW + sx
        }
    }

    /** Pillow's precompute_coeffs + normalize_coeffs_8bpc for the bilinear (triangle) filter. */
    private class Coefficients(inSize: Int, outSize: Int) {
        val taps: Int
        val start = IntArray(outSize)
        val count = IntArray(outSize)
        val weights: IntArray

        init {
            val scale = inSize.toDouble() / outSize
            val filterScale = maxOf(scale, 1.0)
            val support = 1.0 * filterScale
            taps = kotlin.math.ceil(support).toInt() * 2 + 1
            weights = IntArray(outSize * taps)
            val pre = DoubleArray(taps)
            for (xx in 0 until outSize) {
                val center = (xx + 0.5) * scale
                val ss = 1.0 / filterScale
                val xmin = maxOf((center - support + 0.5).toInt(), 0)
                val xmax = minOf((center + support + 0.5).toInt(), inSize) - xmin
                var sum = 0.0
                for (x in 0 until xmax) {
                    val w = triangle((x + xmin - center + 0.5) * ss)
                    pre[x] = w
                    sum += w
                }
                for (x in 0 until xmax) {
                    val w = if (sum != 0.0) pre[x] / sum else pre[x]
                    weights[xx * taps + x] = if (w < 0) {
                        (-0.5 + w * (1 shl PRECISION_BITS)).toInt()
                    } else {
                        (0.5 + w * (1 shl PRECISION_BITS)).toInt()
                    }
                }
                start[xx] = xmin
                count[xx] = xmax
            }
        }

        private fun triangle(v: Double): Double {
            val x = if (v < 0) -v else v
            return if (x < 1.0) 1.0 - x else 0.0
        }
    }

    private fun clip8(v: Int): Byte {
        val s = v shr PRECISION_BITS
        return (if (s < 0) 0 else if (s > 255) 255 else s).toByte()
    }

    private fun putRgb(dst: ByteArray, at: Int, rgb: Int) {
        dst[at] = (rgb shr 16).toByte()
        dst[at + 1] = (rgb shr 8).toByte()
        dst[at + 2] = rgb.toByte()
    }
}
