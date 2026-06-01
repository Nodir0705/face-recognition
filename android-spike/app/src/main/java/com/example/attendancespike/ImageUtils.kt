package com.example.attendancespike

import android.graphics.Bitmap
import android.graphics.Matrix
import android.graphics.Rect
import androidx.camera.core.ImageProxy

/**
 * Convert a CameraX ImageProxy (configured for OUTPUT_IMAGE_FORMAT_RGBA_8888)
 * into an ARGB_8888 Bitmap. Handles row stride padding.
 */
fun ImageProxy.toRgbaBitmap(): Bitmap {
    val plane = planes[0]
    val buffer = plane.buffer
    val pixelStride = plane.pixelStride
    val rowStride = plane.rowStride
    val rowPadding = rowStride - pixelStride * width

    val bmpWidth = width + rowPadding / pixelStride
    val padded = Bitmap.createBitmap(bmpWidth, height, Bitmap.Config.ARGB_8888)
    padded.copyPixelsFromBuffer(buffer)

    return if (rowPadding == 0) padded
    else {
        val trimmed = Bitmap.createBitmap(padded, 0, 0, width, height)
        padded.recycle()
        trimmed
    }
}

fun Bitmap.rotated(degrees: Int): Bitmap {
    if (degrees == 0) return this
    val m = Matrix().apply { postRotate(degrees.toFloat()) }
    val out = Bitmap.createBitmap(this, 0, 0, width, height, m, true)
    if (out !== this) this.recycle()
    return out
}

/**
 * Crop a SQUARE region centered on the detected face box, sized to the larger
 * of (width, height) × (1 + 2*marginRatio). MobileFaceNet was trained on
 * aligned 112×112 crops; using a consistent square keeps the resulting
 * embeddings stable across detectors (BlazeFace vs ML Kit produce slightly
 * different bounding-box aspect ratios) and across frames (jittery boxes).
 */
fun Bitmap.cropFace(box: Rect, marginRatio: Float = 0.25f): Bitmap? {
    val w = box.width()
    val h = box.height()
    if (w <= 0 || h <= 0) return null
    // Side length of the square crop: the larger box dimension, padded.
    val cx = (box.left + box.right) / 2
    val cy = (box.top + box.bottom) / 2
    val baseSide = maxOf(w, h)
    val side = (baseSide * (1f + 2f * marginRatio)).toInt()
    val half = side / 2
    val left = (cx - half).coerceAtLeast(0)
    val top = (cy - half).coerceAtLeast(0)
    val right = (cx + half).coerceAtMost(width)
    val bottom = (cy + half).coerceAtMost(height)
    val cw = right - left
    val ch = bottom - top
    if (cw <= 0 || ch <= 0) return null
    return Bitmap.createBitmap(this, left, top, cw, ch)
}
