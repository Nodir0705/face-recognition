package com.example.attendancespike

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import kotlin.math.min

/**
 * Techy face-tracking overlay: corner brackets at each detected face box.
 * Coordinates come in NORMALIZED (0-1) relative to the camera frame.
 *
 *   ┌─        ─┐
 *
 *                  (gaps between corners, not a full rectangle)
 *
 *   └─        ─┘
 *
 * Color shifts from white (detected) to green (matched). Mirrors horizontally
 * for front-facing cameras so the box tracks the user's face naturally.
 */
class FaceBoxOverlay @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    enum class State { HIDDEN, DETECTED, MATCHED }

    private val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = dp(3.5f)
        strokeCap = Paint.Cap.ROUND
        color = COLOR_DETECTED
    }

    /** Face box in normalized [0,1] coords relative to the camera frame. */
    private var faceRect: RectF? = null
    private var sourceAspect: Float = 4f / 3f
    private var state: State = State.HIDDEN

    /** Front camera is mirrored in the preview — flip x so the box tracks the user. */
    var mirror: Boolean = true

    fun show(normalizedRect: RectF, sourceAspect: Float, matched: Boolean) {
        faceRect = normalizedRect
        this.sourceAspect = sourceAspect
        state = if (matched) State.MATCHED else State.DETECTED
        paint.color = if (matched) COLOR_MATCHED else COLOR_DETECTED
        invalidate()
    }

    fun hide() {
        if (state == State.HIDDEN) return
        state = State.HIDDEN
        faceRect = null
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val rect = faceRect ?: return
        val vw = width.toFloat()
        val vh = height.toFloat()
        if (vw <= 0f || vh <= 0f) return

        // Replicate PreviewView's FILL_CENTER: scale the camera frame so it
        // covers the view, with equal cropping on the over-flowing axis.
        val viewAspect = vw / vh
        val scaledW: Float; val scaledH: Float
        val offsetX: Float; val offsetY: Float
        if (sourceAspect > viewAspect) {
            // Camera wider than view → height fills, width overflows (cropped sides)
            scaledH = vh
            scaledW = vh * sourceAspect
            offsetX = (scaledW - vw) / 2f
            offsetY = 0f
        } else {
            // Camera narrower than view → width fills, height overflows (cropped top/bottom)
            scaledW = vw
            scaledH = vw / sourceAspect
            offsetX = 0f
            offsetY = (scaledH - vh) / 2f
        }

        // Mirror in normalized space, then map to view coords.
        val nLeft = if (mirror) 1f - rect.right else rect.left
        val nRight = if (mirror) 1f - rect.left else rect.right

        val left = nLeft * scaledW - offsetX
        val right = nRight * scaledW - offsetX
        val top = rect.top * scaledH - offsetY
        val bottom = rect.bottom * scaledH - offsetY

        val w = right - left
        val h = bottom - top
        if (w <= 0f || h <= 0f) return
        val bracketLen = min(w, h) * BRACKET_RATIO

        // Top-left corner
        canvas.drawLine(left, top, left + bracketLen, top, paint)
        canvas.drawLine(left, top, left, top + bracketLen, paint)
        // Top-right corner
        canvas.drawLine(right - bracketLen, top, right, top, paint)
        canvas.drawLine(right, top, right, top + bracketLen, paint)
        // Bottom-left corner
        canvas.drawLine(left, bottom - bracketLen, left, bottom, paint)
        canvas.drawLine(left, bottom, left + bracketLen, bottom, paint)
        // Bottom-right corner
        canvas.drawLine(right - bracketLen, bottom, right, bottom, paint)
        canvas.drawLine(right, bottom - bracketLen, right, bottom, paint)
    }

    private fun dp(v: Float) = v * resources.displayMetrics.density

    companion object {
        private const val BRACKET_RATIO = 0.22f
        private val COLOR_DETECTED = Color.WHITE
        private val COLOR_MATCHED = 0xFF16A34A.toInt()  // success green
    }
}
