package com.example.attendancespike

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.graphics.PorterDuff
import android.graphics.PorterDuffXfermode
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View

/**
 * Face-ID style oval guide overlay. Mirrors the SVG ellipse in enroll.html:
 *   .oval-ring     → white stroke, default
 *   .oval-ring.fit → green stroke, when a face is correctly fitting
 *
 * Also dims the area OUTSIDE the oval so the user's face is naturally drawn
 * into the center — a small visual cue without being heavy-handed.
 */
class FaceGuideView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    enum class State { IDLE, FACE_VISIBLE, FIT, RECOGNIZED }

    private val ringPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        color = Color.WHITE
        alpha = 220
        strokeWidth = dp(3f)
    }

    private val dimPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.BLACK
        alpha = 110
    }

    private val cutoutPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        xfermode = PorterDuffXfermode(PorterDuff.Mode.CLEAR)
    }

    private val ovalRect = RectF()
    private val clipPath = Path()

    var state: State = State.IDLE
        set(value) {
            if (field == value) return
            field = value
            updateRingForState()
            invalidate()
        }

    init {
        // Required so the CLEAR xfermode in onDraw actually punches a hole
        // instead of being composited on the window background.
        setLayerType(LAYER_TYPE_HARDWARE, null)
        updateRingForState()
    }

    private fun updateRingForState() {
        when (state) {
            State.IDLE, State.FACE_VISIBLE -> {
                ringPaint.color = Color.WHITE
                ringPaint.alpha = 200
                ringPaint.strokeWidth = dp(3f)
            }
            State.FIT, State.RECOGNIZED -> {
                ringPaint.color = SUCCESS_GREEN
                ringPaint.alpha = 255
                ringPaint.strokeWidth = dp(5f)
            }
        }
    }

    override fun onSizeChanged(w: Int, h: Int, oldw: Int, oldh: Int) {
        super.onSizeChanged(w, h, oldw, oldh)
        // Face-shaped oval (~3:4 ratio, taller than wide), sized to 75% of the
        // shorter container dimension so it works in both portrait (camera-on-top
        // tall area) and landscape (camera-on-top wide area).
        val shorter = if (w < h) w.toFloat() else h.toFloat()
        val ovalH = shorter * 0.85f
        val ovalW = ovalH * 0.75f
        val cx = w / 2f
        val cy = h * 0.5f
        ovalRect.set(cx - ovalW / 2f, cy - ovalH / 2f, cx + ovalW / 2f, cy + ovalH / 2f)
        clipPath.reset()
        clipPath.addOval(ovalRect, Path.Direction.CW)
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        // 1. Dim everything outside the oval.
        canvas.drawRect(0f, 0f, width.toFloat(), height.toFloat(), dimPaint)
        // 2. Punch a clear hole inside the oval (PorterDuff CLEAR).
        canvas.drawOval(ovalRect, cutoutPaint)
        // 3. Stroke the oval ring on top.
        canvas.drawOval(ovalRect, ringPaint)
    }

    fun isPointInsideOval(px: Float, py: Float): Boolean {
        val a = ovalRect.width() / 2f
        val b = ovalRect.height() / 2f
        val cx = ovalRect.centerX()
        val cy = ovalRect.centerY()
        if (a <= 0f || b <= 0f) return false
        val nx = (px - cx) / a
        val ny = (py - cy) / b
        return nx * nx + ny * ny <= 1f
    }

    private fun dp(v: Float) = v * resources.displayMetrics.density

    companion object {
        private const val SUCCESS_GREEN = 0xFF16A34A.toInt()
    }
}
