package com.example.attendancespike

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Rect
import android.util.Log
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import kotlin.math.PI
import kotlin.math.asin
import kotlin.math.atan2

/**
 * Enrollment-specific analyzer (MediaPipe Face Landmarker edition).
 *
 * Replaces what used to be ML Kit in ACCURATE mode. We get yaw + pitch from
 * the facial transformation matrix that MediaPipe outputs alongside the
 * landmarks, so the 4-step sweep wizard still drives off real head pose.
 */
class EnrollFaceAnalyzer(
    context: Context,
    private val embedder: FaceEmbedder
) : ImageAnalysis.Analyzer {

    data class Frame(
        val faceBox: Rect?,
        val yaw: Float,
        val pitch: Float,
        val capturedEmbedding: FloatArray?,
        val detectMs: Long
    )

    var listener: ((Frame) -> Unit)? = null

    @Volatile var captureNext = false

    private val landmarker: FaceLandmarker

    init {
        fun create(delegate: Delegate): FaceLandmarker {
            val baseOpts = BaseOptions.builder()
                .setModelAssetPath("face_landmarker.task")
                .setDelegate(delegate)
                .build()
            val opts = FaceLandmarker.FaceLandmarkerOptions.builder()
                .setBaseOptions(baseOpts)
                .setRunningMode(RunningMode.IMAGE)
                .setNumFaces(1)
                .setMinFaceDetectionConfidence(0.5f)
                .setOutputFaceBlendshapes(false)
                .setOutputFacialTransformationMatrixes(true)  // → head pose
                .build()
            return FaceLandmarker.createFromOptions(context, opts)
        }
        landmarker = try {
            create(Delegate.GPU)
        } catch (e: Throwable) {
            Log.w(TAG, "GPU landmarker failed in enrollment, falling back to CPU", e)
            create(Delegate.CPU)
        }
    }

    @SuppressLint("UnsafeOptInUsageError")
    override fun analyze(imageProxy: ImageProxy) {
        val rotation = imageProxy.imageInfo.rotationDegrees
        val bitmap: Bitmap = try {
            imageProxy.toRgbaBitmap().rotated(rotation)
        } catch (e: Exception) {
            Log.e(TAG, "Bitmap conversion failed", e)
            imageProxy.close()
            return
        }

        val t0 = System.currentTimeMillis()
        val result = try {
            landmarker.detect(BitmapImageBuilder(bitmap).build())
        } catch (e: Exception) {
            Log.e(TAG, "detect failed", e)
            bitmap.recycle()
            imageProxy.close()
            return
        }
        val detectMs = System.currentTimeMillis() - t0

        val landmarks = result.faceLandmarks()
        if (landmarks.isNullOrEmpty()) {
            listener?.invoke(Frame(null, 0f, 0f, null, detectMs))
            bitmap.recycle()
            imageProxy.close()
            return
        }
        val lms = landmarks[0]

        // Bounding box from landmark extents.
        var minX = 1f; var maxX = 0f; var minY = 1f; var maxY = 0f
        for (lm in lms) {
            if (lm.x() < minX) minX = lm.x()
            if (lm.x() > maxX) maxX = lm.x()
            if (lm.y() < minY) minY = lm.y()
            if (lm.y() > maxY) maxY = lm.y()
        }
        val faceBox = Rect(
            (minX * bitmap.width).toInt(),
            (minY * bitmap.height).toInt(),
            (maxX * bitmap.width).toInt(),
            (maxY * bitmap.height).toInt()
        )

        // Pose from the 4x4 transformation matrix.
        val transforms = result.facialTransformationMatrixes().orElse(emptyList())
        val (yaw, pitch) = if (transforms.isNotEmpty())
            yawPitchFromMatrix(transforms[0]) else 0f to 0f

        // Optional capture for sweep step.
        var embedding: FloatArray? = null
        if (captureNext) {
            captureNext = false
            val crop = bitmap.cropFace(faceBox)
            if (crop != null) {
                try {
                    embedding = embedder.embed(crop)
                } catch (e: Exception) {
                    Log.e(TAG, "Embedding failed", e)
                }
                crop.recycle()
            }
        }

        listener?.invoke(Frame(faceBox, yaw, pitch, embedding, detectMs))
        bitmap.recycle()
        imageProxy.close()
    }

    /**
     * Extract head yaw + pitch (degrees) from MediaPipe's 4x4 column-major
     * facial transformation matrix. Sign convention here (verified on-device):
     *   pitch > 0 = looking up,  pitch < 0 = looking down
     *   yaw   > 0 = head turned to subject's left
     */
    private fun yawPitchFromMatrix(m: FloatArray): Pair<Float, Float> {
        if (m.size < 16) return 0f to 0f
        // Column-major: m[col*4 + row]. Upper-left 3x3 is the rotation.
        val r12 = m[9].coerceIn(-1f, 1f)
        val pitch = asin(r12) * 180f / PI.toFloat()
        val yaw = atan2(m[8], m[10]) * 180f / PI.toFloat()
        return yaw to pitch
    }

    fun close() {
        try { landmarker.close() } catch (_: Exception) {}
    }

    companion object {
        private const val TAG = "EnrollAnalyzer"
    }
}
