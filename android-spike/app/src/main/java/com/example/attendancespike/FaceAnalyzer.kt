package com.example.attendancespike

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Rect
import android.graphics.RectF
import android.util.Log
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facedetector.FaceDetector
import com.google.mediapipe.tasks.vision.facedetector.FaceDetectorResult

/**
 * Kiosk-side face analyzer (MediaPipe **FaceDetector / BlazeFace** edition).
 *
 * The kiosk only needs a face *bounding box* to crop and embed — it does NOT
 * need the 478-point face mesh or head pose. We previously ran the full
 * FaceLandmarker every frame (~40 ms on the Exynos 7870, ~85-95% of the frame
 * budget). BlazeFace short-range is a ~230 KB detector that returns just the box
 * in a fraction of the time, roughly doubling steady-state FPS.
 *
 * Crop convention: the box from BlazeFace is fed to [Bitmap.cropFace], the SAME
 * helper enrollment uses, so kiosk and enrollment crops match → embeddings stay
 * comparable. (EnrollFaceAnalyzer keeps FaceLandmarker only for sweep pose, but
 * crops from this same BlazeFace box.)
 *
 * Embedder stays MobileFaceNet TFLite (only runs when a face is present).
 */
class FaceAnalyzer(
    private val detector: FaceDetector,
    val backend: String,
    private val embedder: FaceEmbedder,
    private val store: EmbeddingStore,
    private val onStatus: (String) -> Unit,
    private val onEvent: (KioskEvent) -> Unit
) : ImageAnalysis.Analyzer {

    init {
        Log.i(TAG, "FaceDetector backend = $backend")
    }

    @Volatile private var pendingEnrollName: String? = null
    @Volatile private var pendingEnrollId: String? = null

    // Sliding window of the last N high-confidence match IDs. We only declare
    // a real match when the last [CONFIRMATION_FRAMES] frames all picked the
    // SAME person above threshold. Kills single-frame false positives where
    // someone else's face momentarily scores higher than ours.
    private val recentMatches = ArrayDeque<String>()

    // Embedder throttle state. The embedder (~85 ms) is what drags FPS down to
    // ~6 when a face is present; detect alone is ~32 ms. We cap embeds to a few
    // per second and, on skipped frames, re-emit the last decision so the box
    // overlay still tracks at full detector FPS. All touched only on the single
    // analysisExecutor thread.
    private var lastEmbedAtMs = 0L
    private var lastConfirmed = false
    private var lastMatchId: String? = null
    private var lastMatchName: String? = null

    /** Dev shortcut for the EnrollActivity "Test capture" path. */
    fun captureNextAsEnrollment(id: String, name: String = id) {
        pendingEnrollId = id
        pendingEnrollName = name
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
        val result: FaceDetectorResult = try {
            val mpImage = BitmapImageBuilder(bitmap).build()
            detector.detect(mpImage)
        } catch (e: Exception) {
            Log.e(TAG, "FaceDetector.detect failed", e)
            bitmap.recycle()
            imageProxy.close()
            return
        }
        val detectMs = System.currentTimeMillis() - t0

        lastEmbedMs = 0
        try {
            handleResult(bitmap, result, detectMs)
        } finally {
            // handleResult recycles the bitmap on its normal return paths (as
            // early as possible); this is the backstop so an exception in there
            // can't leak the ~full-frame bitmap. isRecycled guards double-free.
            if (!bitmap.isRecycled) bitmap.recycle()
            imageProxy.close()
            recordPerf(detectMs, lastEmbedMs)
        }
    }

    // --- Lightweight rolling FPS / latency instrumentation (INFO level) ------
    // All accessed only on the single analysisExecutor thread.
    private var perfFrames = 0
    private var perfWindowStart = 0L
    private var perfDetectSum = 0L
    private var perfEmbedSum = 0L
    private var perfEmbedSamples = 0
    private var lastEmbedMs = 0L

    private fun recordPerf(detectMs: Long, embedMs: Long) {
        if (perfWindowStart == 0L) perfWindowStart = System.currentTimeMillis()
        perfFrames++
        perfDetectSum += detectMs
        if (embedMs > 0) { perfEmbedSum += embedMs; perfEmbedSamples++ }
        if (perfFrames < PERF_LOG_EVERY) return

        val now = System.currentTimeMillis()
        val elapsed = (now - perfWindowStart).coerceAtLeast(1)
        val fps = perfFrames * 1000f / elapsed
        val detAvg = perfDetectSum.toFloat() / perfFrames
        val embAvg = if (perfEmbedSamples > 0) perfEmbedSum.toFloat() / perfEmbedSamples else 0f
        Log.i(TAG, "PERF backend=$backend fps=%.1f detectAvg=%.1fms embedAvg=%.1fms (embed %d/%d frames)"
            .format(fps, detAvg, embAvg, perfEmbedSamples, perfFrames))
        perfFrames = 0; perfDetectSum = 0; perfEmbedSum = 0; perfEmbedSamples = 0; perfWindowStart = now
    }

    private fun handleResult(bitmap: Bitmap, result: FaceDetectorResult, detectMs: Long) {
        val detections = result.detections()
        if (detections.isNullOrEmpty()) {
            onStatus("No face  detect=${detectMs}ms  enrolled=${store.size()}")
            onEvent(KioskEvent.NoFace)
            recentMatches.clear()  // face gone → drop the confirmation window
            // Reset throttle so the NEXT face embeds on its very first frame
            // (instant recognition start), and clear any stale match display.
            lastEmbedAtMs = 0L
            lastConfirmed = false
            lastMatchId = null
            lastMatchName = null
            bitmap.recycle()
            return
        }

        // Pick the largest detection (most likely the person in front of the kiosk).
        // boundingBox() is in pixel coordinates of the (rotated) input bitmap.
        val largest = detections.maxByOrNull { it.boundingBox().width() * it.boundingBox().height() }!!
        val box = largest.boundingBox()

        val bw = bitmap.width.toFloat()
        val bh = bitmap.height.toFloat()
        val normRect = RectF(
            (box.left / bw).coerceIn(0f, 1f),
            (box.top / bh).coerceIn(0f, 1f),
            (box.right / bw).coerceIn(0f, 1f),
            (box.bottom / bh).coerceIn(0f, 1f)
        )
        val sourceAspect = bw / bh

        // Embedder throttle: only crop+embed every EMBED_MIN_INTERVAL_MS. On a
        // skipped frame we still emit a tracking event (box follows the face at
        // full detector FPS) reusing the last match decision. The enrollment
        // "test capture" path (pendingEnrollId) bypasses the throttle so a
        // requested capture always fires.
        val now = System.currentTimeMillis()
        val mustEmbed = pendingEnrollId != null
        if (!mustEmbed && now - lastEmbedAtMs < EMBED_MIN_INTERVAL_MS) {
            if (lastConfirmed && lastMatchName != null) {
                onEvent(KioskEvent.Match(lastMatchId!!, lastMatchName!!, "IN", normRect, sourceAspect))
            } else {
                onEvent(KioskEvent.FaceVisibleNoMatch(normRect, sourceAspect))
            }
            bitmap.recycle()
            return
        }
        lastEmbedAtMs = now

        val pixelBox = Rect(
            box.left.toInt(), box.top.toInt(), box.right.toInt(), box.bottom.toInt()
        )
        val crop = bitmap.cropFace(pixelBox)
        if (crop == null) {
            onStatus("Face out of bounds  detect=${detectMs}ms")
            onEvent(KioskEvent.NoFace)
            bitmap.recycle()
            return
        }

        val t1 = System.currentTimeMillis()
        val embedding = embedder.embed(crop)
        val embedMs = System.currentTimeMillis() - t1
        lastEmbedMs = embedMs
        crop.recycle()
        bitmap.recycle()

        val enrollId = pendingEnrollId
        val enrollName = pendingEnrollName
        if (enrollId != null && enrollName != null) {
            store.enrollMulti(enrollId, enrollName, listOf(embedding))
            pendingEnrollId = null
            pendingEnrollName = null
            onStatus(
                "Enrolled '$enrollId'  total=${store.size()}\n" +
                "detect=${detectMs}ms  embed=${embedMs}ms"
            )
            onEvent(KioskEvent.Match(enrollId, enrollName, "IN", normRect, sourceAspect))
            return
        }

        val (matchId, matchName, score) = store.bestMatch(embedding)

        // Slide the per-frame top candidate into the confirmation window.
        val candidate = if (matchId != null && score >= MATCH_THRESHOLD) matchId else ""
        recentMatches.addLast(candidate)
        while (recentMatches.size > CONFIRMATION_FRAMES) recentMatches.removeFirst()

        val confirmed = recentMatches.size >= CONFIRMATION_FRAMES &&
                candidate.isNotEmpty() &&
                recentMatches.all { it == candidate }

        if (confirmed && matchName != null) {
            // Remember the decision so throttle-skipped frames keep the
            // recognition card up and the box tracking the same person.
            lastConfirmed = true
            lastMatchId = matchId
            lastMatchName = matchName
            onStatus(
                "MATCH '$matchId'  score=${"%.3f".format(score)}  (confirmed ${recentMatches.size}/$CONFIRMATION_FRAMES)\n" +
                "detect=${detectMs}ms  embed=${embedMs}ms"
            )
            onEvent(KioskEvent.Match(matchId!!, matchName, "IN", normRect, sourceAspect))
        } else {
            lastConfirmed = false
            lastMatchId = null
            lastMatchName = null
            val tag = if (candidate.isNotEmpty()) "pending '${candidate}'" else "no match"
            onStatus(
                "$tag  best='${matchId ?: "-"}' score=${"%.3f".format(score)}  " +
                "window=[${recentMatches.joinToString(",")}]\n" +
                "detect=${detectMs}ms  embed=${embedMs}ms"
            )
            onEvent(KioskEvent.FaceVisibleNoMatch(normRect, sourceAspect))
        }
    }

    fun close() {
        // The detector is an app-scoped singleton (AttendanceApp.faceDetector),
        // intentionally NOT closed here so it survives MainActivity recreation and
        // returning from enrollment doesn't pay a cold GPU-graph init every time.
        // AttendanceApp.onTerminate frees it.
    }

    companion object {
        private const val TAG = "AttendanceSpike"

        // MobileFaceNet cosine threshold. 0.55 is conservative-ish — combined
        // with multi-frame confirmation, it kills most false positives while
        // keeping true matches inside the typical same-person band of 0.5-0.75.
        private const val MATCH_THRESHOLD = 0.55f
        // Consecutive frames that must pick the SAME person above threshold
        // before declaring a match.
        private const val CONFIRMATION_FRAMES = 3
        // Emit one rolling PERF summary every this many analyzed frames.
        private const val PERF_LOG_EVERY = 30
        // Minimum gap between embeds while a face is present. Detect still runs
        // every frame (box tracks smoothly); only the ~85ms embed is throttled.
        // 150ms ⇒ up to ~6 embeds/sec — enough to confirm a match (3 frames) in
        // well under a second while keeping recognition-moment FPS high.
        private const val EMBED_MIN_INTERVAL_MS = 150L

        /**
         * Build a BlazeFace short-range [FaceDetector] (GPU, falling back to CPU).
         * Heavy GPU-graph init — call off the main thread. Returns (detector, backend).
         * Shared app-wide (see AttendanceApp.faceDetector) by both kiosk and the
         * enrollment crop path so crops stay consistent.
         */
        fun buildDetector(context: Context): Pair<FaceDetector, String> {
            fun create(delegate: Delegate): FaceDetector {
                val baseOpts = BaseOptions.builder()
                    .setModelAssetPath("blaze_face_short_range.tflite")
                    .setDelegate(delegate)
                    .build()
                val opts = FaceDetector.FaceDetectorOptions.builder()
                    .setBaseOptions(baseOpts)
                    .setRunningMode(RunningMode.IMAGE)
                    .setMinDetectionConfidence(0.6f)
                    .build()
                return FaceDetector.createFromOptions(context, opts)
            }
            // Measured on the Exynos 7870 / Mali-T830: GPU ~32ms/26fps vs
            // CPU ~46ms/18fps for this BlazeFace model. GPU wins, so prefer it.
            return try {
                create(Delegate.GPU) to "GPU"
            } catch (e: Throwable) {
                Log.w(TAG, "GPU detector failed, falling back to CPU", e)
                create(Delegate.CPU) to "CPU"
            }
        }

        /**
         * Prime the native graph + compile GPU shaders with one blank frame so
         * the first real camera frame after the kiosk opens isn't a stall.
         * Safe off the main thread; must not run concurrently with a real
         * detect() on the same detector.
         */
        fun warmUp(detector: FaceDetector) {
            try {
                val blank = Bitmap.createBitmap(256, 256, Bitmap.Config.ARGB_8888)
                val t0 = System.currentTimeMillis()
                detector.detect(BitmapImageBuilder(blank).build())
                blank.recycle()
                Log.i(TAG, "Detector warm-up took ${System.currentTimeMillis() - t0}ms")
            } catch (e: Exception) {
                Log.w(TAG, "Detector warm-up failed (will still try real frames)", e)
            }
        }
    }
}

/**
 * Structured signal the kiosk UI listens to.
 * [faceRect] is normalized [0,1] against the camera frame.
 * [sourceAspect] = bitmapWidth / bitmapHeight so the overlay can replicate
 * PreviewView's FILL_CENTER scaling.
 */
sealed class KioskEvent {
    object NoFace : KioskEvent()
    data class FaceVisibleNoMatch(val faceRect: RectF, val sourceAspect: Float) : KioskEvent()
    data class Match(
        val empId: String,
        val name: String,
        val event: String,
        val faceRect: RectF,
        val sourceAspect: Float
    ) : KioskEvent()
}
