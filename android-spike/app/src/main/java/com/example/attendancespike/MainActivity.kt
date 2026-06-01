package com.example.attendancespike

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.view.View
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/**
 * Kiosk activity — face-ID style.
 * Recognition pipeline runs natively; UI is fully native (no WebView).
 * Admin/enrollment is reached via the top-right gear icon.
 */
class MainActivity : AppCompatActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var faceBoxOverlay: FaceBoxOverlay
    private lateinit var recognitionGroup: LinearLayout
    private lateinit var recogName: TextView
    private lateinit var recogStatus: TextView
    private lateinit var hintText: TextView
    private lateinit var clockText: TextView
    private lateinit var gearIcon: ImageView

    private val mainHandler = Handler(Looper.getMainLooper())
    private val clockFmt = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault())
    private val timeOnlyFmt = SimpleDateFormat("HH:mm", Locale.getDefault())
    private val localDateFmt = SimpleDateFormat("yyyy-MM-dd", Locale.getDefault())
    @Volatile private var lastCloseoutDate: String? = null

    private var lastFaceSeenAt = System.currentTimeMillis()
    private var hintVisible = false

    private var lastRecogEmpId = ""
    private var lastRecogAt = 0L
    private val recogHideRunnable = Runnable { hideRecognition() }

    private var pendingEnrollIdCounter = 0

    private lateinit var analysisExecutor: ExecutorService
    private lateinit var faceAnalyzer: FaceAnalyzer
    private lateinit var embedder: FaceEmbedder
    private lateinit var store: EmbeddingStore

    private val cameraPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) startCamera()
        else Toast.makeText(this, "Camera permission required", Toast.LENGTH_LONG).show()
    }

    private val enrollLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) {
        // EnrollActivity writes directly to the shared store (AttendanceApp.store)
        // when its sweep completes, so MainActivity doesn't need to do anything
        // here. Recognition on the next frame will pick up the new embeddings.
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        // Kiosk immersive — hide status bar AND navigation bar; sticky means
        // a swipe shows them briefly then re-hides.
        @Suppress("DEPRECATION")
        window.decorView.systemUiVisibility = (
            View.SYSTEM_UI_FLAG_LAYOUT_STABLE
                or View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION
                or View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN
                or View.SYSTEM_UI_FLAG_HIDE_NAVIGATION
                or View.SYSTEM_UI_FLAG_FULLSCREEN
                or View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY
        )
        // Real kiosk: never sleep while activity is visible.
        window.addFlags(android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        previewView = findViewById(R.id.previewView)
        // TextureView path — required for legacy Camera HAL on Android 5.x Samsungs.
        previewView.implementationMode = PreviewView.ImplementationMode.COMPATIBLE

        faceBoxOverlay = findViewById(R.id.faceBoxOverlay)
        recognitionGroup = findViewById(R.id.recognitionGroup)
        recogName = findViewById(R.id.recogName)
        recogStatus = findViewById(R.id.recogStatus)
        hintText = findViewById(R.id.hintText)
        clockText = findViewById(R.id.clockText)
        gearIcon = findViewById(R.id.gearIcon)

        gearIcon.setOnClickListener {
            enrollLauncher.launch(Intent(this, SettingsActivity::class.java))
        }
        // Tap inside the camera area to dismiss the recognition card early.
        previewView.setOnClickListener {
            if (recognitionGroup.alpha > 0.5f) hideRecognition()
        }

        startClock()

        val app = application as AttendanceApp
        embedder = app.embedder
        store = app.store

        analysisExecutor = Executors.newSingleThreadExecutor()

        // Build the analyzer off the main thread: fetching the kiosk landmarker
        // can block while its GPU graph initialises + warms up (only on the very
        // first launch — it's an app-scoped singleton reused thereafter). Bind
        // the camera once it's ready so the UI never freezes.
        Thread({
            val analyzer = FaceAnalyzer(
                detector = app.faceDetector(),
                backend = app.kioskBackend,
                embedder = embedder,
                store = store,
                onStatus = { msg -> Log.d(TAG, msg) },
                onEvent = { event -> runOnUiThread { handleKioskEvent(event) } }
            )
            runOnUiThread {
                faceAnalyzer = analyzer
                if (hasCameraPermission()) startCamera()
                else cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
            }
        }, "kiosk-analyzer-init").start()
    }

    override fun onResume() {
        super.onResume()
        // EnrollActivity calls provider.unbindAll() to set up its own camera.
        // That nukes our binding, so we re-bind every time we come back to
        // the foreground. startCamera() is idempotent (unbindAll → bind).
        if (::faceAnalyzer.isInitialized && hasCameraPermission()) {
            Log.d(TAG, "onResume: re-binding camera; store has ${store.size()} person(s)")
            startCamera()
        }
    }

    // --- Kiosk UI ---------------------------------------------------------

    private fun startClock() {
        val tick = object : Runnable {
            override fun run() {
                clockText.text = clockFmt.format(Date())
                mainHandler.postDelayed(this, 1000)
            }
        }
        mainHandler.post(tick)
    }

    private fun handleKioskEvent(event: KioskEvent) {
        when (event) {
            is KioskEvent.NoFace -> {
                onNoFace()
                faceBoxOverlay.hide()
            }
            is KioskEvent.FaceVisibleNoMatch -> {
                onFaceVisible()
                faceBoxOverlay.show(event.faceRect, event.sourceAspect, matched = false)
            }
            is KioskEvent.Match -> {
                onFaceVisible()
                faceBoxOverlay.show(event.faceRect, event.sourceAspect, matched = true)
                maybeShowRecognition(event)
            }
        }
    }

    private fun onFaceVisible() {
        lastFaceSeenAt = System.currentTimeMillis()
        if (hintVisible) {
            hintVisible = false
            hintText.animate().alpha(0f).setDuration(400).start()
        }
    }

    private fun onNoFace() {
        if (!hintVisible && System.currentTimeMillis() - lastFaceSeenAt > 4000) {
            hintVisible = true
            hintText.animate().alpha(1f).setDuration(500).start()
        }
    }

    private fun maybeShowRecognition(match: KioskEvent.Match) {
        val now = System.currentTimeMillis()
        // Debounce — don't re-greet the same person inside one banner cycle.
        if (match.empId == lastRecogEmpId && now - lastRecogAt < GREETING_DURATION + 1000) return
        lastRecogEmpId = match.empId
        lastRecogAt = now

        val app = application as AttendanceApp
        val today = localDateFmt.format(Date(now))

        // Closeout: emit OUT events for anyone who had an open IN on a previous
        // day (kiosk restarted, or this is the first match of a new day).
        closeoutOldPresencesIfNeeded(today)

        // First-IN-of-day / last-OUT-of-day logic:
        //   first sighting today  → record IN event, notify
        //   repeat sighting today → silently bump last_seen_ts (no event, no sync)
        val isFirstToday = app.enrollmentDb.touchDailyPresence(
            match.empId, match.name, today, now
        )
        val statusText: String
        if (isFirstToday) {
            app.enrollmentDb.insertEvent(match.empId, match.name, "IN", now)
            Log.i(TAG, "IN recorded: ${match.empId} (${match.name}) @ ${Date(now)}")
            app.syncManager.enqueueSync()
            statusText = "${getString(R.string.event_in)}  ·  ${timeOnlyFmt.format(Date(now))}"
        } else {
            statusText = "Xush kelibsiz  ·  ${timeOnlyFmt.format(Date(now))}"
        }

        recogName.text = match.name
        recogStatus.text = statusText

        recognitionGroup.animate()
            .alpha(1f)
            .scaleX(1f).scaleY(1f)
            .setDuration(280)
            .withStartAction {
                recognitionGroup.scaleX = 0.85f
                recognitionGroup.scaleY = 0.85f
            }
            .start()
        mainHandler.removeCallbacks(recogHideRunnable)
        mainHandler.postDelayed(recogHideRunnable, GREETING_DURATION)
    }

    /**
     * Idempotent end-of-day closeout: any `daily_presence` row from before
     * [today] that isn't closed yet → emit an OUT event using last_seen_ts.
     * Runs once per day (gated by [lastCloseoutDate]).
     */
    private fun closeoutOldPresencesIfNeeded(today: String) {
        if (lastCloseoutDate == today) return
        lastCloseoutDate = today
        val app = application as AttendanceApp
        val open = app.enrollmentDb.openPresenceBefore(today)
        if (open.isEmpty()) return
        for (row in open) {
            app.enrollmentDb.insertEvent(row.empId, row.name, "OUT", row.lastSeenTs)
            app.enrollmentDb.markPresenceClosedOut(row.empId, row.date)
            Log.i(TAG, "OUT closeout: ${row.empId} (${row.name}) " +
                    "@ ${Date(row.lastSeenTs)} for day ${row.date}")
        }
        app.syncManager.enqueueSync()
    }

    private fun hideRecognition() {
        recognitionGroup.animate()
            .alpha(0f)
            .setDuration(280)
            .start()
        mainHandler.removeCallbacks(recogHideRunnable)
    }

    // --- Camera -----------------------------------------------------------

    private fun hasCameraPermission() = ContextCompat.checkSelfPermission(
        this, Manifest.permission.CAMERA
    ) == PackageManager.PERMISSION_GRANTED

    private fun startCamera() {
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            val provider = providerFuture.get()
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }
            val analysis = ImageAnalysis.Builder()
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                // Cap analysis frames to ~VGA. BlazeFace short-range works on a
                // 128x128 internal tensor, so feeding it full-sensor frames just
                // wastes bytes on the per-frame RGBA copy + rotate. Smaller frames
                // = less CPU per frame = higher FPS on the Exynos 7870. Expressed
                // in sensor-natural (landscape) orientation.
                .setTargetResolution(android.util.Size(640, 480))
                .build()
                .apply { setAnalyzer(analysisExecutor, faceAnalyzer) }

            try {
                provider.unbindAll()
                provider.bindToLifecycle(
                    this,
                    CameraSelector.DEFAULT_FRONT_CAMERA,
                    preview,
                    analysis
                )
            } catch (e: Exception) {
                Log.e(TAG, "Camera bind failed", e)
                Toast.makeText(this, "Camera bind failed: ${e.message}", Toast.LENGTH_LONG).show()
            }
        }, ContextCompat.getMainExecutor(this))
    }

    override fun onDestroy() {
        super.onDestroy()
        mainHandler.removeCallbacksAndMessages(null)
        if (::analysisExecutor.isInitialized) {
            analysisExecutor.shutdown()
            // Drain any in-flight detect() before closing the landmarker —
            // closing mid-detect crashes MediaPipe's native lib.
            try {
                analysisExecutor.awaitTermination(
                    1, java.util.concurrent.TimeUnit.SECONDS
                )
            } catch (_: InterruptedException) {}
        }
        if (::faceAnalyzer.isInitialized) faceAnalyzer.close()
        // FaceEmbedder is owned by AttendanceApp — don't close it here.
    }

    companion object {
        private const val TAG = "AttendanceSpike"
        private const val GREETING_DURATION = 4000L
    }
}
