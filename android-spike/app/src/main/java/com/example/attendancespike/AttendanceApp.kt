package com.example.attendancespike

import android.app.Application
import com.google.mediapipe.tasks.vision.facedetector.FaceDetector

/**
 * Process-scoped singletons: TFLite embedder, SQLite store, sync manager.
 * Shared between MainActivity, EnrollActivity, SettingsActivity, ConfigActivity.
 */
class AttendanceApp : Application() {

    lateinit var embedder: FaceEmbedder
        private set
    lateinit var enrollmentDb: EnrollmentDb
        private set
    lateinit var store: EmbeddingStore
        private set
    lateinit var syncManager: SyncManager
        private set

    // App-scoped BlazeFace FaceDetector: built + warmed ONCE and reused across
    // MainActivity recreations so returning from enrollment never pays a cold
    // GPU-graph init + shader compile. Built lazily off the main thread.
    // Shared by the kiosk AND the enrollment crop path so crops stay consistent.
    @Volatile private var _faceDetector: FaceDetector? = null
    @Volatile var kioskBackend: String = "?"
        private set

    /**
     * Get (or build) the shared face detector. Blocks the first caller while the
     * GPU graph initialises + warms up, so call this OFF the main thread.
     */
    fun faceDetector(): FaceDetector {
        _faceDetector?.let { return it }
        return synchronized(this) {
            _faceDetector ?: run {
                val (det, backend) = FaceAnalyzer.buildDetector(this)
                FaceAnalyzer.warmUp(det)
                kioskBackend = backend
                _faceDetector = det
                det
            }
        }
    }

    override fun onCreate() {
        super.onCreate()
        embedder = FaceEmbedder(this)
        enrollmentDb = EnrollmentDb(this)
        store = EmbeddingStore(enrollmentDb)
        syncManager = SyncManager(this)

        // Pre-build + warm the face detector in the background so it's ready by
        // the time MainActivity binds the camera (and stays ready forever).
        Thread({ faceDetector() }, "face-detector-init").start()

        // End-of-day closeout: emit OUT events for anyone with an open IN on
        // a day before today. Runs whenever the process starts — catches the
        // case where the kiosk was off at midnight.
        val today = java.text.SimpleDateFormat(
            "yyyy-MM-dd", java.util.Locale.getDefault()
        ).format(java.util.Date())
        val open = enrollmentDb.openPresenceBefore(today)
        for (row in open) {
            enrollmentDb.insertEvent(row.empId, row.name, "OUT", row.lastSeenTs)
            enrollmentDb.markPresenceClosedOut(row.empId, row.date)
        }
        if (open.isNotEmpty()) {
            android.util.Log.i("AttendanceApp",
                "Closed out ${open.size} stale presence row(s) on startup")
        }

        // Flush events the last session couldn't deliver (network/config issues).
        syncManager.enqueueSync()
    }

    override fun onTerminate() {
        super.onTerminate()
        if (::syncManager.isInitialized) syncManager.shutdown()
        if (::embedder.isInitialized) embedder.close()
        if (::enrollmentDb.isInitialized) enrollmentDb.close()
        _faceDetector?.let { try { it.close() } catch (_: Exception) {} }
        _faceDetector = null
    }
}
