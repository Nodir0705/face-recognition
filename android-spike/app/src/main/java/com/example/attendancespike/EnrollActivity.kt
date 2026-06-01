package com.example.attendancespike

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.EditText
import android.widget.ImageView
import android.widget.ProgressBar
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
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.abs

/**
 * 4-step sweep enrollment, native port of src/web/templates/enroll.html.
 *
 * Flow:
 *   STEP 1 (FORM)    — collect emp_id + name (+ optional dept/email)
 *   STEP 2 (CAPTURE) — 4-phase sweep: FIT → LEFT → RIGHT → UP → DOWN
 *                       each phase captures one embedding
 *   STEP 3 (DONE)    — show sample count, offer "enroll another"
 */
class EnrollActivity : AppCompatActivity() {

    private enum class WizardStep { FORM, CAPTURE, DONE }
    private enum class Phase { FIT, LEFT, RIGHT, UP, DOWN, COMPLETE }

    // --- Form fields
    private lateinit var empIdInput: EditText
    private lateinit var empNameInput: EditText
    private lateinit var empDeptInput: EditText

    // --- Capture screen views
    private lateinit var stepForm: View
    private lateinit var stepCapture: View
    private lateinit var stepDone: View
    private lateinit var enrollPreviewView: PreviewView
    private lateinit var enrollFaceGuide: FaceGuideView
    private lateinit var instructionText: TextView
    private lateinit var arrowText: TextView
    private lateinit var pipLeft: TextView
    private lateinit var pipRight: TextView
    private lateinit var pipUp: TextView
    private lateinit var pipDown: TextView
    private lateinit var poseProgress: ProgressBar
    private lateinit var readoutText: TextView
    private lateinit var captureLoading: View

    // --- Done
    private lateinit var doneSummary: TextView

    // --- State
    private var wizardStep = WizardStep.FORM
    private var phase = Phase.FIT
    private var fitFramesNeeded = 0
    private var stableAtTargetFrames = 0
    // Resting head pose, measured during FIT. Sweep targets are taken relative
    // to this so every direction needs the same head movement regardless of the
    // kiosk's mounting angle / the user's posture (otherwise a non-zero neutral
    // makes e.g. "down" much harder than "up").
    private var neutralYaw = 0f
    private var neutralPitch = 0f
    private val capturedEmbeddings = mutableListOf<FloatArray>()
    // Frames gathered at the current pose; averaged into one embedding once
    // FRAMES_PER_POSE of them are collected.
    private val poseFrames = mutableListOf<FloatArray>()
    private var pendingEmpId = ""
    private var pendingName = ""
    private var pendingDept = ""

    private lateinit var analysisExecutor: ExecutorService
    private var enrollAnalyzer: EnrollFaceAnalyzer? = null
    private val analyzersToClose = mutableListOf<EnrollFaceAnalyzer>()
    private val mainHandler = Handler(Looper.getMainLooper())

    private val cameraPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) bindCameraForCapture()
        else Toast.makeText(this, "Camera permission required", Toast.LENGTH_LONG).show()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_enroll)

        stepForm = findViewById(R.id.stepForm)
        stepCapture = findViewById(R.id.stepCapture)
        stepDone = findViewById(R.id.stepDone)

        empIdInput = findViewById(R.id.empIdInput)
        empNameInput = findViewById(R.id.empNameInput)
        empDeptInput = findViewById(R.id.empDeptInput)

        enrollPreviewView = findViewById(R.id.enrollPreviewView)
        enrollPreviewView.implementationMode = PreviewView.ImplementationMode.COMPATIBLE
        enrollFaceGuide = findViewById(R.id.enrollFaceGuide)
        instructionText = findViewById(R.id.instructionText)
        arrowText = findViewById(R.id.arrowText)
        pipLeft = findViewById(R.id.pipLeft)
        pipRight = findViewById(R.id.pipRight)
        pipUp = findViewById(R.id.pipUp)
        pipDown = findViewById(R.id.pipDown)
        poseProgress = findViewById(R.id.poseProgress)
        readoutText = findViewById(R.id.readoutText)
        captureLoading = findViewById(R.id.captureLoading)

        doneSummary = findViewById(R.id.doneSummary)

        findViewById<Button>(R.id.continueButton).setOnClickListener { onContinueFromForm() }
        findViewById<Button>(R.id.formCancelButton).setOnClickListener { finish() }
        findViewById<Button>(R.id.captureCancelButton).setOnClickListener { cancelCapture() }
        findViewById<Button>(R.id.enrollAnotherButton).setOnClickListener { restartWizard() }
        findViewById<Button>(R.id.backToKioskButton).setOnClickListener { finish() }

        analysisExecutor = Executors.newSingleThreadExecutor()
    }

    // -----------------------------------------------------------------
    // STEP 1 → STEP 2
    // -----------------------------------------------------------------

    private fun onContinueFromForm() {
        val empId = empIdInput.text.toString().trim()
        val name = empNameInput.text.toString().trim()
        if (empId.isEmpty() || name.isEmpty()) {
            Toast.makeText(this, getString(R.string.validation_required), Toast.LENGTH_SHORT).show()
            return
        }
        pendingEmpId = empId
        pendingName = name
        pendingDept = empDeptInput.text.toString().trim()

        switchTo(WizardStep.CAPTURE)
        resetCaptureState()

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED) {
            bindCameraForCapture()
        } else {
            cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    // -----------------------------------------------------------------
    // STEP 2 — Capture sweep
    // -----------------------------------------------------------------

    private fun resetCaptureState() {
        phase = Phase.FIT
        fitFramesNeeded = 0
        stableAtTargetFrames = 0
        neutralYaw = 0f
        neutralPitch = 0f
        capturedEmbeddings.clear()
        poseFrames.clear()
        enrollFaceGuide.state = FaceGuideView.State.IDLE
        arrowText.alpha = 0f
        arrowText.scaleX = 1f
        arrowText.scaleY = 1f
        poseProgress.progress = 0
        readoutText.text = ""
        instructionText.text = getString(R.string.instr_fit_face)
        setPipState(pipLeft, R.drawable.pip_default, MUTED_COLOR)
        setPipState(pipRight, R.drawable.pip_default, MUTED_COLOR)
        setPipState(pipUp, R.drawable.pip_default, MUTED_COLOR)
        setPipState(pipDown, R.drawable.pip_default, MUTED_COLOR)
        showCaptureLoading()  // veil stays up until the first camera frame
    }

    private fun showCaptureLoading() {
        captureLoading.alpha = 1f
        captureLoading.visibility = View.VISIBLE
    }

    private fun hideCaptureLoading() {
        if (captureLoading.visibility != View.VISIBLE) return
        captureLoading.animate()
            .alpha(0f)
            .setDuration(250)
            .withEndAction {
                captureLoading.visibility = View.GONE
                captureLoading.alpha = 1f  // reset for the next enrollment
            }
            .start()
    }

    private fun bindCameraForCapture() {
        val app = application as AttendanceApp
        val analyzer = EnrollFaceAnalyzer(this, app.embedder).also {
            it.listener = { frame -> runOnUiThread { handleFrame(frame) } }
        }
        enrollAnalyzer = analyzer
        analyzersToClose.add(analyzer)

        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            val provider = providerFuture.get()
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(enrollPreviewView.surfaceProvider)
            }
            val analysis = ImageAnalysis.Builder()
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setTargetResolution(android.util.Size(640, 480))
                .build()
                .apply { setAnalyzer(analysisExecutor, analyzer) }
            try {
                provider.unbindAll()
                provider.bindToLifecycle(
                    this, CameraSelector.DEFAULT_FRONT_CAMERA, preview, analysis
                )
            } catch (e: Exception) {
                Log.e(TAG, "Camera bind failed", e)
                hideCaptureLoading()  // don't leave the user stuck on the veil
                Toast.makeText(this, "Camera bind failed: ${e.message}", Toast.LENGTH_LONG).show()
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun handleFrame(frame: EnrollFaceAnalyzer.Frame) {
        // Frames are flowing → CameraX is live. Drop the loading veil.
        hideCaptureLoading()

        // 1. Embedding came back for a frame we asked to capture. Collect
        //    FRAMES_PER_POSE of them at this pose, then store their averaged +
        //    re-normalized vector — denoises each pose without inflating the
        //    gallery (5 poses → 5 stored embeddings).
        if (frame.capturedEmbedding != null) {
            poseFrames.add(frame.capturedEmbedding)
            if (poseFrames.size < FRAMES_PER_POSE) {
                enrollAnalyzer?.captureNext = true  // grab the next frame at this pose
                return
            }
            capturedEmbeddings.add(averageNormalized(poseFrames))
            poseFrames.clear()
            flashCapturedArrow()
            advancePhase()
            return
        }

        // 2. No face — keep showing fit instruction, oval stays white.
        if (frame.faceBox == null) {
            enrollFaceGuide.state = FaceGuideView.State.IDLE
            readoutText.text = "yuz topilmadi"
            poseProgress.progress = 0
            stableAtTargetFrames = 0
            poseFrames.clear()  // face dropped mid-burst → restart this pose's frames
            return
        }

        // 3. Face visible — oval green; specific behavior depends on phase.
        enrollFaceGuide.state = FaceGuideView.State.FIT
        updateReadout(frame.yaw, frame.pitch)

        when (phase) {
            Phase.FIT -> { captureNeutral(frame.yaw, frame.pitch); handleFitPhase() }
            Phase.LEFT -> handleSweep(frame.yaw, neutralYaw, +YAW_THRESHOLD)
            Phase.RIGHT -> handleSweep(frame.yaw, neutralYaw, -YAW_THRESHOLD)
            Phase.UP -> handleSweep(frame.pitch, neutralPitch, +PITCH_THRESHOLD)
            Phase.DOWN -> handleSweep(frame.pitch, neutralPitch, -PITCH_THRESHOLD)
            Phase.COMPLETE -> Unit
        }
    }

    /**
     * Track the resting head pose while the user holds still in FIT. First
     * frame seeds it; later frames smooth it so a single noisy reading doesn't
     * skew the baseline the sweeps are measured against.
     */
    private fun captureNeutral(yaw: Float, pitch: Float) {
        if (fitFramesNeeded == 0) {
            neutralYaw = yaw
            neutralPitch = pitch
        } else {
            neutralYaw = neutralYaw * 0.5f + yaw * 0.5f
            neutralPitch = neutralPitch * 0.5f + pitch * 0.5f
        }
    }

    /**
     * Element-wise mean of several (already L2-normalized) embeddings, then
     * re-normalized to unit length so it stays comparable under the cosine
     * matcher. Averaging in embedding space cancels per-frame noise.
     */
    private fun averageNormalized(vectors: List<FloatArray>): FloatArray {
        val dim = vectors[0].size
        val mean = FloatArray(dim)
        for (v in vectors) {
            for (i in 0 until dim) mean[i] = mean[i] + v[i]
        }
        var sumSq = 0.0
        for (i in 0 until dim) {
            val m = mean[i] / vectors.size
            mean[i] = m
            sumSq += (m * m).toDouble()
        }
        val norm = kotlin.math.sqrt(sumSq).toFloat().coerceAtLeast(1e-10f)
        for (i in 0 until dim) mean[i] = mean[i] / norm
        return mean
    }

    private fun handleFitPhase() {
        // Stable face for FIT_FRAMES_NEEDED frames → request the frontal
        // embedding. Don't change phase here; advancePhase moves us to LEFT
        // when the embedding arrives so the LEFT instruction isn't shown
        // before we have the frontal sample.
        if (enrollAnalyzer?.captureNext == true) return  // already requesting
        fitFramesNeeded++
        poseProgress.progress = (fitFramesNeeded * 100 / FIT_FRAMES_NEEDED).coerceAtMost(100)
        if (fitFramesNeeded >= FIT_FRAMES_NEEDED) {
            enrollAnalyzer?.captureNext = true
        }
    }

    /**
     * Drive a single sweep step. The target is [neutral] + [delta] (degrees),
     * where [delta] is signed (+ = left/up, − = right/down). Progress is how far
     * the head has moved from its resting pose toward that target, so each
     * direction needs the same movement regardless of the neutral bias. When the
     * head holds past target for [STABLE_FRAMES_AT_TARGET] frames, ask the
     * analyzer to capture the embedding on the next frame.
     */
    private fun handleSweep(currentValue: Float, neutral: Float, delta: Float) {
        val progress = ((currentValue - neutral) / delta).coerceIn(0f, 1f)
        poseProgress.progress = (progress * 100f).toInt()

        if (progress >= 1f) {
            stableAtTargetFrames++
            if (stableAtTargetFrames >= STABLE_FRAMES_AT_TARGET) {
                enrollAnalyzer?.captureNext = true
                stableAtTargetFrames = 0
            }
        } else {
            stableAtTargetFrames = 0
        }
    }

    private fun advancePhase() {
        when (phase) {
            Phase.FIT -> {
                phase = Phase.LEFT
                instructionText.text = getString(R.string.instr_look_left)
                showArrow(getString(R.string.arrow_left))
                setPipState(pipLeft, R.drawable.pip_active, ACTIVE_COLOR)
            }
            Phase.LEFT -> {
                setPipState(pipLeft, R.drawable.pip_done, DONE_COLOR)
                phase = Phase.RIGHT
                instructionText.text = getString(R.string.instr_look_right)
                showArrow(getString(R.string.arrow_right))
                setPipState(pipRight, R.drawable.pip_active, ACTIVE_COLOR)
            }
            Phase.RIGHT -> {
                setPipState(pipRight, R.drawable.pip_done, DONE_COLOR)
                phase = Phase.UP
                instructionText.text = getString(R.string.instr_look_up)
                showArrow(getString(R.string.arrow_up))
                setPipState(pipUp, R.drawable.pip_active, ACTIVE_COLOR)
            }
            Phase.UP -> {
                setPipState(pipUp, R.drawable.pip_done, DONE_COLOR)
                phase = Phase.DOWN
                instructionText.text = getString(R.string.instr_look_down)
                showArrow(getString(R.string.arrow_down))
                setPipState(pipDown, R.drawable.pip_active, ACTIVE_COLOR)
            }
            Phase.DOWN -> {
                setPipState(pipDown, R.drawable.pip_done, DONE_COLOR)
                phase = Phase.COMPLETE
                finishEnrollment()
            }
            Phase.COMPLETE -> Unit
        }
        stableAtTargetFrames = 0
        poseProgress.progress = 0
    }

    private fun finishEnrollment() {
        // Save to the shared store (which persists to SQLite + caches in memory).
        val app = application as AttendanceApp
        app.store.enrollMulti(
            empId = pendingEmpId,
            name = pendingName,
            embeddings = capturedEmbeddings.toList(),
            department = pendingDept.takeIf { it.isNotBlank() }
        )
        Log.i(TAG, "Enrolled '$pendingEmpId' ($pendingName) with " +
                "${capturedEmbeddings.size} embeddings. Store size: ${app.store.size()}")
        val deptStr = if (pendingDept.isEmpty()) "—" else pendingDept
        doneSummary.text = getString(
            R.string.enroll_done_summary_fmt,
            pendingName,
            capturedEmbeddings.size,
            pendingEmpId
        ) + "\n" + deptStr

        // Release the camera before showing the done screen.
        unbindCamera()
        switchTo(WizardStep.DONE)
    }

    private fun cancelCapture() {
        unbindCamera()
        switchTo(WizardStep.FORM)
    }

    private fun unbindCamera() {
        // Just unbind from the camera lifecycle. DO NOT close the landmarker
        // here — there may still be a frame in-flight on analysisExecutor that
        // would segfault if the landmarker's native handle is freed mid-call.
        // We close everything safely in onDestroy after the executor terminates.
        ProcessCameraProvider.getInstance(this).addListener({
            try {
                ProcessCameraProvider.getInstance(this).get().unbindAll()
            } catch (e: Exception) {
                Log.w(TAG, "unbind", e)
            }
        }, ContextCompat.getMainExecutor(this))
        enrollAnalyzer = null
    }

    // -----------------------------------------------------------------
    // Wizard navigation
    // -----------------------------------------------------------------

    private fun switchTo(step: WizardStep) {
        wizardStep = step
        stepForm.visibility = if (step == WizardStep.FORM) View.VISIBLE else View.GONE
        stepCapture.visibility = if (step == WizardStep.CAPTURE) View.VISIBLE else View.GONE
        stepDone.visibility = if (step == WizardStep.DONE) View.VISIBLE else View.GONE
    }

    private fun restartWizard() {
        empIdInput.setText("")
        empNameInput.setText("")
        empDeptInput.setText("")
        switchTo(WizardStep.FORM)
    }

    // -----------------------------------------------------------------
    // UI helpers
    // -----------------------------------------------------------------

    private fun setPipState(pip: TextView, bgRes: Int, textColor: Int) {
        pip.setBackgroundResource(bgRes)
        pip.setTextColor(textColor)
    }

    private fun showArrow(symbol: String) {
        arrowText.text = symbol
        arrowText.alpha = 0f
        arrowText.scaleX = 0.7f
        arrowText.scaleY = 0.7f
        arrowText.animate()
            .alpha(0.95f)
            .scaleX(1f).scaleY(1f)
            .setDuration(220)
            .start()
    }

    private fun flashCapturedArrow() {
        arrowText.setTextColor(DONE_COLOR)
        arrowText.animate()
            .scaleX(1.35f).scaleY(1.35f)
            .setDuration(180)
            .withEndAction {
                arrowText.animate()
                    .alpha(0f)
                    .setDuration(220)
                    .withEndAction {
                        arrowText.setTextColor(0xFFFFFFFF.toInt())
                    }
                    .start()
            }
            .start()
    }

    private fun updateReadout(yaw: Float, pitch: Float) {
        val phaseStr = when (phase) {
            Phase.FIT -> "fit"; Phase.LEFT -> "left"; Phase.RIGHT -> "right"
            Phase.UP -> "up"; Phase.DOWN -> "down"; Phase.COMPLETE -> "done"
        }
        readoutText.text = String.format(
            "yaw %+5.1f°  pitch %+5.1f°  phase=%s  saved=%d/5",
            yaw, pitch, phaseStr, capturedEmbeddings.size
        )
    }

    override fun onDestroy() {
        super.onDestroy()
        mainHandler.removeCallbacksAndMessages(null)
        if (::analysisExecutor.isInitialized) {
            analysisExecutor.shutdown()
            // Wait for any in-flight detect() to finish before closing the
            // landmarker — closing it mid-detect crashes MediaPipe's native lib.
            try {
                analysisExecutor.awaitTermination(
                    1, java.util.concurrent.TimeUnit.SECONDS
                )
            } catch (_: InterruptedException) {}
        }
        for (a in analyzersToClose) {
            try { a.close() } catch (_: Exception) {}
        }
        analyzersToClose.clear()
    }

    companion object {
        private const val TAG = "EnrollActivity"

        // Pose thresholds (degrees of head movement FROM the resting pose
        // measured during FIT — see neutralYaw/neutralPitch). Because they're
        // relative to neutral, left/right and up/down need equal movement
        // regardless of the kiosk's mounting angle. Tune from real captures.
        private const val YAW_THRESHOLD = 15f
        private const val PITCH_THRESHOLD = 12f

        // Frames captured at each pose, averaged into one embedding to cut
        // per-frame noise. 3 ≈ ~0.5s of holding the pose. The gallery still
        // ends up with 5 vectors (one averaged per pose), so the matcher's
        // false-accept surface is unchanged.
        private const val FRAMES_PER_POSE = 3

        // Frames a face must be visible before FIT advances to LEFT.
        // Lowered from 4 to 2: ML Kit ACCURATE runs ~300ms/frame on Exynos 7870,
        // so 2 frames = ~600ms total FIT delay (feels snappy without being twitchy).
        private const val FIT_FRAMES_NEEDED = 2

        // Frames the pose must stay past target before capturing.
        // Lowered from 2 to 1: capture fires the instant the user reaches the
        // target angle. ML Kit's yaw/pitch are noisy enough that occasional
        // false captures could happen, but the user can always re-enroll.
        // If false captures become a real problem, bump back to 2.
        private const val STABLE_FRAMES_AT_TARGET = 1

        private const val MUTED_COLOR = 0xFF6A6A66.toInt()
        private const val ACTIVE_COLOR = 0xFF2563EB.toInt()
        private const val DONE_COLOR = 0xFF16A34A.toInt()
    }
}
