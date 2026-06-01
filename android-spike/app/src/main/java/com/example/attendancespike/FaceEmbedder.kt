package com.example.attendancespike

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.sqrt

/**
 * Wraps a MobileFaceNet TFLite model.
 *
 * Expected model: 112x112 RGB input, float32 in [-1, 1], output a 192-d embedding.
 * Drop the .tflite into app/src/main/assets/ as `mobile_face_net.tflite`.
 *
 * Backend selection (init time):
 *   1. GPU delegate if device supports OpenGL ES 3.1 (CompatibilityList)
 *   2. CPU with 2 threads otherwise
 * Backend used is logged at INFO level with tag "FaceEmbedder".
 */
class FaceEmbedder(context: Context) {

    private val interpreter: Interpreter
    private var gpuDelegate: GpuDelegate? = null
    private val inputBuf: ByteBuffer
    private val outputBuf: Array<FloatArray>
    val backend: String

    init {
        val model = loadModel(context, MODEL_FILE)
        val opts = Interpreter.Options()
        backend = tryConfigureGpu(opts) ?: configureCpu(opts)
        Log.i(TAG, "FaceEmbedder backend = $backend")

        interpreter = Interpreter(model, opts)

        inputBuf = ByteBuffer
            .allocateDirect(4 * INPUT_SIZE * INPUT_SIZE * 3)
            .order(ByteOrder.nativeOrder())
        outputBuf = Array(1) { FloatArray(EMBEDDING_DIM) }

        // Warm-up. First GPU inference compiles shaders and can take ~1s;
        // run a dummy frame here so the first real call isn't a stall.
        try {
            val warmStart = System.currentTimeMillis()
            inputBuf.rewind()
            for (i in 0 until INPUT_SIZE * INPUT_SIZE * 3) inputBuf.putFloat(0f)
            interpreter.run(inputBuf, outputBuf)
            Log.i(TAG, "Warm-up inference took ${System.currentTimeMillis() - warmStart}ms")
        } catch (e: Exception) {
            Log.w(TAG, "Warm-up failed (will still try real inference)", e)
        }
    }

    /** Returns "GPU" on success, null on failure (so caller falls back to CPU). */
    private fun tryConfigureGpu(opts: Interpreter.Options): String? {
        return try {
            val compat = CompatibilityList()
            if (!compat.isDelegateSupportedOnThisDevice) {
                Log.i(TAG, "GPU delegate not supported on this device (no OpenGL ES 3.1?)")
                return null
            }
            val delegate = GpuDelegate(compat.bestOptionsForThisDevice)
            opts.addDelegate(delegate)
            gpuDelegate = delegate
            "GPU"
        } catch (e: Throwable) {
            // GPU delegate native lib may fail to load on some devices; bail to CPU.
            Log.w(TAG, "GPU delegate init failed, falling back to CPU", e)
            null
        }
    }

    private fun configureCpu(opts: Interpreter.Options): String {
        opts.setNumThreads(2)
        return "CPU (2 threads)"
    }

    fun embed(face: Bitmap): FloatArray {
        val resized = if (face.width == INPUT_SIZE && face.height == INPUT_SIZE) face
        else Bitmap.createScaledBitmap(face, INPUT_SIZE, INPUT_SIZE, true)

        inputBuf.rewind()
        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        resized.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)
        for (px in pixels) {
            val r = ((px shr 16) and 0xFF) / 127.5f - 1f
            val g = ((px shr 8) and 0xFF) / 127.5f - 1f
            val b = (px and 0xFF) / 127.5f - 1f
            inputBuf.putFloat(r)
            inputBuf.putFloat(g)
            inputBuf.putFloat(b)
        }
        if (resized !== face) resized.recycle()

        interpreter.run(inputBuf, outputBuf)
        return l2Normalize(outputBuf[0])
    }

    fun close() {
        interpreter.close()
        gpuDelegate?.close()
        gpuDelegate = null
    }

    private fun l2Normalize(v: FloatArray): FloatArray {
        var sumSq = 0.0
        for (x in v) sumSq += (x * x).toDouble()
        val norm = sqrt(sumSq).toFloat().coerceAtLeast(1e-10f)
        return FloatArray(v.size) { v[it] / norm }
    }

    private fun loadModel(context: Context, assetName: String): MappedByteBuffer {
        val afd = context.assets.openFd(assetName)
        return FileInputStream(afd.fileDescriptor).channel.map(
            FileChannel.MapMode.READ_ONLY,
            afd.startOffset,
            afd.declaredLength
        )
    }

    companion object {
        private const val TAG = "FaceEmbedder"
        private const val MODEL_FILE = "mobile_face_net.tflite"

        /**
         * One-shot startup micro-benchmark: time the SAME float32 model on
         * GPU vs CPU(2thr) vs CPU(4thr)+XNNPACK so we can decide empirically
         * whether the ~110 ms embed time is delegate overhead. Logs one line
         * per config at INFO. Builds + tears down its own interpreters; does
         * NOT touch the live embedder. Call OFF the main thread.
         */
        fun benchmark(context: Context, iters: Int = 20) {
            fun loadBuf(): MappedByteBuffer {
                val afd = context.assets.openFd(MODEL_FILE)
                return FileInputStream(afd.fileDescriptor).channel.map(
                    FileChannel.MapMode.READ_ONLY, afd.startOffset, afd.declaredLength
                )
            }
            val input = ByteBuffer
                .allocateDirect(4 * INPUT_SIZE * INPUT_SIZE * 3)
                .order(ByteOrder.nativeOrder())
            val output = Array(1) { FloatArray(EMBEDDING_DIM) }

            fun time(label: String, configure: (Interpreter.Options) -> GpuDelegate?) {
                var delegate: GpuDelegate? = null
                try {
                    val opts = Interpreter.Options()
                    delegate = configure(opts)
                    val interp = Interpreter(loadBuf(), opts)
                    // warm-up (shader compile / allocs) — excluded from timing
                    input.rewind(); repeat(INPUT_SIZE * INPUT_SIZE * 3) { input.putFloat(0f) }
                    interp.run(input, output)
                    var sum = 0L
                    for (i in 0 until iters) {
                        val t = System.nanoTime()
                        interp.run(input, output)
                        sum += System.nanoTime() - t
                    }
                    Log.i(TAG, "EMBED-BENCH %-14s avg=%.1fms (%d iters)"
                        .format(label, sum / 1e6 / iters, iters))
                    interp.close()
                } catch (e: Throwable) {
                    Log.w(TAG, "EMBED-BENCH $label failed: ${e.message}")
                } finally {
                    delegate?.close()
                }
            }

            time("GPU") { opts ->
                val compat = CompatibilityList()
                if (!compat.isDelegateSupportedOnThisDevice) { return@time null }
                val d = GpuDelegate(compat.bestOptionsForThisDevice)
                opts.addDelegate(d); d
            }
            time("CPU-2thr") { opts -> opts.setNumThreads(2); null }
            time("CPU-4thr-XNN") { opts ->
                opts.setNumThreads(4)
                try { opts.setUseXNNPACK(true) } catch (_: Throwable) {}
                null
            }
        }
        private const val INPUT_SIZE = 112
        private const val EMBEDDING_DIM = 192
    }
}
