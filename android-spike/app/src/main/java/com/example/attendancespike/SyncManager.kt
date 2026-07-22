package com.example.attendancespike

import android.util.Log
import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.URL
import java.net.URLEncoder
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit

/**
 * Background pusher for attendance events to Google Sheets (Apps Script
 * webhook) and Telegram (bot sendMessage).
 *
 * Events are queued in SQLite by [EnrollmentDb.insertEvent]; this class
 * walks the `sheets_sent` / `telegram_sent` flags and POSTs each one,
 * marking it sent on HTTP 2xx. Failures (network down, server down,
 * misconfigured URLs) leave the row unsent so the next call retries.
 *
 * Triggered from:
 *   - [AttendanceApp.onCreate] to flush leftovers from the previous process
 *   - Right after each new event insert in MainActivity
 */
class SyncManager(private val app: AttendanceApp) {

    private val executor = Executors.newSingleThreadExecutor { r ->
        Thread(r, "AttendanceSync").apply { isDaemon = true }
    }

    fun enqueueSync() {
        executor.submit {
            try { syncAllPending() }
            catch (e: Exception) { Log.w(TAG, "sync pass failed", e) }
        }
    }

    fun shutdown() {
        executor.shutdown()
        try { executor.awaitTermination(2, TimeUnit.SECONDS) } catch (_: Exception) {}
    }

    private fun syncAllPending() {
        val prefs = SettingsPrefs(app)
        val pending = app.enrollmentDb.unsentEvents()
        if (pending.isEmpty()) return
        Log.i(TAG, "Syncing ${pending.size} pending event(s)")

        var sheetsOk = 0; var sheetsErr = 0
        var tgOk = 0; var tgErr = 0
        for (e in pending) {
            if (!e.sheetsSent && prefs.isSheetsConfigured()) {
                if (postToSheets(prefs.sheetsWebhookUrl, e)) {
                    app.enrollmentDb.markSheetsSent(e.id); sheetsOk++
                } else sheetsErr++
            }
            if (!e.telegramSent && prefs.isTelegramConfigured()) {
                if (postToTelegram(prefs.telegramBotToken, prefs.telegramChatId, e)) {
                    app.enrollmentDb.markTelegramSent(e.id); tgOk++
                } else tgErr++
            }
        }
        Log.i(TAG, "Sync pass: sheets ok=$sheetsOk err=$sheetsErr  tg ok=$tgOk err=$tgErr")
    }

    // ---------- Google Sheets (Apps Script web app) ----------

    private fun postToSheets(webhookUrl: String, e: EnrollmentDb.EventRow): Boolean {
        val payload = JSONObject().apply {
            put("emp_id", e.empId)
            put("name", e.name)
            put("event", e.eventType)
            put("ts", isoFmt.format(Date(e.ts)))
            put("ts_epoch_ms", e.ts)
        }
        return postJson(webhookUrl, payload.toString())
    }

    // ---------- Telegram bot ----------

    private fun postToTelegram(
        token: String, chatId: String, e: EnrollmentDb.EventRow
    ): Boolean {
        val emoji = if (e.eventType == "IN") "🟢" else "🔴"
        val verb = if (e.eventType == "IN") "kirdi" else "chiqdi"
        val time = timeFmt.format(Date(e.ts))
        val text = "$emoji ${e.name} $verb · $time"
        val url = "https://api.telegram.org/bot$token/sendMessage"
        val body = "chat_id=${urlEnc(chatId)}&text=${urlEnc(text)}"
        return postForm(url, body)
    }

    // ---------- HTTP helpers ----------

    private fun postJson(urlStr: String, body: String): Boolean {
        return doPost(urlStr, body, "application/json; charset=utf-8")
    }

    private fun postForm(urlStr: String, body: String): Boolean {
        return doPost(urlStr, body, "application/x-www-form-urlencoded; charset=utf-8")
    }

    private fun doPost(urlStr: String, body: String, contentType: String): Boolean {
        var conn: HttpURLConnection? = null
        return try {
            conn = (URL(urlStr).openConnection() as HttpURLConnection).apply {
                requestMethod = "POST"
                doOutput = true
                connectTimeout = 20_000
                readTimeout = 30_000
                instanceFollowRedirects = true
                setRequestProperty("Content-Type", contentType)
                setRequestProperty("Accept", "application/json")
                outputStream.use { it.write(body.toByteArray(Charsets.UTF_8)) }
            }
            val code = conn.responseCode
            val ok = code in 200..299
            // Read the response body for diagnostic logging — what did the
            // Apps Script (or Telegram API) actually say?
            val respStream = if (ok) conn.inputStream else conn.errorStream
            val respBody = respStream?.bufferedReader(Charsets.UTF_8)
                ?.use { it.readText() } ?: ""
            val trimmed = if (respBody.length > 280) respBody.take(280) + "…" else respBody
            val host = URL(urlStr).host
            Log.i(TAG, "POST $host → HTTP $code  body=$trimmed")
            ok
        } catch (e: Exception) {
            Log.w(TAG, "POST $urlStr failed: ${e.message}")
            false
        } finally {
            try { conn?.disconnect() } catch (_: Exception) {}
        }
    }

    private fun urlEnc(s: String): String = URLEncoder.encode(s, "UTF-8")

    companion object {
        private const val TAG = "SyncManager"
        // SimpleDateFormat is not thread-safe; each instance is only used on
        // the single executor thread.
        private val isoFmt = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ssXXX", Locale.US)
        private val timeFmt = SimpleDateFormat("HH:mm", Locale.getDefault())
    }
}
