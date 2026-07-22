package com.example.attendancespike

import android.content.Context
import android.content.SharedPreferences

/**
 * Thin wrapper around SharedPreferences for kiosk admin settings:
 * the Google Sheets Apps Script webhook URL and the Telegram bot
 * credentials used for IN/OUT notifications.
 */
class SettingsPrefs(context: Context) {

    private val prefs: SharedPreferences =
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    var sheetsWebhookUrl: String
        get() = prefs.getString(KEY_SHEETS_URL, "") ?: ""
        set(value) = prefs.edit().putString(KEY_SHEETS_URL, value.trim()).apply()

    var telegramBotToken: String
        get() = prefs.getString(KEY_TG_TOKEN, "") ?: ""
        set(value) = prefs.edit().putString(KEY_TG_TOKEN, value.trim()).apply()

    var telegramChatId: String
        get() = prefs.getString(KEY_TG_CHAT, "") ?: ""
        set(value) = prefs.edit().putString(KEY_TG_CHAT, value.trim()).apply()

    /** How long after an event to suppress re-firing for the same emp_id. */
    val cooldownSeconds: Int
        get() = prefs.getInt(KEY_COOLDOWN, 60)

    fun isSheetsConfigured(): Boolean = sheetsWebhookUrl.startsWith("https://")
    fun isTelegramConfigured(): Boolean =
        telegramBotToken.isNotBlank() && telegramChatId.isNotBlank()

    companion object {
        private const val PREFS_NAME = "attendance_settings"
        private const val KEY_SHEETS_URL = "sheets_webhook_url"
        private const val KEY_TG_TOKEN = "telegram_bot_token"
        private const val KEY_TG_CHAT = "telegram_chat_id"
        private const val KEY_COOLDOWN = "cooldown_seconds"
    }
}
