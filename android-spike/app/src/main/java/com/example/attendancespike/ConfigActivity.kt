package com.example.attendancespike

import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity

/**
 * One-time configuration screen — Google Sheets webhook URL and Telegram
 * bot credentials. Reached from the Konfiguratsiya button on SettingsActivity.
 * The kiosk only needs this set once at install; nothing here changes
 * during day-to-day employee management.
 */
class ConfigActivity : AppCompatActivity() {

    private lateinit var prefs: SettingsPrefs
    private lateinit var sheetsUrlInput: EditText
    private lateinit var telegramTokenInput: EditText
    private lateinit var telegramChatInput: EditText

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_config)
        prefs = SettingsPrefs(this)

        sheetsUrlInput = findViewById(R.id.sheetsUrlInput)
        telegramTokenInput = findViewById(R.id.telegramTokenInput)
        telegramChatInput = findViewById(R.id.telegramChatInput)

        sheetsUrlInput.setText(prefs.sheetsWebhookUrl)
        telegramTokenInput.setText(prefs.telegramBotToken)
        telegramChatInput.setText(prefs.telegramChatId)

        findViewById<Button>(R.id.backButton).setOnClickListener { finish() }
        findViewById<Button>(R.id.saveButton).setOnClickListener { save() }
    }

    private fun save() {
        prefs.sheetsWebhookUrl = sheetsUrlInput.text.toString()
        prefs.telegramBotToken = telegramTokenInput.text.toString()
        prefs.telegramChatId = telegramChatInput.text.toString()
        Toast.makeText(this, R.string.settings_saved, Toast.LENGTH_SHORT).show()
        finish()
    }
}
