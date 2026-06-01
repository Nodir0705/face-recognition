package com.example.attendancespike

import android.app.AlertDialog
import android.content.Intent
import android.os.Bundle
import android.view.View
import android.widget.Button
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView

/**
 * Recurring admin tasks: list of enrolled employees, add new, remove existing.
 * Opens from the kiosk's gear icon. One-time setup (Sheets webhook, Telegram
 * credentials) lives in [ConfigActivity], reachable via the "Konfiguratsiya"
 * button in the header.
 */
class SettingsActivity : AppCompatActivity() {

    private lateinit var employeesHeading: TextView
    private lateinit var usersRecycler: RecyclerView
    private lateinit var usersEmpty: TextView

    private lateinit var adapter: UsersAdapter

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_settings)

        employeesHeading = findViewById(R.id.employeesHeading)
        usersRecycler = findViewById(R.id.usersRecycler)
        usersEmpty = findViewById(R.id.usersEmpty)

        adapter = UsersAdapter(emptyList()) { person -> confirmDelete(person) }
        usersRecycler.layoutManager = LinearLayoutManager(this)
        usersRecycler.adapter = adapter

        findViewById<Button>(R.id.backButton).setOnClickListener { finish() }
        findViewById<Button>(R.id.configButton).setOnClickListener {
            startActivity(Intent(this, ConfigActivity::class.java))
        }
        findViewById<Button>(R.id.addEmployeeButton).setOnClickListener {
            startActivity(Intent(this, EnrollActivity::class.java))
        }
    }

    override fun onResume() {
        super.onResume()
        refreshUsers()
    }

    private fun refreshUsers() {
        val app = application as AttendanceApp
        val users = app.enrollmentDb.listPersons()
        adapter.submit(users)
        employeesHeading.text = getString(R.string.settings_employees_fmt, users.size)
        usersRecycler.visibility = if (users.isEmpty()) View.GONE else View.VISIBLE
        usersEmpty.visibility = if (users.isEmpty()) View.VISIBLE else View.GONE
    }

    private fun confirmDelete(person: EnrollmentDb.PersonRow) {
        AlertDialog.Builder(this)
            .setTitle(person.name)
            .setMessage("${person.empId} · ${getString(R.string.settings_delete_employee)}?")
            .setPositiveButton(R.string.settings_delete_employee) { _, _ ->
                val app = application as AttendanceApp
                app.store.removePerson(person.empId)
                refreshUsers()
            }
            .setNegativeButton(R.string.cancel_btn, null)
            .show()
    }
}
