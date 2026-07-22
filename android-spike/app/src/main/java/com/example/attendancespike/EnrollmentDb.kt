package com.example.attendancespike

import android.content.ContentValues
import android.content.Context
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper
import android.util.Log
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * SQLite-backed storage for enrolled persons + their multi-pose embeddings.
 * No external dependencies — uses Android's built-in SQLiteOpenHelper.
 *
 * Schema:
 *   persons     (emp_id PK, name, department, email, created_at)
 *   embeddings  (id PK, emp_id FK, idx, vec BLOB)
 */
class EnrollmentDb(context: Context) :
    SQLiteOpenHelper(context, DB_NAME, null, DB_VERSION) {

    override fun onCreate(db: SQLiteDatabase) {
        db.execSQL("""
            CREATE TABLE persons (
                emp_id      TEXT    PRIMARY KEY,
                name        TEXT    NOT NULL,
                department  TEXT,
                email       TEXT,
                created_at  INTEGER NOT NULL
            )
        """.trimIndent())
        db.execSQL("""
            CREATE TABLE embeddings (
                id      INTEGER PRIMARY KEY AUTOINCREMENT,
                emp_id  TEXT    NOT NULL,
                idx     INTEGER NOT NULL,
                vec     BLOB    NOT NULL,
                FOREIGN KEY (emp_id) REFERENCES persons(emp_id) ON DELETE CASCADE
            )
        """.trimIndent())
        db.execSQL("CREATE INDEX idx_embeddings_emp_id ON embeddings(emp_id)")
        // Attendance events — every IN/OUT that fires goes here, then syncs
        // to Google Sheets + Telegram (sync_state tracks progress).
        db.execSQL("""
            CREATE TABLE events (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                emp_id      TEXT    NOT NULL,
                name        TEXT    NOT NULL,
                event_type  TEXT    NOT NULL,  -- 'IN' or 'OUT'
                ts          INTEGER NOT NULL,
                sheets_sent INTEGER NOT NULL DEFAULT 0,
                telegram_sent INTEGER NOT NULL DEFAULT 0
            )
        """.trimIndent())
        db.execSQL("CREATE INDEX idx_events_emp_ts ON events(emp_id, ts)")
        db.execSQL("CREATE INDEX idx_events_unsynced ON events(sheets_sent, telegram_sent)")
        // First-IN / last-OUT-per-day tracker. One row per (employee, local date).
        // Created on first sighting, last_seen_ts is updated on every subsequent
        // sighting that same day. At end-of-day (or first detection of a new day)
        // we emit the OUT event for any still-open row.
        db.execSQL("""
            CREATE TABLE daily_presence (
                emp_id        TEXT    NOT NULL,
                name          TEXT    NOT NULL,
                local_date    TEXT    NOT NULL,
                first_seen_ts INTEGER NOT NULL,
                last_seen_ts  INTEGER NOT NULL,
                closed_out    INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (emp_id, local_date)
            )
        """.trimIndent())
        db.execSQL("CREATE INDEX idx_presence_open ON daily_presence(closed_out, local_date)")
        db.execSQL("PRAGMA foreign_keys = ON")
    }

    override fun onUpgrade(db: SQLiteDatabase, oldVersion: Int, newVersion: Int) {
        Log.i(TAG, "Migrating db $oldVersion → $newVersion")
        // v1 → v2: add events table for attendance logging.
        if (oldVersion < 2) {
            db.execSQL("""
                CREATE TABLE IF NOT EXISTS events (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    emp_id      TEXT    NOT NULL,
                    name        TEXT    NOT NULL,
                    event_type  TEXT    NOT NULL,
                    ts          INTEGER NOT NULL,
                    sheets_sent INTEGER NOT NULL DEFAULT 0,
                    telegram_sent INTEGER NOT NULL DEFAULT 0
                )
            """.trimIndent())
            db.execSQL("CREATE INDEX IF NOT EXISTS idx_events_emp_ts ON events(emp_id, ts)")
            db.execSQL("CREATE INDEX IF NOT EXISTS idx_events_unsynced ON events(sheets_sent, telegram_sent)")
        }
        // v2 → v3: add daily_presence for first-IN/last-OUT-per-day tracking.
        if (oldVersion < 3) {
            db.execSQL("""
                CREATE TABLE IF NOT EXISTS daily_presence (
                    emp_id        TEXT    NOT NULL,
                    name          TEXT    NOT NULL,
                    local_date    TEXT    NOT NULL,
                    first_seen_ts INTEGER NOT NULL,
                    last_seen_ts  INTEGER NOT NULL,
                    closed_out    INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (emp_id, local_date)
                )
            """.trimIndent())
            db.execSQL("CREATE INDEX IF NOT EXISTS idx_presence_open ON daily_presence(closed_out, local_date)")
        }
    }

    override fun onOpen(db: SQLiteDatabase) {
        super.onOpen(db)
        db.execSQL("PRAGMA foreign_keys = ON")
    }

    fun savePerson(
        empId: String,
        name: String,
        department: String?,
        email: String?,
        embeddings: List<FloatArray>
    ) {
        val db = writableDatabase
        db.beginTransaction()
        try {
            // Upsert person — same emp_id replaces previous enrollment.
            val personValues = ContentValues().apply {
                put("emp_id", empId)
                put("name", name)
                put("department", department)
                put("email", email)
                put("created_at", System.currentTimeMillis())
            }
            db.insertWithOnConflict(
                "persons", null, personValues, SQLiteDatabase.CONFLICT_REPLACE
            )

            // Wipe any old embeddings for this emp_id, then insert new ones.
            db.delete("embeddings", "emp_id = ?", arrayOf(empId))
            for ((idx, vec) in embeddings.withIndex()) {
                val embValues = ContentValues().apply {
                    put("emp_id", empId)
                    put("idx", idx)
                    put("vec", floatArrayToBytes(vec))
                }
                db.insert("embeddings", null, embValues)
            }
            db.setTransactionSuccessful()
            Log.i(TAG, "Saved $empId ($name) with ${embeddings.size} embeddings")
        } finally {
            db.endTransaction()
        }
    }

    fun loadAll(): List<EmbeddingStore.Person> {
        val out = mutableListOf<EmbeddingStore.Person>()
        val db = readableDatabase
        db.rawQuery(
            "SELECT emp_id, name FROM persons ORDER BY created_at", null
        ).use { c ->
            while (c.moveToNext()) {
                val empId = c.getString(0)
                val name = c.getString(1)
                val embeddings = loadEmbeddingsFor(db, empId)
                if (embeddings.isNotEmpty()) {
                    out.add(EmbeddingStore.Person(empId, name, embeddings.toMutableList()))
                } else {
                    Log.w(TAG, "Skipping $empId — has no embeddings")
                }
            }
        }
        Log.i(TAG, "Loaded ${out.size} person(s) from db")
        return out
    }

    private fun loadEmbeddingsFor(db: SQLiteDatabase, empId: String): List<FloatArray> {
        val list = mutableListOf<FloatArray>()
        db.rawQuery(
            "SELECT vec FROM embeddings WHERE emp_id = ? ORDER BY idx",
            arrayOf(empId)
        ).use { c ->
            while (c.moveToNext()) {
                list.add(bytesToFloatArray(c.getBlob(0)))
            }
        }
        return list
    }

    fun deleteAll() {
        val db = writableDatabase
        db.beginTransaction()
        try {
            db.delete("embeddings", null, null)
            db.delete("persons", null, null)
            db.setTransactionSuccessful()
            Log.i(TAG, "Wiped all enrolled persons")
        } finally {
            db.endTransaction()
        }
    }

    fun count(): Int {
        val db = readableDatabase
        db.rawQuery("SELECT COUNT(*) FROM persons", null).use { c ->
            return if (c.moveToFirst()) c.getInt(0) else 0
        }
    }

    private fun floatArrayToBytes(vec: FloatArray): ByteArray {
        val bb = ByteBuffer.allocate(vec.size * 4).order(ByteOrder.LITTLE_ENDIAN)
        for (f in vec) bb.putFloat(f)
        return bb.array()
    }

    private fun bytesToFloatArray(bytes: ByteArray): FloatArray {
        val bb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        val n = bytes.size / 4
        val out = FloatArray(n)
        for (i in 0 until n) out[i] = bb.float
        return out
    }

    // -----------------------------------------------------------------
    // Person helpers (used by Settings UI)
    // -----------------------------------------------------------------

    data class PersonRow(val empId: String, val name: String, val department: String?)

    fun listPersons(): List<PersonRow> {
        val out = mutableListOf<PersonRow>()
        readableDatabase.rawQuery(
            "SELECT emp_id, name, department FROM persons ORDER BY created_at DESC",
            null
        ).use { c ->
            while (c.moveToNext()) {
                out.add(PersonRow(c.getString(0), c.getString(1), c.getString(2)))
            }
        }
        return out
    }

    fun deletePerson(empId: String) {
        val db = writableDatabase
        db.beginTransaction()
        try {
            // Embeddings cascade via FK; events stay for historical record.
            val rows = db.delete("persons", "emp_id = ?", arrayOf(empId))
            db.setTransactionSuccessful()
            Log.i(TAG, "Deleted person '$empId' ($rows rows)")
        } finally {
            db.endTransaction()
        }
    }

    // -----------------------------------------------------------------
    // Event helpers (recognition log)
    // -----------------------------------------------------------------

    /** Insert an attendance event. Returns the auto-generated row id. */
    fun insertEvent(empId: String, name: String, eventType: String, ts: Long): Long {
        val db = writableDatabase
        val v = ContentValues().apply {
            put("emp_id", empId)
            put("name", name)
            put("event_type", eventType)
            put("ts", ts)
        }
        return db.insert("events", null, v)
    }

    /** Returns ("IN" | "OUT", lastTs) for [empId], or null if no prior event. */
    fun lastEventFor(empId: String): Pair<String, Long>? {
        readableDatabase.rawQuery(
            "SELECT event_type, ts FROM events WHERE emp_id = ? ORDER BY ts DESC LIMIT 1",
            arrayOf(empId)
        ).use { c ->
            return if (c.moveToFirst()) c.getString(0) to c.getLong(1) else null
        }
    }

    data class EventRow(
        val id: Long,
        val empId: String,
        val name: String,
        val eventType: String,
        val ts: Long,
        val sheetsSent: Boolean,
        val telegramSent: Boolean
    )

    /** Events that still need to be pushed to Sheets and/or Telegram. */
    fun unsentEvents(limit: Int = 100): List<EventRow> {
        val out = mutableListOf<EventRow>()
        readableDatabase.rawQuery(
            "SELECT id, emp_id, name, event_type, ts, sheets_sent, telegram_sent " +
                    "FROM events WHERE sheets_sent = 0 OR telegram_sent = 0 " +
                    "ORDER BY ts ASC LIMIT ?",
            arrayOf(limit.toString())
        ).use { c ->
            while (c.moveToNext()) {
                out.add(
                    EventRow(
                        id = c.getLong(0),
                        empId = c.getString(1),
                        name = c.getString(2),
                        eventType = c.getString(3),
                        ts = c.getLong(4),
                        sheetsSent = c.getInt(5) == 1,
                        telegramSent = c.getInt(6) == 1
                    )
                )
            }
        }
        return out
    }

    fun markSheetsSent(id: Long) {
        writableDatabase.execSQL(
            "UPDATE events SET sheets_sent = 1 WHERE id = ?", arrayOf(id)
        )
    }

    fun markTelegramSent(id: Long) {
        writableDatabase.execSQL(
            "UPDATE events SET telegram_sent = 1 WHERE id = ?", arrayOf(id)
        )
    }

    // -----------------------------------------------------------------
    // Daily presence (first-IN, last-OUT logic)
    // -----------------------------------------------------------------

    data class DailyPresence(
        val empId: String,
        val name: String,
        val date: String,        // 'YYYY-MM-DD' local
        val firstSeenTs: Long,
        val lastSeenTs: Long,
        val closedOut: Boolean
    )

    /**
     * Mark this person seen at [ts] on [date].
     *   - First sighting on [date] → inserts a new row → returns true.
     *   - Already seen today → bumps last_seen_ts only → returns false.
     */
    fun touchDailyPresence(empId: String, name: String, date: String, ts: Long): Boolean {
        val db = writableDatabase
        db.beginTransaction()
        try {
            val existing = getDailyPresence(empId, date)
            val isFirst: Boolean
            if (existing == null) {
                db.insert("daily_presence", null, ContentValues().apply {
                    put("emp_id", empId)
                    put("name", name)
                    put("local_date", date)
                    put("first_seen_ts", ts)
                    put("last_seen_ts", ts)
                    put("closed_out", 0)
                })
                isFirst = true
            } else {
                db.execSQL(
                    "UPDATE daily_presence SET last_seen_ts = ? WHERE emp_id = ? AND local_date = ?",
                    arrayOf(ts, empId, date)
                )
                isFirst = false
            }
            db.setTransactionSuccessful()
            return isFirst
        } finally {
            db.endTransaction()
        }
    }

    fun getDailyPresence(empId: String, date: String): DailyPresence? {
        readableDatabase.rawQuery(
            "SELECT emp_id, name, local_date, first_seen_ts, last_seen_ts, closed_out " +
                    "FROM daily_presence WHERE emp_id = ? AND local_date = ?",
            arrayOf(empId, date)
        ).use { c ->
            return if (c.moveToFirst()) DailyPresence(
                empId = c.getString(0), name = c.getString(1), date = c.getString(2),
                firstSeenTs = c.getLong(3), lastSeenTs = c.getLong(4),
                closedOut = c.getInt(5) == 1
            ) else null
        }
    }

    /** Open rows from days strictly before [date], for end-of-day closeout. */
    fun openPresenceBefore(date: String): List<DailyPresence> {
        val out = mutableListOf<DailyPresence>()
        readableDatabase.rawQuery(
            "SELECT emp_id, name, local_date, first_seen_ts, last_seen_ts, closed_out " +
                    "FROM daily_presence WHERE closed_out = 0 AND local_date < ? " +
                    "ORDER BY local_date ASC, first_seen_ts ASC",
            arrayOf(date)
        ).use { c ->
            while (c.moveToNext()) {
                out.add(DailyPresence(
                    empId = c.getString(0), name = c.getString(1), date = c.getString(2),
                    firstSeenTs = c.getLong(3), lastSeenTs = c.getLong(4),
                    closedOut = c.getInt(5) == 1
                ))
            }
        }
        return out
    }

    fun markPresenceClosedOut(empId: String, date: String) {
        writableDatabase.execSQL(
            "UPDATE daily_presence SET closed_out = 1 WHERE emp_id = ? AND local_date = ?",
            arrayOf(empId, date)
        )
    }

    companion object {
        private const val TAG = "EnrollmentDb"
        private const val DB_NAME = "attendance.db"
        // v3: added daily_presence for first-IN / last-OUT per day.
        private const val DB_VERSION = 3
    }
}
