package com.example.attendancespike

import android.util.Log
import java.util.Collections

/**
 * Two-layer store for enrolled persons:
 *   - SQLite ([EnrollmentDb]) is the source of truth, survives process death.
 *   - In-memory [LinkedHashMap] is a match-time cache, rebuilt on startup.
 *
 * Each employee can have multiple embeddings (one per pose from the sweep);
 * matching picks the highest cosine across all of that person's embeddings.
 */
class EmbeddingStore(private val db: EnrollmentDb? = null) {

    data class Person(
        val empId: String,
        val name: String,
        val embeddings: MutableList<FloatArray> = mutableListOf()
    )

    private val enrolled: MutableMap<String, Person> =
        Collections.synchronizedMap(LinkedHashMap())

    init {
        db?.let {
            for (person in it.loadAll()) {
                enrolled[person.empId] = person
            }
            Log.i(TAG, "Hydrated ${enrolled.size} person(s) from sqlite")
        }
    }

    fun enrollMulti(
        empId: String,
        name: String,
        embeddings: List<FloatArray>,
        department: String? = null,
        email: String? = null
    ) {
        enrolled[empId] = Person(empId, name).apply { this.embeddings.addAll(embeddings) }
        db?.savePerson(empId, name, department, email, embeddings)
    }

    fun clear() {
        enrolled.clear()
        db?.deleteAll()
    }

    /** Remove a single person from both the in-memory cache and SQLite. */
    fun removePerson(empId: String) {
        enrolled.remove(empId)
        db?.deletePerson(empId)
    }

    fun size(): Int = enrolled.size

    /**
     * Returns (best_emp_id, best_name, best_score). Score range [0, 1]
     * (already-normalized embeddings → dot product = cosine).
     * Returns (null, null, 0f) if nothing is enrolled.
     */
    fun bestMatch(query: FloatArray): Triple<String?, String?, Float> {
        var bestPerson: Person? = null
        var bestScore = -1f
        synchronized(enrolled) {
            for (person in enrolled.values) {
                var personMax = -1f
                for (emb in person.embeddings) {
                    val s = dot(query, emb)
                    if (s > personMax) personMax = s
                }
                if (personMax > bestScore) {
                    bestScore = personMax
                    bestPerson = person
                }
            }
        }
        return Triple(bestPerson?.empId, bestPerson?.name, bestScore.coerceAtLeast(0f))
    }

    private fun dot(a: FloatArray, b: FloatArray): Float {
        if (a.size != b.size) return -1f
        var s = 0f
        for (i in a.indices) s += a[i] * b[i]
        return s
    }

    companion object {
        private const val TAG = "EmbeddingStore"
    }
}
