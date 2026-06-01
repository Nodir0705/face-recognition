package com.example.attendancespike

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView

/**
 * RecyclerView adapter for the enrolled-employees list in SettingsActivity.
 */
class UsersAdapter(
    private var items: List<EnrollmentDb.PersonRow>,
    private val onDelete: (EnrollmentDb.PersonRow) -> Unit
) : RecyclerView.Adapter<UsersAdapter.VH>() {

    class VH(view: View) : RecyclerView.ViewHolder(view) {
        val name: TextView = view.findViewById(R.id.userName)
        val meta: TextView = view.findViewById(R.id.userMeta)
        val deleteButton: Button = view.findViewById(R.id.deleteButton)
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): VH {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_user, parent, false)
        return VH(view)
    }

    override fun onBindViewHolder(holder: VH, position: Int) {
        val p = items[position]
        holder.name.text = p.name
        holder.meta.text = if (p.department.isNullOrBlank()) p.empId else "${p.empId} · ${p.department}"
        holder.deleteButton.setOnClickListener { onDelete(p) }
    }

    override fun getItemCount(): Int = items.size

    fun submit(newItems: List<EnrollmentDb.PersonRow>) {
        items = newItems
        notifyDataSetChanged()
    }
}
