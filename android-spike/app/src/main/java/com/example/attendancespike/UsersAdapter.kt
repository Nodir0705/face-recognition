package com.example.attendancespike

import android.graphics.Color
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView

/**
 * RecyclerView adapter for the enrolled-employees TABLE in SettingsActivity.
 * Binds one row per person across the columns #, Ism, ID, Bo'lim, Amal, with
 * zebra striping on odd rows. Column widths live in item_user.xml and must
 * match the static header in activity_settings.xml.
 */
class UsersAdapter(
    private var items: List<EnrollmentDb.PersonRow>,
    private val onDelete: (EnrollmentDb.PersonRow) -> Unit
) : RecyclerView.Adapter<UsersAdapter.VH>() {

    class VH(view: View) : RecyclerView.ViewHolder(view) {
        val row: View = view.findViewById(R.id.userRow)
        val index: TextView = view.findViewById(R.id.userIndex)
        val name: TextView = view.findViewById(R.id.userName)
        val empId: TextView = view.findViewById(R.id.userEmpId)
        val dept: TextView = view.findViewById(R.id.userDept)
        val deleteButton: Button = view.findViewById(R.id.deleteButton)
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): VH {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_user, parent, false)
        return VH(view)
    }

    override fun onBindViewHolder(holder: VH, position: Int) {
        val p = items[position]
        holder.index.text = (position + 1).toString()
        holder.name.text = p.name
        holder.empId.text = p.empId
        holder.dept.text =
            if (p.department.isNullOrBlank())
                holder.itemView.context.getString(R.string.table_em_dash)
            else p.department
        // Zebra: assign on EVERY bind (both branches) so recycled rows never
        // keep a stale tint.
        if (position % 2 == 1) {
            holder.row.setBackgroundResource(R.drawable.row_zebra_bg)
        } else {
            holder.row.setBackgroundColor(Color.TRANSPARENT)
        }
        holder.deleteButton.setOnClickListener { onDelete(p) }
    }

    override fun getItemCount(): Int = items.size

    fun submit(newItems: List<EnrollmentDb.PersonRow>) {
        items = newItems
        notifyDataSetChanged()
    }
}
