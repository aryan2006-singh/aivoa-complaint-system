import { useState } from "react";

import { useUpdateComplaintMutation, type Complaint } from "../../services/api";

export default function ComplaintForm({ complaint }: { complaint: Complaint }) {
  const [fields, setFields] = useState({
    product_name: complaint.product_name,
    batch_lot_number: complaint.batch_lot_number,
    customer_name: complaint.customer_name ?? "",
    description: complaint.description,
    status: complaint.status,
  });
  const [updateComplaint, { isLoading }] = useUpdateComplaintMutation();

  const set = (key: keyof typeof fields) => (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>) =>
    setFields((f) => ({ ...f, [key]: e.target.value }));

  return (
    <div className="space-y-3">
      <h2 className="font-semibold">Log Customer Complaint</h2>
      <label className="block text-sm">
        Product Name
        <input className="w-full border rounded p-2" value={fields.product_name} onChange={set("product_name")} />
      </label>
      <label className="block text-sm">
        Batch/Lot Number
        <input className="w-full border rounded p-2" value={fields.batch_lot_number} onChange={set("batch_lot_number")} />
      </label>
      <label className="block text-sm">
        Customer Name
        <input className="w-full border rounded p-2" value={fields.customer_name} onChange={set("customer_name")} />
      </label>
      <label className="block text-sm">
        Description
        <textarea className="w-full border rounded p-2 h-24" value={fields.description} onChange={set("description")} />
      </label>
      <label className="block text-sm">
        Status
        <select className="w-full border rounded p-2" value={fields.status} onChange={set("status")}>
          {["New", "Under Review", "CAPA Initiated", "Closed"].map((s) => (
            <option key={s} value={s}>
              {s}
            </option>
          ))}
        </select>
      </label>
      <button
        className="rounded bg-slate-900 text-white px-4 py-2 disabled:opacity-50"
        disabled={isLoading}
        onClick={() => updateComplaint({ id: complaint.id, patch: { ...fields, changed_by: "reviewer" } })}
      >
        Save Complaint
      </button>
    </div>
  );
}
