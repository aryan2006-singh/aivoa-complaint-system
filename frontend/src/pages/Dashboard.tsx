import { Link } from "react-router-dom";

import { useGetComplaintsQuery } from "../services/api";

const severityColor: Record<string, string> = {
  Critical: "bg-red-100 text-red-800",
  Major: "bg-amber-100 text-amber-800",
  Minor: "bg-emerald-100 text-emerald-800",
};

export default function Dashboard() {
  const { data: complaints, isLoading } = useGetComplaintsQuery({});

  if (isLoading) return <p className="p-6">Loading complaints...</p>;

  return (
    <div className="p-6">
      <div className="flex items-center justify-between mb-4">
        <h1 className="text-2xl font-semibold">Customer Complaints</h1>
        <Link to="/intake" className="rounded bg-slate-900 text-white px-4 py-2">
          + Log Complaint
        </Link>
      </div>
      <table className="w-full text-left border-collapse">
        <thead>
          <tr className="border-b text-sm text-slate-500">
            <th className="py-2">Complaint #</th>
            <th>Product</th>
            <th>Batch/Lot</th>
            <th>Severity</th>
            <th>Status</th>
            <th>Date</th>
          </tr>
        </thead>
        <tbody>
          {(complaints ?? []).map((c) => (
            <tr key={c.id} className="border-b hover:bg-slate-50">
              <td className="py-2">
                <Link to={`/complaints/${c.id}`} className="text-blue-600">
                  {c.complaint_number}
                </Link>
              </td>
              <td>{c.product_name}</td>
              <td>{c.batch_lot_number}</td>
              <td>
                <span className={`rounded px-2 py-1 text-xs ${severityColor[c.severity ?? ""] ?? "bg-slate-100 text-slate-700"}`}>
                  {c.severity ?? "Unclassified"}
                </span>
              </td>
              <td>{c.status}</td>
              <td>{new Date(c.date_received).toLocaleDateString()}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
