import { useGetComplaintHistoryQuery } from "../../services/api";

export default function AuditTrailTimeline({ complaintId }: { complaintId: string }) {
  const { data: history } = useGetComplaintHistoryQuery(complaintId);

  if (!history || history.length === 0) return null;

  return (
    <div className="rounded border p-4">
      <h2 className="font-semibold mb-2">Audit Trail</h2>
      <ul className="space-y-1 text-xs text-slate-600">
        {history.map((h, i) => (
          <li key={i}>
            {new Date(h.changed_at).toLocaleString()} — {h.changed_by} changed {h.field} from "{h.old_value}" to "{h.new_value}"
          </li>
        ))}
      </ul>
    </div>
  );
}
