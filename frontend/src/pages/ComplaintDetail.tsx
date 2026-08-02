import { useParams } from "react-router-dom";

import AuditTrailTimeline from "../features/complaints/AuditTrailTimeline";
import ComplaintForm from "../features/complaints/ComplaintForm";
import RiskAssessmentPanel from "../features/complaints/RiskAssessmentPanel";
import { useGetComplaintQuery } from "../services/api";

export default function ComplaintDetail() {
  const { id } = useParams<{ id: string }>();
  const { data: complaint, isLoading } = useGetComplaintQuery(id!);

  if (isLoading || !complaint) return <p className="p-6">Loading...</p>;

  return (
    <div className="p-6 grid grid-cols-2 gap-6">
      <ComplaintForm complaint={complaint} />
      <div className="space-y-4">
        <RiskAssessmentPanel assessment={complaint.assessment} />
        <AuditTrailTimeline complaintId={complaint.id} />
      </div>
    </div>
  );
}
