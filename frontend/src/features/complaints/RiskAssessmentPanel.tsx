import { Link } from "react-router-dom";

import type { AIAssessment } from "../../services/api";

export default function RiskAssessmentPanel({ assessment }: { assessment: AIAssessment | null }) {
  if (!assessment) return <p className="text-sm text-slate-500">No AI assessment available.</p>;

  return (
    <div className="rounded border p-4 space-y-3">
      <h2 className="font-semibold">AI Copilot Risk Assessment</h2>

      <div>
        <p className="text-sm font-medium">Completeness: {Math.round(assessment.completeness_score * 100)}%</p>
        {assessment.missing_fields.length > 0 && (
          <p className="text-xs text-amber-700">Missing: {assessment.missing_fields.join(", ")}</p>
        )}
      </div>

      {assessment.duplicate_of_id && (
        <p className="text-sm text-amber-700">
          Possible duplicate of{" "}
          <Link className="underline" to={`/complaints/${assessment.duplicate_of_id}`}>
            existing complaint
          </Link>{" "}
          ({Math.round((assessment.duplicate_confidence ?? 0) * 100)}% match)
        </p>
      )}

      <div>
        <p className="text-sm font-medium">Risk: {assessment.risk_classification ?? "Unclassified"}</p>
        <p className="text-xs text-slate-600">{assessment.risk_rationale}</p>
      </div>

      <div>
        <p className="text-sm font-medium">
          Regulatory Reportable: {assessment.regulatory_reportable === null ? "Unknown" : assessment.regulatory_reportable ? "Yes" : "No"}
        </p>
        <p className="text-xs text-slate-600">{assessment.regulatory_rationale}</p>
      </div>

      <div>
        <p className="text-sm font-medium">Root Cause</p>
        <p className="text-xs text-slate-600">{assessment.root_cause_suggestion}</p>
      </div>

      <div>
        <p className="text-sm font-medium">CAPA Recommendation</p>
        <p className="text-xs text-slate-600">{assessment.capa_recommendation}</p>
      </div>

      <div>
        <p className="text-sm font-medium">Summary</p>
        <p className="text-xs text-slate-600">{assessment.summary}</p>
      </div>
    </div>
  );
}
