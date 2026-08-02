import { useSelector } from "react-redux";

import type { RootState } from "../../store";

const PIPELINE_STEPS = [
  "extract_fields",
  "check_completeness",
  "check_duplicates",
  "classify_risk",
  "regulatory_reportability",
  "suggest_root_cause",
  "recommend_capa",
  "summarize",
];

const ICON: Record<string, string> = { pending: "○", started: "◐", completed: "●", error: "✕" };

export default function AiCopilotStepper() {
  const { steps, summaryText, status } = useSelector((state: RootState) => state.intake);

  return (
    <div className="rounded border p-4 space-y-2">
      <h2 className="font-semibold">AI Copilot</h2>
      <ul className="space-y-1 text-sm">
        {PIPELINE_STEPS.map((step) => {
          const s = steps[step]?.status ?? "pending";
          return (
            <li key={step} className="flex items-center gap-2">
              <span>{ICON[s]}</span>
              <span className={s === "completed" ? "text-slate-900" : "text-slate-400"}>
                {step.replaceAll("_", " ")}
              </span>
            </li>
          );
        })}
      </ul>
      {summaryText && (
        <div className="mt-3 rounded bg-slate-50 p-3 text-sm">
          <strong>Summary:</strong> {summaryText}
        </div>
      )}
      {status === "error" && <p className="text-red-600 text-sm">Something went wrong processing this complaint.</p>}
    </div>
  );
}
