import { useState } from "react";
import { useSelector } from "react-redux";
import { useNavigate } from "react-router-dom";

import AiCopilotStepper from "../features/intake/AiCopilotStepper";
import { useIntakeStream } from "../features/intake/useIntakeStream";
import type { RootState } from "../store";

export default function Intake() {
  const [text, setText] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const { start } = useIntakeStream();
  const { status, resultComplaintId } = useSelector((state: RootState) => state.intake);
  const navigate = useNavigate();

  if (status === "done" && resultComplaintId) {
    navigate(`/complaints/${resultComplaintId}`);
  }

  return (
    <div className="p-6 grid grid-cols-2 gap-6">
      <div>
        <h1 className="text-2xl font-semibold mb-4">Log Customer Complaint</h1>
        <textarea
          className="w-full border rounded p-3 h-40"
          placeholder="Paste the complaint text (email, portal message, transcript)..."
          value={text}
          onChange={(e) => setText(e.target.value)}
        />
        <input
          type="file"
          accept=".pdf,.eml"
          className="mt-3"
          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
        />
        <button
          className="mt-4 rounded bg-slate-900 text-white px-4 py-2 disabled:opacity-50"
          disabled={status === "streaming" || (!text && !file)}
          onClick={() => start({ text: text || undefined, file: file || undefined })}
        >
          {status === "streaming" ? "Processing..." : "Submit"}
        </button>
      </div>
      <AiCopilotStepper />
    </div>
  );
}
