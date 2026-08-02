import { useCallback } from "react";
import { useDispatch } from "react-redux";

import type { AppDispatch } from "../../store";
import { intakeDone, intakeError, intakeStarted, stepUpdated, tokenReceived } from "./intakeSlice";

interface StartPayload {
  text?: string;
  file?: File;
}

export function useIntakeStream() {
  const dispatch = useDispatch<AppDispatch>();

  const start = useCallback(
    async (payload: StartPayload) => {
      dispatch(intakeStarted());
      const body = new FormData();
      if (payload.file) body.append("file", payload.file);
      if (payload.text) body.append("text", payload.text);

      try {
        const response = await fetch("/api/complaints/intake", { method: "POST", body });
        if (!response.ok || !response.body) {
          throw new Error(`Intake failed with status ${response.status}`);
        }
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        for (;;) {
          const { value, done } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const frames = buffer.split("\n\n");
          buffer = frames.pop() ?? "";
          for (const frame of frames) {
            const line = frame.trim();
            if (!line.startsWith("data:")) continue;
            const event = JSON.parse(line.slice("data:".length).trim());
            if (event.step === "finalize" || event.step === "finalize_duplicate") {
              dispatch(
                intakeDone({
                  complaintId: event.data?.complaint_id ?? null,
                  duplicateOf: event.data?.duplicate_of ?? null,
                })
              );
            } else if (event.step === "summarize" && event.status === "token") {
              dispatch(tokenReceived(event.data as string));
            } else {
              dispatch(stepUpdated(event));
            }
          }
        }
      } catch (err) {
        dispatch(intakeError(err instanceof Error ? err.message : "Unknown error"));
      }
    },
    [dispatch]
  );

  return { start };
}
