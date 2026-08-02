import { configureStore } from "@reduxjs/toolkit";
import { renderHook, waitFor } from "@testing-library/react";
import { Provider } from "react-redux";
import { describe, expect, it, vi } from "vitest";

import intakeReducer from "./intakeSlice";
import { useIntakeStream } from "./useIntakeStream";

function sseBody(events: object[]): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  const frames = events.map((e) => `data: ${JSON.stringify(e)}\n\n`).join("");
  return new ReadableStream({
    start(controller) {
      controller.enqueue(encoder.encode(frames));
      controller.close();
    },
  });
}

describe("useIntakeStream", () => {
  it("parses SSE frames and dispatches completion", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        body: sseBody([
          { step: "extract_fields", status: "completed" },
          { step: "summarize", status: "token", data: "Hello " },
          { step: "finalize", status: "completed", data: { complaint_id: "abc-123" } },
        ]),
      })
    );

    const store = configureStore({ reducer: { intake: intakeReducer } });
    const wrapper = ({ children }: { children: React.ReactNode }) => <Provider store={store}>{children}</Provider>;
    const { result } = renderHook(() => useIntakeStream(), { wrapper });

    await result.current.start({ text: "test complaint" });

    await waitFor(() => {
      expect(store.getState().intake.status).toBe("done");
    });
    expect(store.getState().intake.resultComplaintId).toBe("abc-123");
    expect(store.getState().intake.summaryText).toBe("Hello ");
  });
});
