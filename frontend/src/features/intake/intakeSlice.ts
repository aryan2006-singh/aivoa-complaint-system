import { createSlice, type PayloadAction } from "@reduxjs/toolkit";

export interface StepState {
  status: "pending" | "started" | "completed" | "error";
  data?: unknown;
}

interface IntakeState {
  status: "idle" | "streaming" | "done" | "error";
  steps: Record<string, StepState>;
  summaryText: string;
  resultComplaintId: string | null;
  duplicateOf: string | null;
  error: string | null;
}

const initialState: IntakeState = {
  status: "idle",
  steps: {},
  summaryText: "",
  resultComplaintId: null,
  duplicateOf: null,
  error: null,
};

const intakeSlice = createSlice({
  name: "intake",
  initialState,
  reducers: {
    intakeStarted(state) {
      Object.assign(state, initialState, { status: "streaming" as const });
    },
    stepUpdated(state, action: PayloadAction<{ step: string; status: StepState["status"]; data?: unknown }>) {
      const { step, status, data } = action.payload;
      state.steps[step] = { status, data };
    },
    tokenReceived(state, action: PayloadAction<string>) {
      state.summaryText += action.payload;
    },
    intakeDone(state, action: PayloadAction<{ complaintId: string | null; duplicateOf: string | null }>) {
      state.status = "done";
      state.resultComplaintId = action.payload.complaintId;
      state.duplicateOf = action.payload.duplicateOf;
    },
    intakeError(state, action: PayloadAction<string>) {
      state.status = "error";
      state.error = action.payload;
    },
    intakeReset() {
      return initialState;
    },
  },
});

export const { intakeStarted, stepUpdated, tokenReceived, intakeDone, intakeError, intakeReset } = intakeSlice.actions;
export default intakeSlice.reducer;
