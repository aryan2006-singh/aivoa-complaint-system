import { configureStore } from "@reduxjs/toolkit";
import { render, screen } from "@testing-library/react";
import { Provider } from "react-redux";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it } from "vitest";

import intakeReducer, { type StepState } from "../features/intake/intakeSlice";
import Intake from "./Intake";

interface IntakeStateOverrides {
  status?: "idle" | "streaming" | "done" | "error";
  steps?: Record<string, StepState>;
  summaryText?: string;
  resultComplaintId?: string | null;
  duplicateOf?: string | null;
  error?: string | null;
}

function renderIntake(overrides: IntakeStateOverrides) {
  const store = configureStore({
    reducer: { intake: intakeReducer },
    preloadedState: {
      intake: {
        status: overrides.status ?? "idle",
        steps: overrides.steps ?? {},
        summaryText: overrides.summaryText ?? "",
        resultComplaintId: overrides.resultComplaintId ?? null,
        duplicateOf: overrides.duplicateOf ?? null,
        error: overrides.error ?? null,
      },
    },
  });
  render(
    <Provider store={store}>
      <MemoryRouter>
        <Intake />
      </MemoryRouter>
    </Provider>
  );
}

describe("Intake", () => {
  it("shows duplicate feedback with a link to the existing complaint when a hard duplicate is found", () => {
    renderIntake({ status: "done", resultComplaintId: null, duplicateOf: "existing-id-123" });

    expect(screen.getByText(/appears to be a duplicate/i)).toBeInTheDocument();
    const link = screen.getByRole("link", { name: /view existing complaint/i });
    expect(link).toHaveAttribute("href", "/complaints/existing-id-123");
  });

  it("does not show duplicate feedback while idle", () => {
    renderIntake({});

    expect(screen.queryByText(/appears to be a duplicate/i)).not.toBeInTheDocument();
  });
});
