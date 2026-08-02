import { render, screen } from "@testing-library/react";
import { Provider } from "react-redux";
import { MemoryRouter } from "react-router-dom";
import { configureStore } from "@reduxjs/toolkit";
import { describe, expect, it, vi } from "vitest";

import { api } from "../services/api";
import Dashboard from "./Dashboard";

vi.mock("../services/api", async () => {
  const actual = await vi.importActual<typeof import("../services/api")>("../services/api");
  return {
    ...actual,
    useGetComplaintsQuery: () => ({
      data: [
        {
          id: "1",
          complaint_number: "CMPL-2026-000001",
          product_name: "Amoxicillin",
          batch_lot_number: "B123",
          severity: "Major",
          status: "New",
          date_received: "2026-08-01T00:00:00Z",
        },
      ],
      isLoading: false,
    }),
  };
});

describe("Dashboard", () => {
  it("renders the complaint list", () => {
    const store = configureStore({ reducer: { [api.reducerPath]: api.reducer } });
    render(
      <Provider store={store}>
        <MemoryRouter>
          <Dashboard />
        </MemoryRouter>
      </Provider>
    );
    expect(screen.getByText("CMPL-2026-000001")).toBeInTheDocument();
    expect(screen.getByText("Major")).toBeInTheDocument();
  });
});
