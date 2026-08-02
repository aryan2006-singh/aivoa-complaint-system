import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Provider } from "react-redux";
import { configureStore } from "@reduxjs/toolkit";
import { describe, expect, it } from "vitest";

import { api } from "../../services/api";
import ComplaintForm from "./ComplaintForm";

const complaint = {
  id: "1",
  complaint_number: "CMPL-2026-000001",
  product_name: "Amoxicillin",
  batch_lot_number: "B123",
  customer_name: null,
  customer_contact: null,
  source: "manual",
  date_received: "2026-08-01T00:00:00Z",
  description: "Tablets discolored.",
  category: null,
  severity: "Major",
  status: "New",
  assigned_to: null,
  assessment: null,
};

describe("ComplaintForm", () => {
  it("lets a reviewer edit the product name field", async () => {
    const store = configureStore({
      reducer: { [api.reducerPath]: api.reducer },
      middleware: (getDefault) => getDefault().concat(api.middleware),
    });
    render(
      <Provider store={store}>
        <ComplaintForm complaint={complaint} />
      </Provider>
    );
    const input = screen.getByDisplayValue("Amoxicillin") as HTMLInputElement;
    await userEvent.clear(input);
    await userEvent.type(input, "Ibuprofen");
    expect(input.value).toBe("Ibuprofen");
  });
});
