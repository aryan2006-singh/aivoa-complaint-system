import { createApi, fetchBaseQuery } from "@reduxjs/toolkit/query/react";

export interface ExtractedField {
  value: string | null;
  confidence: number;
}

export interface AIAssessment {
  completeness_score: number;
  missing_fields: string[];
  duplicate_of_id: string | null;
  duplicate_confidence: number | null;
  risk_classification: string | null;
  risk_rationale: string | null;
  regulatory_reportable: boolean | null;
  regulatory_rationale: string | null;
  root_cause_suggestion: string | null;
  capa_recommendation: string | null;
  summary: string | null;
}

export interface Complaint {
  id: string;
  complaint_number: string;
  product_name: string;
  batch_lot_number: string;
  customer_name: string | null;
  customer_contact: string | null;
  source: string;
  date_received: string;
  description: string;
  category: string | null;
  severity: string | null;
  status: string;
  assigned_to: string | null;
  assessment: AIAssessment | null;
}

export interface HistoryEntry {
  field: string;
  old_value: string | null;
  new_value: string | null;
  changed_by: string;
  changed_at: string;
}

export const api = createApi({
  reducerPath: "api",
  baseQuery: fetchBaseQuery({ baseUrl: "/api" }),
  tagTypes: ["Complaint"],
  endpoints: (builder) => ({
    getComplaints: builder.query<Complaint[], Record<string, string | number | undefined>>({
      query: (params) => ({ url: "/complaints", params }),
      providesTags: ["Complaint"],
    }),
    getComplaint: builder.query<Complaint, string>({
      query: (id) => `/complaints/${id}`,
      providesTags: ["Complaint"],
    }),
    updateComplaint: builder.mutation<Complaint, { id: string; patch: Partial<Complaint> & { changed_by?: string } }>({
      query: ({ id, patch }) => ({ url: `/complaints/${id}`, method: "PATCH", body: patch }),
      invalidatesTags: ["Complaint"],
    }),
    getComplaintHistory: builder.query<HistoryEntry[], string>({
      query: (id) => `/complaints/${id}/history`,
      providesTags: ["Complaint"],
    }),
  }),
});

export const {
  useGetComplaintsQuery,
  useGetComplaintQuery,
  useUpdateComplaintMutation,
  useGetComplaintHistoryQuery,
} = api;
