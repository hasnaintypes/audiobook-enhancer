/**
 * Typed client for the FastAPI job-lifecycle API (see docs/02-Architecture.md,
 * docs/07-Decisions.md — this contract is stable even before the API ships;
 * Phase 6 of the roadmap implements the server side).
 */

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export type EnhancementPreset = "gentle" | "balanced" | "strong" | "custom";

export type JobStatus =
  | "queued"
  | "running"
  | "succeeded"
  | "failed"
  | "cancelled";

export interface JobSubmitRequest {
  sourceUrl: string;
  preset: EnhancementPreset;
  addMusic?: boolean;
  idempotencyKey: string;
}

export interface Job {
  id: string;
  status: JobStatus;
  progress: number; // 0-100
  sourceUrl: string;
  preset: EnhancementPreset;
  error?: string;
  createdAt: string;
  updatedAt: string;
}

export interface JobArtifact {
  name: string;
  url: string;
  sizeBytes: number;
}

export interface PreviewRequest {
  sourceUrl: string;
  preset: EnhancementPreset;
}

export interface PreviewResult {
  originalUrl: string;
  enhancedUrl: string;
}

class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...init?.headers,
    },
  });

  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new ApiError(body || res.statusText, res.status);
  }

  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

export const api = {
  submitJob: (payload: JobSubmitRequest) =>
    request<Job>("/jobs", {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  getJob: (jobId: string) => request<Job>(`/jobs/${jobId}`),

  cancelJob: (jobId: string) =>
    request<Job>(`/jobs/${jobId}/cancel`, { method: "POST" }),

  getJobArtifacts: (jobId: string) =>
    request<JobArtifact[]>(`/jobs/${jobId}/artifacts`),

  createPreview: (payload: PreviewRequest) =>
    request<PreviewResult>("/previews", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
};

export { ApiError };
