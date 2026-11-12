import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, type Job, type JobSubmitRequest, type PreviewRequest } from "@/lib/api";

const jobKeys = {
  all: ["jobs"] as const,
  detail: (id: string) => ["jobs", id] as const,
};

export function useJob(jobId: string | undefined) {
  return useQuery({
    queryKey: jobKeys.detail(jobId ?? ""),
    queryFn: () => api.getJob(jobId as string),
    enabled: !!jobId,
    // Poll while the job is still in flight; stop once it settles.
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      return status === "queued" || status === "running" ? 2000 : false;
    },
  });
}

export function useSubmitJob() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (payload: JobSubmitRequest) => api.submitJob(payload),
    onSuccess: (job: Job) => {
      queryClient.setQueryData(jobKeys.detail(job.id), job);
    },
  });
}

export function useCancelJob() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (jobId: string) => api.cancelJob(jobId),
    onSuccess: (job: Job) => {
      queryClient.setQueryData(jobKeys.detail(job.id), job);
    },
  });
}

export function useCreatePreview() {
  return useMutation({
    mutationFn: (payload: PreviewRequest) => api.createPreview(payload),
  });
}
