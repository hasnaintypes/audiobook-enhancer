"use client";

import { useState } from "react";
import { toast } from "sonner";

import {
  isCancellable,
  type EnhancementPreset,
  type JobStatus,
} from "@/lib/api";
import {
  useCancelJob,
  useJob,
  useJobArtifacts,
  useSubmitJob,
} from "@/lib/queries/jobs";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";

const PRESETS: EnhancementPreset[] = ["gentle", "balanced", "strong", "custom"];

const STATUS_VARIANT: Record<
  JobStatus,
  "default" | "secondary" | "destructive" | "outline"
> = {
  queued: "secondary",
  running: "default",
  succeeded: "outline",
  failed: "destructive",
  cancelled: "outline",
};

export default function Home() {
  const [sourceUrl, setSourceUrl] = useState("");
  const [preset, setPreset] = useState<EnhancementPreset>("balanced");
  const [activeJobId, setActiveJobId] = useState<string | undefined>();

  const submitJob = useSubmitJob();
  const job = useJob(activeJobId);
  const cancelJob = useCancelJob();
  const artifacts = useJobArtifacts(activeJobId, job.data?.status);

  function handleSubmit(event: React.FormEvent) {
    event.preventDefault();
    if (!sourceUrl.trim()) return;

    submitJob.mutate(
      {
        sourceUrl: sourceUrl.trim(),
        preset,
        addMusic: false,
        idempotencyKey: crypto.randomUUID(),
      },
      {
        onSuccess: (submitted) => setActiveJobId(submitted.id),
        onError: (error) => toast.error(`Could not submit job: ${error.message}`),
      },
    );
  }

  function handleCancel() {
    if (!activeJobId) return;
    cancelJob.mutate(activeJobId, {
      onError: (error) => toast.error(`Could not cancel job: ${error.message}`),
    });
  }

  return (
    <div className="flex flex-col flex-1 items-center bg-zinc-50 py-16 dark:bg-black">
      <main className="flex w-full max-w-xl flex-col gap-6 px-4">
        <div>
          <h1 className="text-2xl font-semibold">Audiobook Enhancer</h1>
          <p className="text-sm text-muted-foreground">
            Submit a YouTube narration URL to clean it up.
          </p>
        </div>

        <Card>
          <CardHeader>
            <CardTitle>Submit a job</CardTitle>
            <CardDescription>
              The source video is downloaded and processed once a Colab worker picks it up.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit} className="flex flex-col gap-4">
              <div className="flex flex-col gap-1.5">
                <Label htmlFor="sourceUrl">Source URL</Label>
                <Input
                  id="sourceUrl"
                  placeholder="https://youtu.be/..."
                  value={sourceUrl}
                  onChange={(event) => setSourceUrl(event.target.value)}
                  required
                />
              </div>

              <div className="flex flex-col gap-1.5">
                <Label htmlFor="preset">Preset</Label>
                <div className="flex gap-1.5">
                  {PRESETS.map((option) => (
                    <Button
                      key={option}
                      type="button"
                      size="sm"
                      variant={option === preset ? "default" : "outline"}
                      onClick={() => setPreset(option)}
                    >
                      {option}
                    </Button>
                  ))}
                </div>
              </div>

              <div className="flex items-center justify-between rounded-lg border border-border px-3 py-2">
                <div>
                  <p className="text-sm font-medium text-muted-foreground">
                    Add background music
                  </p>
                  <p className="text-xs text-muted-foreground">
                    No music-source mechanism exists yet -- not implemented.
                  </p>
                </div>
                <Badge variant="secondary">Coming soon</Badge>
              </div>

              <Button type="submit" disabled={submitJob.isPending}>
                {submitJob.isPending ? "Submitting..." : "Submit"}
              </Button>
            </form>
          </CardContent>
        </Card>

        {activeJobId && (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                <span>Job status</span>
                {job.data && (
                  <Badge variant={STATUS_VARIANT[job.data.status]}>
                    {job.data.status}
                  </Badge>
                )}
              </CardTitle>
              <CardDescription>{activeJobId}</CardDescription>
            </CardHeader>
            <CardContent className="flex flex-col gap-3">
              {job.isLoading && <p className="text-sm text-muted-foreground">Loading...</p>}
              {job.data && (
                <>
                  <Progress value={job.data.progress}>
                    <span className="text-sm text-muted-foreground">
                      {job.data.progress}%
                    </span>
                  </Progress>
                  <p className="truncate text-sm text-muted-foreground">
                    {job.data.sourceUrl}
                  </p>
                  {job.data.error && (
                    <p className="text-sm text-destructive">{job.data.error}</p>
                  )}
                </>
              )}
            </CardContent>
            {job.data && isCancellable(job.data.status) && (
              <CardFooter>
                <Button
                  variant="destructive"
                  size="sm"
                  onClick={handleCancel}
                  disabled={cancelJob.isPending}
                >
                  {cancelJob.isPending ? "Cancelling..." : "Cancel"}
                </Button>
              </CardFooter>
            )}
          </Card>
        )}

        {activeJobId && job.data?.status === "succeeded" && (
          <Card>
            <CardHeader>
              <CardTitle>Artifacts</CardTitle>
            </CardHeader>
            <CardContent className="flex flex-col gap-2">
              {artifacts.isLoading && (
                <p className="text-sm text-muted-foreground">Loading...</p>
              )}
              {artifacts.data?.length === 0 && (
                <p className="text-sm text-muted-foreground">No artifacts produced.</p>
              )}
              {artifacts.data?.map((artifact) => (
                <a
                  key={artifact.name}
                  href={artifact.url}
                  download
                  className="flex items-center justify-between rounded-lg border border-border px-3 py-2 text-sm hover:bg-muted"
                >
                  <span>{artifact.name}</span>
                  <span className="text-muted-foreground">
                    {(artifact.sizeBytes / 1_000_000).toFixed(1)} MB
                  </span>
                </a>
              ))}
            </CardContent>
          </Card>
        )}

        <Separator />

        <Card className="opacity-60">
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              <span>Before/after preview</span>
              <Badge variant="secondary">Coming soon</Badge>
            </CardTitle>
            <CardDescription>
              Needs real audio processing the API can&apos;t do yet.
            </CardDescription>
          </CardHeader>
        </Card>
      </main>
    </div>
  );
}
