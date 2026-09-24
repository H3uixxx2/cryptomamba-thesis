import { useEffect, useState } from "react";
import { ChevronLeft, ChevronRight, Cpu, Play } from "lucide-react";

import { ForecastChart } from "@/components/ForecastChart";
import { Callout, ErrorBanner, Field, PageHeader, Panel, Pill } from "@/components/primitives";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Skeleton } from "@/components/ui/skeleton";
import { Slider } from "@/components/ui/slider";
import type { CheckpointModelOption, PredictMode, PredictResult } from "@/lib/api";
import { pct, signedPct, usd } from "@/lib/format";
import { useConsole } from "@/store";

const MODES: Array<{ id: PredictMode; label: string }> = [
  { id: "checkpoint", label: "Checkpoint" },
  { id: "historical", label: "Historical Replay" },
  { id: "live", label: "Live HTTP (Advanced)" },
];

function nextUtcDate(value: string): string {
  const date = new Date(`${value}T00:00:00Z`);
  date.setUTCDate(date.getUTCDate() + 1);
  return date.toISOString().slice(0, 10);
}

function predictionDateBounds(candles: Array<{ date: string }> | undefined, required: number) {
  const firstCompleteWindowEnd = candles?.[required - 1]?.date;
  const lastCandle = candles?.at(-1)?.date;
  return {
    minimumPredictionDate: firstCompleteWindowEnd ? nextUtcDate(firstCompleteWindowEnd) : "",
    maximumPredictionDate: lastCandle ? nextUtcDate(lastCandle) : "",
  };
}

function displayedVariant(result: PredictResult, fallbackLabel: string) {
  return (
    result.forecast_variants?.find((variant) => variant.id === "raw") ?? {
      predicted_close: result.predicted_close,
      move_pct: result.expected_return_pct ?? result.move_pct,
      error_pct: result.error_pct,
      direction_correct: result.direction_correct,
      label: fallbackLabel,
    }
  );
}

function ModeToggle() {
  const { predict } = useConsole();
  return (
    <div role="group" aria-label="Prediction mode" className="flex flex-wrap gap-1 rounded bg-[var(--bg-subtle)] p-1">
      {MODES.map((option) => (
        <button
          key={option.id}
          type="button"
          aria-pressed={predict.mode === option.id}
          onClick={() => predict.setMode(option.id)}
          className={
            "rounded px-3 py-1.5 text-[11px] font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--brand-accent)] " +
            (predict.mode === option.id
              ? "bg-[var(--bg-card)] text-[var(--brand-accent)] shadow-sm"
              : "text-[var(--text-muted)] hover:text-[var(--text-primary)]")
          }
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}

function CheckpointControls() {
  const { data, predict } = useConsole();
  const dataset = data.active;
  const models = predict.setup?.checkpoint_models ?? [];
  const minRequired = models.length ? Math.min(...models.map((m) => m.window_days)) : 14;
  const available = dataset?.candles.filter((candle) => candle.date < predict.predictDate).length ?? 0;
  const { minimumPredictionDate, maximumPredictionDate } = predictionDateBounds(dataset?.candles, minRequired);
  const dateWithinBounds = Boolean(
    predict.predictDate &&
      minimumPredictionDate &&
      maximumPredictionDate &&
      predict.predictDate >= minimumPredictionDate &&
      predict.predictDate <= maximumPredictionDate,
  );
  const ready = Boolean(
    dataset &&
      predict.setup?.checkpoint_available &&
      dateWithinBounds &&
      available >= minRequired &&
      !predict.loading,
  );

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-[minmax(0,1fr)_auto] xl:items-end">
        <div className="flex flex-col gap-2">
          <Label htmlFor="checkpoint-prediction-date" className="font-mono text-[11px] text-[var(--text-secondary)]">
            Prediction date
          </Label>
          <Input
            id="checkpoint-prediction-date"
            type="date"
            value={predict.predictDate}
            min={minimumPredictionDate}
            max={maximumPredictionDate}
            onChange={(event) => predict.setPredictDate(event.target.value)}
            className="border-[var(--line)] bg-[var(--bg-sunken)] font-mono text-xs [color-scheme:dark] xl:max-w-[220px]"
          />
        </div>

        <Button onClick={() => void predict.runCheckpoint()} disabled={!ready} className="shadow-none">
          <Cpu className="mr-2 h-3.5 w-3.5" />
          {predict.loading ? "Running…" : "Run both checkpoints"}
        </Button>
      </div>

      <div className="grid gap-2 sm:grid-cols-2">
        {models.map((model) => (
          <Field key={model.id} label={`${model.label} history`}>
            {available.toLocaleString("en-US")} available · {model.window_days} required
          </Field>
        ))}
      </div>
      <div className="grid gap-2 sm:grid-cols-2">
        <Field label="Dataset">{data.mode === "upload" ? data.uploadName || "Uploaded CSV" : "Thesis dataset"}</Field>
        <Field label="Execution">CPU · strict checkpoint · no fallback</Field>
      </div>

      {!predict.setup?.checkpoint_available && (
        <Callout tone="amber" title="Checkpoint inference unavailable">
          {predict.setup?.checkpoint_error ?? "The core worker or final checkpoint evidence is missing."}
        </Callout>
      )}
      {dataset && available < minRequired && (
        <Callout tone="amber" title="Insufficient history">
          Select a later prediction date or provide at least {minRequired} daily candles before the target date.
        </Callout>
      )}
      {dataset && predict.predictDate && minimumPredictionDate && maximumPredictionDate && !dateWithinBounds && (
        <Callout tone="amber" title="Prediction date outside active dataset">
          Select a date from {minimumPredictionDate} through {maximumPredictionDate}. The final allowed date is one day after the latest candle.
        </Callout>
      )}
    </div>
  );
}

function HistoricalControls() {
  const { predict } = useConsole();
  const dates = predict.setup?.offline_dates ?? [];
  const currentIndex = Math.max(0, dates.indexOf(predict.offlineDate));
  const [draftIndex, setDraftIndex] = useState(currentIndex);

  useEffect(() => setDraftIndex(currentIndex), [currentIndex]);

  return (
    <div className="grid gap-5 lg:grid-cols-[220px_minmax(0,1fr)] lg:items-center">
      <div className="flex items-center gap-2">
        <Button variant="outline" size="icon" disabled={currentIndex <= 0} onClick={() => predict.stepOfflineDate(-1)} aria-label="Previous replay date">
          <ChevronLeft className="h-4 w-4" />
        </Button>
        <div className="min-w-0 flex-1 text-center">
          <div className="eyebrow">Forecast date</div>
          <div className="tnum mt-1 text-base font-semibold">{predict.offlineDate || "—"}</div>
        </div>
        <Button variant="outline" size="icon" disabled={currentIndex >= dates.length - 1} onClick={() => predict.stepOfflineDate(1)} aria-label="Next replay date">
          <ChevronRight className="h-4 w-4" />
        </Button>
      </div>
      <div className="min-w-0">
        <Slider
          value={[draftIndex]}
          min={0}
          max={Math.max(0, dates.length - 1)}
          step={1}
          disabled={dates.length < 2 || predict.loading}
          onValueChange={([index]) => setDraftIndex(index)}
          onValueCommit={([index]) => {
            const date = dates[index];
            if (date) predict.setOfflineDate(date);
          }}
          aria-label="Historical forecast timeline"
          className="py-2"
        />
        <div className="mt-2 flex justify-between font-mono text-[10px] text-[var(--text-muted)]">
          <span>{dates[0] ?? "—"}</span><span>{dates.length} frozen forecasts</span><span>{dates.at(-1) ?? "—"}</span>
        </div>
      </div>
    </div>
  );
}

function LiveControls() {
  const { predict } = useConsole();
  return (
    <div className="grid grid-cols-1 items-end gap-4 lg:grid-cols-[minmax(0,1fr)_190px_auto]">
      <div className="flex flex-col gap-2">
        <Label htmlFor="prediction-service-url" className="font-mono text-[11px] text-[var(--text-secondary)]">Prediction service URL</Label>
        <Input id="prediction-service-url" value={predict.apiUrl} onChange={(event) => predict.setApiUrl(event.target.value)} placeholder="https://your-service.example" className="border-[var(--line)] bg-[var(--bg-sunken)] font-mono text-xs" />
      </div>
      <div className="flex flex-col gap-2">
        <Label htmlFor="live-forecast-date" className="font-mono text-[11px] text-[var(--text-secondary)]">Forecast date</Label>
        <Input id="live-forecast-date" type="date" value={predict.predictDate} onChange={(event) => predict.setPredictDate(event.target.value)} className="border-[var(--line)] bg-[var(--bg-sunken)] font-mono text-xs [color-scheme:dark]" />
      </div>
      <Button onClick={() => void predict.runLive()} disabled={predict.loading} className="shadow-none">
        <Play className="mr-2 h-3.5 w-3.5" />{predict.loading ? "Running…" : "Run HTTP inference"}
      </Button>
    </div>
  );
}

function CheckpointResultCard({ model, result, error, loading }: {
  model: CheckpointModelOption;
  result?: PredictResult;
  error?: string;
  loading: boolean;
}) {
  if (result) {
    const displayed = displayedVariant(result, model.label);
    return (
      <div className="flex min-w-0 flex-col gap-4">
        <Panel eyebrow={model.label} actions={<Pill tone="blue">{`${result.window_days ?? model.window_days}-day window`}</Pill>} bodyClassName="p-0">
          <ForecastChart candles={result.window.rows} predictionDate={result.prediction_date} predictedClose={displayed.predicted_close} actualClose={result.actual_close} variantLabel={displayed.label} />
        </Panel>

        {result.ood && (
          <Callout tone="amber" title="Out-of-distribution forecast">
            Extrapolative only. Not thesis or trading evidence.
          </Callout>
        )}

        <Panel eyebrow="Prediction output">
          <div className="tnum text-3xl font-semibold tracking-tight text-[var(--signal-blue)]">{usd(displayed.predicted_close)}</div>
          <div className="mt-4 grid grid-cols-2 gap-4 border-t border-[var(--line-subtle)] pt-4">
            <div><div className="eyebrow">Previous close</div><div className="tnum mt-1.5 text-base font-semibold">{usd(result.last_close)}</div></div>
            <div><div className="eyebrow">Expected return</div><div className="tnum mt-1.5 text-base font-semibold">{signedPct(displayed.move_pct)}</div></div>
          </div>
        </Panel>

        {result.actual_close != null && (
          <Panel eyebrow="Observed historical outcome">
            <div className="flex items-start justify-between gap-4">
              <div><div className="eyebrow">Actual close</div><div className="tnum mt-1.5 text-xl font-semibold">{usd(result.actual_close)}</div></div>
              <Pill tone={displayed.direction_correct ? "green" : "red"}>{displayed.direction_correct ? "Correct direction" : "Wrong direction"}</Pill>
            </div>
            <div className="mt-4 border-t border-[var(--line-subtle)] pt-4"><div className="eyebrow">Absolute error</div><div className="tnum mt-1.5 text-base font-semibold text-[var(--signal-amber)]">{pct(displayed.error_pct)}</div></div>
          </Panel>
        )}
      </div>
    );
  }

  if (loading) {
    return <Skeleton className="h-[420px] rounded-lg bg-[var(--bg-subtle)]" />;
  }

  if (error) {
    return (
      <Panel eyebrow={model.label}>
        <Callout tone="amber" title="Not run">{error}</Callout>
      </Panel>
    );
  }

  return (
    <Panel eyebrow={model.label}>
      <p className="text-xs text-[var(--text-muted)]">Run both checkpoints to see this model's forecast.</p>
    </Panel>
  );
}

function CheckpointResults() {
  const { predict } = useConsole();
  const models = predict.setup?.checkpoint_models ?? [];
  if (!models.length) return null;
  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
      {models.map((model) => (
        <CheckpointResultCard
          key={model.id}
          model={model}
          result={predict.checkpointResults[model.id]}
          error={predict.checkpointErrors[model.id]}
          loading={predict.loading}
        />
      ))}
    </div>
  );
}

function SingleResultPanel({ result, sourceLabel }: { result: PredictResult; sourceLabel: string }) {
  const { predict } = useConsole();
  const displayed = displayedVariant(result, result.model_id);
  return (
    <div className="grid grid-cols-1 gap-4 xl:grid-cols-[minmax(0,1fr)_340px]">
      <Panel eyebrow={`${result.window_days ?? result.window.size}-day causal input → next-day close`} actions={<Pill tone="blue">{result.prediction_date}</Pill>} bodyClassName="p-0">
        <ForecastChart candles={result.window.rows} predictionDate={result.prediction_date} predictedClose={displayed.predicted_close} actualClose={result.actual_close} variantLabel={displayed.label} />
      </Panel>

      <div className="flex min-w-0 flex-col gap-4">
        <Panel eyebrow="Prediction output" actions={<Pill tone={predict.mode === "historical" ? "blue" : "amber"}>{sourceLabel}</Pill>}>
          <div className="text-xs font-semibold text-[var(--text-primary)]">{displayed.label}</div>
          <div className="tnum mt-3 text-4xl font-semibold tracking-tight text-[var(--signal-blue)]">{usd(displayed.predicted_close)}</div>
          <div className="mt-5 grid grid-cols-2 gap-4 border-t border-[var(--line-subtle)] pt-4">
            <div><div className="eyebrow">Previous close</div><div className="tnum mt-1.5 text-lg font-semibold">{usd(result.last_close)}</div></div>
            <div><div className="eyebrow">Expected return</div><div className="tnum mt-1.5 text-lg font-semibold">{signedPct(displayed.move_pct)}</div></div>
          </div>
        </Panel>

        {result.actual_close != null && (
          <Panel eyebrow="Observed historical outcome">
            <div className="flex items-start justify-between gap-4">
              <div><div className="eyebrow">Actual close</div><div className="tnum mt-1.5 text-2xl font-semibold">{usd(result.actual_close)}</div></div>
              <Pill tone={displayed.direction_correct ? "green" : "red"}>{displayed.direction_correct ? "Correct direction" : "Wrong direction"}</Pill>
            </div>
            <div className="mt-4 border-t border-[var(--line-subtle)] pt-4"><div className="eyebrow">Absolute error</div><div className="tnum mt-1.5 text-lg font-semibold text-[var(--signal-amber)]">{pct(displayed.error_pct)}</div></div>
          </Panel>
        )}
      </div>
    </div>
  );
}

export function PredictScreen() {
  const { data, predict } = useConsole();
  const result = predict.result;
  const models = predict.setup?.checkpoint_models ?? [];
  const minRequired = models.length ? Math.min(...models.map((m) => m.window_days)) : 14;
  const availableHistory =
    data.active?.candles.filter((candle) => candle.date < predict.predictDate).length ?? 0;
  const { minimumPredictionDate, maximumPredictionDate } = predictionDateBounds(
    data.active?.candles,
    minRequired,
  );
  const checkpointReady = Boolean(
    data.active &&
      predict.setup?.checkpoint_available &&
      predict.predictDate &&
      minimumPredictionDate &&
      maximumPredictionDate &&
      predict.predictDate >= minimumPredictionDate &&
      predict.predictDate <= maximumPredictionDate &&
      availableHistory >= minRequired &&
      !predict.loading &&
      !predict.error,
  );
  const hasCheckpointOutput =
    Object.keys(predict.checkpointResults).length > 0 || Object.keys(predict.checkpointErrors).length > 0;

  const sourceLabel = predict.mode === "historical" ? "Historical replay" : "Optional live HTTP service";

  return (
    <>
      <PageHeader
        title="Next-Day Prediction"
        description="Primary flow: select the active dataset and run both frozen checkpoints. Historical replay remains available; live HTTP inference is optional."
        actions={<ModeToggle />}
      />

      {predict.error && <ErrorBanner message={predict.error} />}

      <Panel
        eyebrow={predict.mode === "checkpoint" ? "Active dataset → frozen checkpoints" : predict.mode === "historical" ? "Historical replay" : "Live HTTP inference · advanced"}
        actions={<Pill tone={predict.mode === "checkpoint" ? "green" : predict.mode === "historical" ? "blue" : "amber"}>{predict.mode === "checkpoint" ? "Primary" : predict.mode === "historical" ? "Replay" : "Optional"}</Pill>}
      >
        {predict.mode === "checkpoint" && <CheckpointControls />}
        {predict.mode === "historical" && <HistoricalControls />}
        {predict.mode === "live" && <LiveControls />}
      </Panel>

      {predict.mode === "checkpoint" && (predict.loading || hasCheckpointOutput) && <CheckpointResults />}

      {predict.mode !== "checkpoint" && predict.loading && !result && (
        <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_340px]">
          <Skeleton className="h-[480px] rounded-lg bg-[var(--bg-subtle)]" />
          <Skeleton className="h-[320px] rounded-lg bg-[var(--bg-subtle)]" />
        </div>
      )}

      {predict.mode !== "checkpoint" && result && <SingleResultPanel result={result} sourceLabel={sourceLabel} />}

      {predict.mode === "checkpoint" && !hasCheckpointOutput && !predict.loading && checkpointReady && (
        <Callout tone="blue" title="Ready for checkpoint inference">Pick a prediction date, then run. The backend uses only candles strictly before the target date, for each model's own window.</Callout>
      )}
      {predict.mode === "historical" && !result && predict.offlineUnavailable && (
        <Callout tone="amber" title="Historical artifact unavailable">No validated replay prediction is available for the selected date.</Callout>
      )}
    </>
  );
}
