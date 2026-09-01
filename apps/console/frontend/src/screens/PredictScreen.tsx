import { useEffect, useMemo, useState } from "react";
import { ChevronLeft, ChevronRight, Cpu, Play } from "lucide-react";

import { ForecastChart } from "@/components/ForecastChart";
import { Callout, ErrorBanner, Field, PageHeader, Panel, Pill } from "@/components/primitives";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Skeleton } from "@/components/ui/skeleton";
import { Slider } from "@/components/ui/slider";
import type { CheckpointModelId, PredictMode } from "@/lib/api";
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
  const option = predict.checkpointOption;
  const required = option?.window_days ?? (predict.checkpointModel === "s5_full" ? 60 : 14);
  const available = dataset?.candles.filter((candle) => candle.date < predict.predictDate).length ?? 0;
  const { minimumPredictionDate, maximumPredictionDate } = predictionDateBounds(dataset?.candles, required);
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
      available >= required &&
      !predict.loading,
  );

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-[minmax(0,1fr)_220px_auto] xl:items-end">
        <fieldset className="min-w-0">
          <legend className="mb-2 font-mono text-[11px] text-[var(--text-secondary)]">Frozen checkpoint</legend>
          <div role="radiogroup" aria-label="Frozen checkpoint" className="grid gap-2 sm:grid-cols-2">
            {(predict.setup?.checkpoint_models ?? []).map((model) => (
              <button
                key={model.id}
                type="button"
                role="radio"
                aria-checked={predict.checkpointModel === model.id}
                onClick={() => predict.setCheckpointModel(model.id as CheckpointModelId)}
                className={
                  "rounded-md border px-3 py-2 text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--brand-accent)] " +
                  (predict.checkpointModel === model.id
                    ? "border-[var(--brand-accent)] bg-[var(--brand-accent-soft)]"
                    : "border-[var(--line)] bg-[var(--bg-sunken)] hover:border-[var(--line-strong)]")
                }
              >
                <span className="block text-xs font-semibold text-[var(--text-primary)]">{model.label}</span>
                <span className="mt-1 block font-mono text-[10px] text-[var(--text-muted)]">
                  {model.window_days}-day window
                </span>
              </button>
            ))}
          </div>
        </fieldset>

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
            className="border-[var(--line)] bg-[var(--bg-sunken)] font-mono text-xs [color-scheme:dark]"
          />
        </div>

        <Button onClick={() => void predict.runCheckpoint()} disabled={!ready} className="shadow-none">
          <Cpu className="mr-2 h-3.5 w-3.5" />
          {predict.loading ? "Running…" : "Run checkpoint"}
        </Button>
      </div>

      <div className="grid gap-2 sm:grid-cols-3">
        <Field label="Dataset">{data.mode === "upload" ? data.uploadName || "Uploaded CSV" : "Thesis dataset"}</Field>
        <Field label="Causal history">{available.toLocaleString("en-US")} available · {required} required</Field>
        <Field label="Execution">CPU · strict checkpoint · no fallback</Field>
      </div>

      {!predict.setup?.checkpoint_available && (
        <Callout tone="amber" title="Checkpoint inference unavailable">
          {predict.setup?.checkpoint_error ?? "The core worker or final checkpoint evidence is missing."}
        </Callout>
      )}
      {dataset && available < required && (
        <Callout tone="amber" title="Insufficient history">
          Select a later prediction date or provide at least {required} daily candles before the target date.
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

export function PredictScreen() {
  const { data, predict } = useConsole();
  const result = predict.result;
  const requiredHistory =
    predict.checkpointOption?.window_days ?? (predict.checkpointModel === "s5_full" ? 60 : 14);
  const availableHistory =
    data.active?.candles.filter((candle) => candle.date < predict.predictDate).length ?? 0;
  const { minimumPredictionDate, maximumPredictionDate } = predictionDateBounds(
    data.active?.candles,
    requiredHistory,
  );
  const checkpointReady = Boolean(
    data.active &&
      predict.setup?.checkpoint_available &&
      predict.predictDate &&
      minimumPredictionDate &&
      maximumPredictionDate &&
      predict.predictDate >= minimumPredictionDate &&
      predict.predictDate <= maximumPredictionDate &&
      availableHistory >= requiredHistory &&
      !predict.loading &&
      !predict.error,
  );
  const displayed = useMemo(() => {
    if (!result) return null;
    return result.forecast_variants?.find((variant) => variant.id === "raw") ?? {
      predicted_close: result.predicted_close,
      move_pct: result.expected_return_pct ?? result.move_pct,
      error_pct: result.error_pct,
      direction_correct: result.direction_correct,
      label: predict.mode === "checkpoint" ? predict.checkpointOption?.label ?? result.model_id : "Frozen model output",
    };
  }, [predict.checkpointOption?.label, predict.mode, result]);

  const sourceLabel =
    predict.mode === "checkpoint" ? "Local frozen checkpoint" :
      predict.mode === "historical" ? "Historical replay" : "Optional live HTTP service";

  return (
    <>
      <PageHeader
        title="Next-Day Prediction"
        description="Primary flow: select the active dataset and a frozen CM-v or S5-Full checkpoint. Historical replay remains available; live HTTP inference is optional."
        actions={<ModeToggle />}
      />

      {predict.error && <ErrorBanner message={predict.error} />}

      <Panel
        eyebrow={predict.mode === "checkpoint" ? "Active dataset → frozen checkpoint" : predict.mode === "historical" ? "Historical replay" : "Live HTTP inference · advanced"}
        actions={<Pill tone={predict.mode === "checkpoint" ? "green" : predict.mode === "historical" ? "blue" : "amber"}>{predict.mode === "checkpoint" ? "Primary" : predict.mode === "historical" ? "Replay" : "Optional"}</Pill>}
      >
        {predict.mode === "checkpoint" && <CheckpointControls />}
        {predict.mode === "historical" && <HistoricalControls />}
        {predict.mode === "live" && <LiveControls />}
      </Panel>

      {result && predict.mode === "checkpoint" && result.ood && (
        <Callout tone="amber" title="Out-of-distribution checkpoint forecast">
          This forecast is qualitative and extrapolative only. It is not thesis evidence or trading evidence.
        </Callout>
      )}

      {predict.loading && !result && (
        <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_340px]">
          <Skeleton className="h-[480px] rounded-lg bg-[var(--bg-subtle)]" />
          <Skeleton className="h-[320px] rounded-lg bg-[var(--bg-subtle)]" />
        </div>
      )}

      {result && displayed && (
        <div className="grid grid-cols-1 gap-4 xl:grid-cols-[minmax(0,1fr)_340px]">
          <Panel eyebrow={`${result.window_days ?? result.window.size}-day causal input → next-day close`} actions={<Pill tone="blue">{result.prediction_date}</Pill>} bodyClassName="p-0">
            <ForecastChart candles={result.window.rows} predictionDate={result.prediction_date} predictedClose={displayed.predicted_close} actualClose={result.actual_close} variantLabel={displayed.label} />
          </Panel>

          <div className="flex min-w-0 flex-col gap-4">
            <Panel eyebrow="Prediction output" actions={<Pill tone={predict.mode === "checkpoint" ? "green" : "blue"}>{sourceLabel}</Pill>}>
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
      )}

      {!result && predict.mode === "checkpoint" && checkpointReady && (
        <Callout tone="blue" title="Ready for checkpoint inference">Choose a model and prediction date, then run. The backend uses only candles strictly before the target date.</Callout>
      )}
      {!result && predict.mode === "historical" && predict.offlineUnavailable && (
        <Callout tone="amber" title="Historical artifact unavailable">No validated replay prediction is available for the selected date.</Callout>
      )}
    </>
  );
}
