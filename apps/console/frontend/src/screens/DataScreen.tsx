import { useEffect, useRef, useState, type DragEvent } from "react";
import { ArrowRight, CheckCircle2, RotateCcw, Trash2, Upload } from "lucide-react";

import { DataExplorerChart } from "@/components/DataExplorerChart";
import { Callout, ErrorBanner, PageHeader, Panel } from "@/components/primitives";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import type { DataResponse, SplitRow } from "@/lib/api";
import { useConsole } from "@/store";

const count = new Intl.NumberFormat("en-US");

function displayDate(value: string): string {
  const [year, month, day] = value.split("-");
  return year && month && day ? `${day}/${month}/${year}` : value;
}

function dateRange(dataset: DataResponse): string {
  const first = dataset.candles[0]?.date;
  const last = dataset.candles.at(-1)?.date;
  return first && last ? `${displayDate(first)} – ${displayDate(last)}` : "—";
}

function nextCalendarDate(date: string): string {
  const value = new Date(`${date}T00:00:00Z`);
  value.setUTCDate(value.getUTCDate() + 1);
  return value.toISOString().slice(0, 10);
}

function splitName(split: string): string {
  if (split === "train") return "Training";
  if (split === "validation") return "Validation";
  if (split === "test") return "Testing";
  return split;
}

function splitColor(split: string): string {
  if (split === "validation") return "var(--signal-blue)";
  if (split === "test") return "var(--orange-500)";
  return "var(--text-secondary)";
}

function splitMethod(dataset: DataResponse): string {
  return dataset.processing.split_strategy === "paper_date"
    ? "Fixed thesis date boundaries"
    : "Chronological 70% / 15% / 15%";
}

function SourceToggle() {
  const { data } = useConsole();
  const fileRef = useRef<HTMLInputElement>(null);
  const [dragActive, setDragActive] = useState(false);

  const upload = (file?: File) => {
    if (file && !data.loading) void data.uploadCsv(file);
  };
  const onDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragActive(false);
    upload(event.dataTransfer.files?.[0]);
  };

  return (
    <div
      role="group"
      aria-label="Select dataset source"
      aria-busy={data.loading}
      onDragEnter={(event) => {
        event.preventDefault();
        setDragActive(true);
      }}
      onDragOver={(event) => event.preventDefault()}
      onDragLeave={() => setDragActive(false)}
      onDrop={onDrop}
      className={
        "flex max-w-full flex-wrap items-center justify-end gap-1 rounded-full border bg-[var(--bg-subtle)] p-1 transition-colors " +
        (dragActive ? "border-[var(--brand-accent)] bg-[var(--brand-accent-soft)]" : "border-[var(--line)]")
      }
    >
      <input
        ref={fileRef}
        type="file"
        accept=".csv,text/csv"
        aria-label="Select a Bitcoin CSV file"
        className="hidden"
        disabled={data.loading}
        onChange={(event) => {
          upload(event.target.files?.[0]);
          event.target.value = "";
        }}
      />
      <button
        type="button"
        aria-pressed={data.mode === "paper"}
        disabled={data.loading && !data.active}
        onClick={() => void data.showPaper()}
        className={
          "rounded-full px-4 py-1.5 text-xs font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--brand-accent)] disabled:opacity-50 " +
          (data.mode === "paper"
            ? "bg-[var(--brand-accent)] text-white"
            : "text-[var(--text-secondary)] hover:text-[var(--text-primary)]")
        }
      >
        Thesis dataset
      </button>
      <button
        type="button"
        aria-pressed={data.mode === "upload"}
        disabled={data.loading}
        onClick={() => (data.hasUpload ? data.showUpload() : fileRef.current?.click())}
        className={
          "flex items-center gap-1.5 rounded-full px-4 py-1.5 text-xs font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--brand-accent)] disabled:opacity-50 " +
          (data.mode === "upload"
            ? "bg-[var(--brand-accent)] text-white"
            : "text-[var(--text-secondary)] hover:text-[var(--text-primary)]")
        }
      >
        <Upload className="h-3.5 w-3.5" />
        {data.loading ? "Processing…" : data.hasUpload ? "Uploaded CSV" : "Upload CSV"}
      </button>
      {data.hasUpload && (
        <>
          <button
            type="button"
            disabled={data.loading}
            onClick={() => fileRef.current?.click()}
            className="rounded-full px-3 py-1.5 text-[11px] text-[var(--text-muted)] hover:text-[var(--text-secondary)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--brand-accent)] disabled:opacity-50"
          >
            Replace
          </button>
          <button
            type="button"
            aria-label="Remove uploaded CSV"
            disabled={data.loading}
            onClick={data.clearUpload}
            className="rounded-full p-1.5 text-[var(--text-muted)] hover:bg-[var(--bg-hover)] hover:text-[var(--signal-red)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--brand-accent)] disabled:opacity-50"
          >
            <Trash2 className="h-3.5 w-3.5" />
          </button>
        </>
      )}
    </div>
  );
}

function ProcessingSummary({ dataset }: { dataset: DataResponse }) {
  const reducedRows = dataset.processing.raw_rows - dataset.processing.daily_rows;

  return (
    <Panel
      eyebrow="Processing result"
    >
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
        <div className="rounded-md border border-[var(--line-subtle)] bg-[var(--bg-subtle)] p-4">
          <div className="eyebrow">Raw rows</div>
          <div className="mt-2 font-mono text-2xl font-semibold text-[var(--text-primary)]">
            {count.format(dataset.processing.raw_rows)}
          </div>
        </div>
        <div className="rounded-md border border-[var(--line-subtle)] bg-[var(--bg-subtle)] p-4">
          <div className="eyebrow">Daily rows</div>
          <div className="mt-2 font-mono text-2xl font-semibold text-[var(--brand-accent)]">
            {count.format(dataset.processing.daily_rows)}
          </div>
        </div>
        <div className="rounded-md border border-[var(--line-subtle)] bg-[var(--bg-subtle)] p-4">
          <div className="eyebrow">Date range</div>
          <div className="mt-2 font-mono text-sm font-semibold leading-relaxed text-[var(--text-primary)]">
            {dateRange(dataset)}
          </div>
        </div>
      </div>

      <div className="mt-4 flex flex-col gap-2 border-t border-[var(--line-subtle)] pt-4 text-xs text-[var(--text-secondary)] sm:flex-row sm:items-center sm:justify-between">
        <span className="flex items-center gap-2">
          <CheckCircle2 className="h-4 w-4 text-[var(--signal-green)]" />
          {reducedRows > 0
            ? `${count.format(reducedRows)} rows aggregated into daily candles.`
            : "One row per day; no aggregation required."}
        </span>
        <span className="font-medium text-[var(--text-primary)]">{splitMethod(dataset)}</span>
      </div>
    </Panel>
  );
}

function DatasetComparison({ paper, uploaded }: { paper: DataResponse; uploaded: DataResponse }) {
  const rows = [
    ["Date range", dateRange(paper), dateRange(uploaded)],
    ["Raw rows", count.format(paper.processing.raw_rows), count.format(uploaded.processing.raw_rows)],
    ["Daily rows", count.format(paper.processing.daily_rows), count.format(uploaded.processing.daily_rows)],
    ["Split method", splitMethod(paper), splitMethod(uploaded)],
    ["Checkpoint history", "CM-v 14 days · S5-Full 60 days", "CM-v 14 days · S5-Full 60 days"],
  ];

  return (
    <Panel
      eyebrow="Processed dataset comparison"
      bodyClassName="p-0"
    >
      <Table>
        <TableHeader>
          <TableRow className="border-[var(--line)] hover:bg-transparent">
            <TableHead className="eyebrow pl-5">Metric</TableHead>
            <TableHead className="eyebrow">Thesis dataset</TableHead>
            <TableHead className="eyebrow pr-5">Uploaded CSV</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {rows.map(([label, paperValue, uploadValue]) => (
            <TableRow key={label} className="border-[var(--line-subtle)] hover:bg-[var(--bg-hover)]">
              <TableCell className="pl-5 text-xs font-medium text-[var(--text-primary)]">{label}</TableCell>
              <TableCell className="font-mono text-xs text-[var(--text-secondary)]">{paperValue}</TableCell>
              <TableCell className="pr-5 font-mono text-xs text-[var(--text-secondary)]">{uploadValue}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </Panel>
  );
}

function SplitOverview({ splits }: { splits: SplitRow[] }) {
  return (
    <Panel
      eyebrow="Chronological data split"
    >
      <div className="grid grid-cols-1 gap-3 lg:grid-cols-3">
        {splits.map((split) => (
          <div
            key={split.split}
            className="rounded-md border border-[var(--line-subtle)] bg-[var(--bg-subtle)] p-4"
            style={{ borderTopColor: splitColor(split.split), borderTopWidth: 3 }}
          >
            <div className="flex items-baseline justify-between gap-3">
              <div className="text-sm font-semibold text-[var(--text-primary)]">{splitName(split.split)}</div>
              <div className="font-mono text-xl font-semibold" style={{ color: splitColor(split.split) }}>
                {count.format(split.rows)}
              </div>
            </div>
            <div className="mt-2 font-mono text-[10.5px] text-[var(--text-muted)]">
              {displayDate(split.from)} – {displayDate(split.to)}
            </div>
          </div>
        ))}
      </div>
    </Panel>
  );
}

function SlidingWindowLens({
  dataset,
  targetDate,
  loading,
  onReset,
}: {
  dataset: DataResponse;
  targetDate: string;
  loading: boolean;
  onReset: () => void;
}) {
  return (
    <Panel
      eyebrow="Reproduced CM-v window preview · 14 days"
      actions={
        <Button type="button" variant="outline" size="sm" onClick={onReset} disabled={loading}>
          <RotateCcw className="h-3.5 w-3.5" /> Latest
        </Button>
      }
    >
      <div className="grid grid-cols-1 items-stretch gap-3 xl:grid-cols-[minmax(0,1fr)_auto_220px]">
        <div className="min-w-0 rounded-lg border-2 border-[var(--brand-accent)] bg-[var(--brand-accent-soft)] p-3">
          <div className="mb-3 flex items-center justify-between gap-3">
            <span className="text-xs font-semibold text-[var(--brand-accent)]">CM-v model-ready input</span>
            <span className="font-mono text-[10px] text-[var(--text-muted)]">
              {displayDate(dataset.window.start)} – {displayDate(dataset.window.end)}
            </span>
          </div>
          <div className="grid grid-cols-7 gap-1.5" aria-label="Fourteen-day causal input window">
            {dataset.window.rows.map((row, index) => (
              <div
                key={row.date}
                className="rounded border border-[var(--line-strong)] bg-[var(--bg-card)] px-1 py-2 text-center shadow-sm"
                title={displayDate(row.date)}
              >
                <div className="font-mono text-[9px] text-[var(--text-muted)]">{index + 1}</div>
                <div className="mt-1 font-mono text-[10px] font-semibold text-[var(--text-primary)]">
                  {row.date.slice(5)}
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="flex items-center justify-center text-[var(--text-muted)]">
          <ArrowRight className="h-6 w-6 rotate-90 xl:rotate-0" />
        </div>

        <div className="flex flex-col justify-center rounded-lg border-2 border-dashed border-[#9a7cff] bg-[#9a7cff]/10 p-4 text-center">
          <div className="eyebrow text-[#9a7cff]">Target date</div>
          <div className="mt-2 font-mono text-lg font-semibold text-[var(--text-primary)]">{displayDate(targetDate)}</div>
          <div className="mt-2 text-[11px] text-[var(--text-muted)]">Next-day close</div>
        </div>
      </div>

      {loading && (
        <p className="mt-4 text-center text-xs text-[var(--text-secondary)]" aria-live="polite">
          Updating window…
        </p>
      )}
    </Panel>
  );
}

export function DataScreen() {
  const { data } = useConsole();
  const dataset = data.active;
  const [pendingTarget, setPendingTarget] = useState<string | null>(null);
  const [selectionNotice, setSelectionNotice] = useState("");

  useEffect(() => {
    setPendingTarget(null);
    setSelectionNotice("");
  }, [dataset?.candles]);

  useEffect(() => {
    if (!data.windowLoading) setPendingTarget(null);
  }, [data.windowLoading]);

  const committedTarget = dataset?.prediction_payload.prediction_date ?? "";
  const targetDate = data.windowLoading && pendingTarget ? pendingTarget : committedTarget;
  const targetIsCandle = Boolean(dataset?.candles.some((candle) => candle.date === targetDate));

  const selectTargetDate = (date: string) => {
    data.cancelWindowSelection();
    setSelectionNotice("");
    if (!dataset || date === committedTarget) return;

    const targetIndex = dataset.candles.findIndex((candle) => candle.date === date);
    if (targetIndex < dataset.window.size) {
      setSelectionNotice(`At least ${dataset.window.size} prior days are required for ${displayDate(date)}.`);
      return;
    }

    const candles = dataset.candles.slice(targetIndex - dataset.window.size, targetIndex);
    setPendingTarget(date);
    void data.selectWindow(candles, date);
  };

  const resetLatestWindow = () => {
    if (!dataset) return;
    const lastCandle = dataset.candles.at(-1);
    if (!lastCandle) return;

    const latestTarget = nextCalendarDate(lastCandle.date);
    data.cancelWindowSelection();
    setSelectionNotice("");
    if (committedTarget === latestTarget) return;

    setPendingTarget(latestTarget);
    void data.selectWindow(dataset.candles.slice(-dataset.window.size), latestTarget);
  };

  return (
    <div aria-busy={data.loading}>
      <PageHeader
        title="Data"
        description="Validate raw OHLCV candles first. Predict then constructs the model-specific causal window: 14 days for reproduced CM-v or 60 days for S5-Full."
        actions={<SourceToggle />}
      />

      <div className="mt-5 flex flex-col gap-4">
        {data.error && (
          <div aria-live="assertive">
            <ErrorBanner message={data.error} />
            {dataset && <p className="mt-1 text-xs text-[var(--text-muted)]">The last valid dataset remains active.</p>}
          </div>
        )}
        {data.loading && dataset && (
          <div aria-live="polite">
            <Callout tone="amber" title="Processing CSV">
              The current dataset remains active until validation completes.
            </Callout>
          </div>
        )}
        {data.windowError && dataset && (
          <div aria-live="assertive">
            <ErrorBanner message={data.windowError} />
            <p className="mt-1 text-xs text-[var(--text-muted)]">The last valid 14-day window remains active.</p>
          </div>
        )}

        {data.loading && !dataset && (
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
            {[0, 1, 2].map((index) => (
              <Skeleton key={index} className="h-[140px] rounded-lg bg-[var(--bg-subtle)]" />
            ))}
            <Skeleton className="h-[520px] rounded-lg bg-[var(--bg-subtle)] md:col-span-2 xl:col-span-3" />
          </div>
        )}

        {dataset && (
          <>
            <Callout
              tone={data.mode === "paper" ? "blue" : "amber"}
              title={data.mode === "paper" ? "Thesis dataset" : "Uploaded dataset"}
            >
              {data.mode === "paper"
                ? "BTC/USD daily OHLCV"
                : data.uploadName || dataset.source.detail}
            </Callout>

            <Callout tone="neutral" title="Raw candles versus model-ready samples">
              This screen keeps the complete validated candle set. The 14-day lens below previews the reproduced CM-v contract only; the Predict screen independently selects the last 14 or 60 candles strictly before its target date.
            </Callout>

            <ProcessingSummary dataset={dataset} />

            {data.paper && data.uploaded && <DatasetComparison paper={data.paper} uploaded={data.uploaded} />}

            <SplitOverview splits={dataset.splits} />

            <Panel
              eyebrow="Bitcoin price timeline"
              bodyClassName="p-0"
            >
              <DataExplorerChart
                candles={dataset.candles}
                selectedDate={targetIsCandle ? targetDate : null}
                windowStart={dataset.window.start}
                windowEnd={dataset.window.end}
                onSelectDate={selectTargetDate}
                height={440}
              />
            </Panel>

            {selectionNotice && (
              <div aria-live="polite">
                <Callout tone="amber">{selectionNotice}</Callout>
              </div>
            )}

            <SlidingWindowLens
              dataset={dataset}
              targetDate={targetDate}
              loading={data.windowLoading}
              onReset={resetLatestWindow}
            />
          </>
        )}
      </div>
    </div>
  );
}
