import { useMemo, type ReactNode } from "react";

import { Plot } from "@/components/Plot";
import { Callout, ErrorBanner, PageHeader, Panel, Pill, StatTile } from "@/components/primitives";
import { Skeleton } from "@/components/ui/skeleton";
import type {
  ControlledForecastMetricRow,
  ForecastRobustness,
  PaperReportedMetricRow,
  PlotlyFigure,
  Reproduction350d,
} from "@/lib/api";
import { fmt, pct, prettyModel } from "@/lib/format";
import { useConsole } from "@/store";

const MODEL_ORDER = ["cmamba_v_reproduced", "s5_full", "naive_persistence"];

function splitLabel(split: string): string {
  return split === "val" ? "Validation" : "Test";
}

function estimateWithInterval(
  estimate: number | undefined,
  low: number | undefined,
  high: number | undefined,
  suffix = "",
): string {
  if (estimate == null || low == null || high == null) return "—";
  return `${fmt(estimate, 2)}${suffix} [${fmt(low, 2)}${suffix}, ${fmt(high, 2)}${suffix}]`;
}

function metricFigure(rows: ControlledForecastMetricRow[]): PlotlyFigure | undefined {
  if (!rows.length) return undefined;
  const labels = MODEL_ORDER.map((id) => rows.find((row) => row.model_id === id)?.display_name ?? id);
  return {
    data: (["val", "test"] as const).map((split, index) => ({
      type: "bar",
      name: split === "val" ? "Validation" : "Test",
      x: labels,
      y: MODEL_ORDER.map((id) => rows.find((row) => row.model_id === id && row.split === split)?.RMSE ?? null),
      marker: { color: index === 0 ? "#4d8dff" : "#b8651f" },
      text: MODEL_ORDER.map((id) => {
        const value = rows.find((row) => row.model_id === id && row.split === split)?.RMSE;
        return value == null ? "—" : value.toFixed(1);
      }),
      textposition: "outside",
      cliponaxis: false,
      hovertemplate: "%{x}<br>%{fullData.name}: %{y:.2f}<extra></extra>",
    })),
    layout: {
      barmode: "group",
      height: 350,
      yaxis: { title: "RMSE · lower is better", rangemode: "tozero" },
      legend: { orientation: "h", y: 1.12, x: 1, xanchor: "right" },
    },
  };
}

function ExactValues({ children }: { children: ReactNode }) {
  return (
    <details className="mt-4 rounded border border-[var(--line-subtle)] bg-[var(--bg-sunken)]">
      <summary className="cursor-pointer px-3 py-2 font-mono text-[11px] font-medium text-[var(--text-secondary)]">
        View exact values
      </summary>
      <div className="overflow-x-auto border-t border-[var(--line-subtle)] p-2">{children}</div>
    </details>
  );
}

function zeroReferenceLine(): Record<string, unknown> {
  return {
    type: "line",
    xref: "x",
    yref: "paper",
    x0: 0,
    x1: 0,
    y0: 0,
    y1: 1,
    line: { color: "#8795a8", width: 1.25, dash: "dash" },
  };
}

function gapToPaperFigure(data: Reproduction350d): PlotlyFigure | undefined {
  if (!data.rows.length || !data.criterion) return undefined;
  const metrics = ["RMSE", "MAE", "MAPE"];
  const gaps = data.rows.flatMap((row) => [row.RMSE_gap_pct, row.MAE_gap_pct, row.MAPE_gap_pct]);
  const upper = Math.max(data.criterion.threshold_pct, ...gaps) * 1.18;
  return {
    data: data.rows.map((row) => ({
      type: "bar",
      name: row.display_name,
      x: metrics,
      y: [row.RMSE_gap_pct, row.MAE_gap_pct, row.MAPE_gap_pct],
      marker: {
        color: row.result_type === "retrained_checkpoint" ? "#4d8dff" : "#8795a8",
      },
      text: [row.RMSE_gap_pct, row.MAE_gap_pct, row.MAPE_gap_pct].map(
        (value) => `${value.toFixed(3)}%`,
      ),
      textposition: "outside",
      cliponaxis: false,
      hovertemplate: "%{fullData.name}<br>%{x} gap: %{y:.3f}%<extra></extra>",
    })),
    layout: {
      barmode: "group",
      height: 300,
      yaxis: { title: "Absolute gap to paper (%)", range: [0, upper] },
      legend: { orientation: "h", y: 1.14, x: 1, xanchor: "right" },
      shapes: [{
        type: "line",
        xref: "paper",
        yref: "y",
        x0: 0,
        x1: 1,
        y0: data.criterion.threshold_pct,
        y1: data.criterion.threshold_pct,
        line: { color: "#b8651f", width: 1.25, dash: "dash" },
      }],
      annotations: [{
        xref: "paper",
        yref: "y",
        x: 1,
        y: data.criterion.threshold_pct,
        text: `Local criterion ${data.criterion.threshold_pct}%`,
        showarrow: false,
        xanchor: "right",
        yanchor: "bottom",
      }],
    },
  };
}

function primaryRmseForestFigure(data: ForecastRobustness): PlotlyFigure | undefined {
  if (!data.rmse_difference.length) return undefined;
  const rows = data.rmse_difference;
  return {
    data: [{
      type: "scatter",
      mode: "markers",
      x: rows.map((row) => row.estimate),
      y: rows.map((row) => splitLabel(row.split)),
      marker: { size: 10, color: rows.map((row) => row.split === "val" ? "#4d8dff" : "#b8651f") },
      error_x: {
        type: "data",
        symmetric: false,
        array: rows.map((row) => row.ci_high - row.estimate),
        arrayminus: rows.map((row) => row.estimate - row.ci_low),
        color: "#aeb9c7",
        thickness: 1.5,
      },
      hovertemplate: "%{y}<br>ΔRMSE %{x:.2f}<extra></extra>",
    }],
    layout: {
      height: 250,
      xaxis: { title: "ΔRMSE · S5-Full − CM-v", zeroline: false },
      shapes: [zeroReferenceLine()],
      showlegend: false,
    },
  };
}

function primaryDirectionForestFigure(data: ForecastRobustness): PlotlyFigure | undefined {
  if (!data.direction_difference.length) return undefined;
  const rows = data.direction_difference;
  return {
    data: [{
      type: "scatter",
      mode: "markers",
      x: rows.map((row) => row.estimate_pp),
      y: rows.map((row) => splitLabel(row.split)),
      marker: { size: 10, color: rows.map((row) => row.split === "val" ? "#4d8dff" : "#b8651f") },
      error_x: {
        type: "data",
        symmetric: false,
        array: rows.map((row) => row.ci_high_pp - row.estimate_pp),
        arrayminus: rows.map((row) => row.estimate_pp - row.ci_low_pp),
        color: "#aeb9c7",
        thickness: 1.5,
      },
      hovertemplate: "%{y}<br>ΔDirection %{x:.2f} pp<extra></extra>",
    }],
    layout: {
      height: 250,
      xaxis: { title: "ΔDirection · CM-v − S5-Full (pp)", zeroline: false },
      shapes: [zeroReferenceLine()],
      showlegend: false,
    },
  };
}

function sensitivityRmseForestFigure(data: ForecastRobustness): PlotlyFigure | undefined {
  if (!data.sensitivity.length) return undefined;
  const rows = data.sensitivity;
  return {
    data: [{
      type: "scatter",
      mode: "markers",
      x: rows.map((row) => row.rmse_difference_estimate),
      y: rows.map((row) => `${splitLabel(row.split)} · L=${row.block_length}`),
      marker: { size: 9, color: rows.map((row) => row.split === "val" ? "#4d8dff" : "#b8651f") },
      error_x: {
        type: "data",
        symmetric: false,
        array: rows.map((row) => row.rmse_ci_high - row.rmse_difference_estimate),
        arrayminus: rows.map((row) => row.rmse_difference_estimate - row.rmse_ci_low),
        color: "#aeb9c7",
        thickness: 1.25,
      },
      hovertemplate: "%{y}<br>ΔRMSE %{x:.2f}<extra></extra>",
    }],
    layout: {
      height: 330,
      xaxis: { title: "ΔRMSE · S5-Full − CM-v", zeroline: false },
      shapes: [zeroReferenceLine()],
      showlegend: false,
    },
  };
}

function sensitivityDirectionForestFigure(data: ForecastRobustness): PlotlyFigure | undefined {
  if (!data.sensitivity.length) return undefined;
  const rows = data.sensitivity;
  return {
    data: [{
      type: "scatter",
      mode: "markers",
      x: rows.map((row) => row.direction_difference_estimate_pp),
      y: rows.map((row) => `${splitLabel(row.split)} · L=${row.block_length}`),
      marker: { size: 9, color: rows.map((row) => row.split === "val" ? "#4d8dff" : "#b8651f") },
      error_x: {
        type: "data",
        symmetric: false,
        array: rows.map((row) => row.direction_ci_high_pp - row.direction_difference_estimate_pp),
        arrayminus: rows.map((row) => row.direction_difference_estimate_pp - row.direction_ci_low_pp),
        color: "#aeb9c7",
        thickness: 1.25,
      },
      hovertemplate: "%{y}<br>ΔDirection %{x:.2f} pp<extra></extra>",
    }],
    layout: {
      height: 330,
      xaxis: { title: "ΔDirection · CM-v − S5-Full (pp)", zeroline: false },
      shapes: [zeroReferenceLine()],
      showlegend: false,
    },
  };
}

function chronologicalRmseFigure(data: ForecastRobustness): PlotlyFigure | undefined {
  if (!data.periods.length) return undefined;
  const periods = data.periods.map((row) => row.period);
  return {
    data: [
      { type: "scatter", mode: "markers", name: "CM-v", x: data.periods.map((row) => row.cmamba_v_rmse), y: periods, marker: { size: 10, color: "#4d8dff" } },
      { type: "scatter", mode: "markers", name: "S5-Full", x: data.periods.map((row) => row.s5_full_rmse), y: periods, marker: { size: 10, color: "#31b77a", symbol: "diamond" } },
      { type: "scatter", mode: "markers", name: "Persistence", x: data.periods.map((row) => row.persistence_rmse), y: periods, marker: { size: 9, color: "#8795a8", symbol: "square" } },
    ],
    layout: {
      height: 300,
      xaxis: { title: "RMSE · lower is better" },
      legend: { orientation: "h", y: 1.12, x: 1, xanchor: "right" },
    },
  };
}

function chronologicalDirectionFigure(data: ForecastRobustness): PlotlyFigure | undefined {
  if (!data.periods.length) return undefined;
  const periods = data.periods.map((row) => row.period);
  return {
    data: [
      { type: "scatter", mode: "markers", name: "CM-v", x: data.periods.map((row) => row.cmamba_v_direction_pct), y: periods, marker: { size: 10, color: "#4d8dff" } },
      { type: "scatter", mode: "markers", name: "S5-Full", x: data.periods.map((row) => row.s5_full_direction_pct), y: periods, marker: { size: 10, color: "#31b77a", symbol: "diamond" } },
    ],
    layout: {
      height: 300,
      xaxis: { title: "Directional accuracy (%)" },
      legend: { orientation: "h", y: 1.12, x: 1, xanchor: "right" },
    },
  };
}

function paperParameterRmseFigure(rows: PaperReportedMetricRow[]): PlotlyFigure | undefined {
  if (!rows.length) return undefined;
  return {
    data: [{
      type: "scatter",
      mode: "markers+text",
      x: rows.map((row) => row.parameter_count),
      y: rows.map((row) => row.RMSE),
      text: rows.map((row) => row.display_name),
      textposition: rows.map((row) => row.model_id === "cmamba_v" ? "bottom center" : "top center"),
      marker: {
        size: rows.map((row) => row.model_id === "cmamba_v" ? 16 : 10),
        color: rows.map((row) => row.model_id === "cmamba_v" ? "#4d8dff" : "#8795a8"),
        line: { width: rows.map((row) => row.model_id === "cmamba_v" ? 2 : 0), color: "#edf2f7" },
      },
      hovertemplate: "%{text}<br>Parameters %{x:,}<br>RMSE %{y:.1f}<extra></extra>",
    }],
    layout: {
      height: 360,
      xaxis: { title: "Parameter count", type: "log" },
      yaxis: { title: "Paper-reported RMSE · lower is better" },
      showlegend: false,
    },
  };
}

function DirectionValue({ row }: { row: ControlledForecastMetricRow }) {
  if (!row.directional_coverage_pct || row.directional_accuracy_pct == null) {
    return <span className="text-[var(--text-muted)]">—</span>;
  }
  return <>{pct(row.directional_accuracy_pct)}</>;
}

function ReproductionPanel({
  data,
  controlledSamples,
}: {
  data: Reproduction350d;
  controlledSamples?: number;
}) {
  const reproduced = data.rows.find((row) => row.result_type === "retrained_checkpoint");
  const gaps = reproduced
    ? [reproduced.RMSE_gap_pct, reproduced.MAE_gap_pct, reproduced.MAPE_gap_pct]
    : [];
  const hasCompleteGaps = gaps.length === 3 && gaps.every(Number.isFinite);
  const maxGap = hasCompleteGaps ? Math.max(...gaps) : undefined;
  const threshold = data.criterion?.threshold_pct;
  const withinCriterion = maxGap != null && threshold != null && maxGap <= threshold;

  return (
    <Panel
      eyebrow={<>RQ1 reproduction · original {data.samples ?? "—"}-date test split</>}
      hint="Official and local CM-v checkpoints are evaluated on the original paper test dates. Lower gaps indicate a closer reproduction of the paper-reported aggregate metrics."
      actions={<Pill tone={withinCriterion ? "neutral" : "amber"}>{withinCriterion ? "Within local criterion" : "Check result"}</Pill>}
    >
      {maxGap != null && threshold != null && (
        <Callout tone={withinCriterion ? "neutral" : "amber"} title="Measured reproduction gap">
          {withinCriterion
            ? <>The local checkpoint stays within {fmt(maxGap, 3)}% across RMSE, MAE and MAPE under the {fmt(threshold, 1)}% project reproduction criterion.</>
            : <>The largest metric gap is {fmt(maxGap, 3)}%, above the {fmt(threshold, 1)}% project reproduction criterion.</>} This criterion is a local project check, not a published benchmark.
        </Callout>
      )}
      <div className="mt-4">
        <Plot
          figure={gapToPaperFigure(data)}
          height={300}
          ariaLabel="RQ1 metric gaps to the paper reference"
          accessibleSummary={maxGap == null ? "RQ1 gaps are unavailable." : `The largest local checkpoint metric gap is ${fmt(maxGap, 3)}%.`}
        />
      </div>
      <ExactValues>
        <table className="w-full text-[12px]">
          <thead><tr className="border-b border-[var(--line-subtle)] text-left">
            <th className="eyebrow px-4 py-2 font-normal">Result</th><th className="eyebrow px-4 py-2 text-right font-normal">N</th><th className="eyebrow px-4 py-2 text-right font-normal">RMSE</th><th className="eyebrow px-4 py-2 text-right font-normal">MAE</th><th className="eyebrow px-4 py-2 text-right font-normal">MAPE</th><th className="eyebrow px-4 py-2 text-right font-normal">RMSE gap</th><th className="eyebrow px-4 py-2 text-right font-normal">MAE gap</th><th className="eyebrow px-4 py-2 text-right font-normal">MAPE gap</th>
          </tr></thead>
          <tbody className="tnum">
            {data.rows.map((row) => (
              <tr key={row.result_type} className="border-b border-[var(--line-subtle)] last:border-0">
                <td className="px-4 py-2 font-medium">{row.display_name}</td><td className="px-4 py-2 text-right">{row.samples}</td><td className="px-4 py-2 text-right">{fmt(row.RMSE, 2)}</td><td className="px-4 py-2 text-right">{fmt(row.MAE, 2)}</td><td className="px-4 py-2 text-right">{pct(row.MAPE_pct, 3)}</td><td className="px-4 py-2 text-right">{pct(row.RMSE_gap_pct, 3)}</td><td className="px-4 py-2 text-right">{pct(row.MAE_gap_pct, 3)}</td><td className="px-4 py-2 text-right">{pct(row.MAPE_gap_pct, 3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </ExactValues>
      <p className="mt-3 text-[11px] text-[var(--text-muted)]">
        Separate from the {controlledSamples ?? "—"}-date controlled comparison below; the latter uses the common CM-v / S5-Full target dates required for paired evaluation.
      </p>
    </Panel>
  );
}

function RobustnessPanel({ data }: { data: ForecastRobustness }) {
  const protocol = data.protocol;
  if (!protocol) {
    return <Callout tone="amber">The robustness protocol is unavailable.</Callout>;
  }
  const primaryRmse = data.rmse_difference;
  const primaryDirection = data.direction_difference;
  const expectedSensitivityRows = 2 * protocol.block_lengths.length;
  const sensitivityKeys = new Set(
    data.sensitivity.map((row) => `${row.split}-${row.block_length}`),
  );
  const allIntervalsCrossZero =
    protocol.block_lengths.length > 0 &&
    data.sensitivity.length === expectedSensitivityRows &&
    sensitivityKeys.size === expectedSensitivityRows &&
    data.sensitivity.every(
      (row) => row.rmse_ci_includes_zero && row.direction_ci_includes_zero,
    );

  return (
    <Panel
      eyebrow="Frozen-forecast robustness"
      hint={`${protocol.resamples.toLocaleString("en-US")} frozen-forecast resamples use a paired non-circular moving-block bootstrap. Primary block length: ${protocol.primary_block_length}; evaluated sensitivity lengths: ${protocol.block_lengths.join(", ")}.`}
      actions={<Pill tone={allIntervalsCrossZero ? "neutral" : "amber"}>{allIntervalsCrossZero ? "All intervals cross zero" : "Review intervals"}</Pill>}
    >
      <Callout tone={allIntervalsCrossZero ? "amber" : "red"} title="Bounded statistical conclusion">
        {allIntervalsCrossZero
          ? <>Across block lengths {protocol.block_lengths.join(", ")}, every paired RMSE and directional 95% interval crosses zero. Crossing zero does not prove equivalence.</>
          : <>At least one paired interval does not cross zero; inspect the sensitivity rows before stating a conclusion.</>} This is not retraining, multi-seed, or future-regime evidence.
      </Callout>

      <section className="mt-5">
        <h3 className="text-sm font-semibold text-[var(--text-primary)]">Primary paired differences · L={protocol.primary_block_length}</h3>
        <p className="mt-1 text-[11px] text-[var(--text-muted)]">
          Horizontal bars are paired 95% intervals; the dashed vertical line is zero difference. Negative ΔRMSE favors S5-Full; positive ΔDirection favors CM-v.
        </p>
        <div className="mt-3 grid grid-cols-1 gap-4 lg:grid-cols-2">
          <div className="rounded border border-[var(--line-subtle)] bg-[var(--bg-sunken)] p-2">
            <div className="eyebrow px-2 pt-1">RMSE difference</div>
            <Plot figure={primaryRmseForestFigure(data)} height={250} ariaLabel="Primary paired RMSE difference forest plot" accessibleSummary="Validation and test paired RMSE differences with 95% intervals and a zero reference." leftMargin={82} />
          </div>
          <div className="rounded border border-[var(--line-subtle)] bg-[var(--bg-sunken)] p-2">
            <div className="eyebrow px-2 pt-1">Directional-accuracy difference</div>
            <Plot figure={primaryDirectionForestFigure(data)} height={250} ariaLabel="Primary paired directional difference forest plot" accessibleSummary="Validation and test paired directional-accuracy differences with 95% intervals and a zero reference." leftMargin={82} />
          </div>
        </div>
        <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
          <ExactValues>
            <table className="w-full text-[12px]">
              <thead><tr className="border-b border-[var(--line-subtle)] text-left">
                <th className="eyebrow px-3 py-2 font-normal">Split</th><th className="eyebrow px-3 py-2 text-right font-normal">CM-v RMSE · 95% CI</th><th className="eyebrow px-3 py-2 text-right font-normal">S5-Full RMSE · 95% CI</th><th className="eyebrow px-3 py-2 text-right font-normal">Persistence RMSE · 95% CI</th><th className="eyebrow px-3 py-2 text-right font-normal">ΔRMSE S5−CM-v · 95% CI</th>
              </tr></thead>
              <tbody className="tnum">{(["val", "test"] as const).map((split) => {
                const cmRow = data.rmse.find((row) => row.split === split && row.model_id === "cmamba_v_reproduced");
                const s5Row = data.rmse.find((row) => row.split === split && row.model_id === "s5_full");
                const persistenceRow = data.rmse.find((row) => row.split === split && row.model_id === "naive_persistence");
                const delta = primaryRmse.find((row) => row.split === split);
                return <tr key={split} className="border-b border-[var(--line-subtle)] last:border-0"><td className="px-3 py-2 font-medium">{splitLabel(split)}</td><td className="px-3 py-2 text-right">{estimateWithInterval(cmRow?.estimate, cmRow?.ci_low, cmRow?.ci_high)}</td><td className="px-3 py-2 text-right">{estimateWithInterval(s5Row?.estimate, s5Row?.ci_low, s5Row?.ci_high)}</td><td className="px-3 py-2 text-right">{estimateWithInterval(persistenceRow?.estimate, persistenceRow?.ci_low, persistenceRow?.ci_high)}</td><td className="px-3 py-2 text-right">{estimateWithInterval(delta?.estimate, delta?.ci_low, delta?.ci_high)}</td></tr>;
              })}</tbody>
            </table>
          </ExactValues>

          <ExactValues>
            <table className="w-full text-[12px]">
              <thead><tr className="border-b border-[var(--line-subtle)] text-left">
                <th className="eyebrow px-3 py-2 font-normal">Split</th><th className="eyebrow px-3 py-2 text-right font-normal">CM-v direction · 95% CI</th><th className="eyebrow px-3 py-2 text-right font-normal">S5-Full direction · 95% CI</th><th className="eyebrow px-3 py-2 text-right font-normal">ΔDirection CM-v−S5 · 95% CI</th><th className="eyebrow px-3 py-2 text-right font-normal">Exact McNemar p</th>
              </tr></thead>
              <tbody className="tnum">{(["val", "test"] as const).map((split) => {
                const cmRow = data.direction.find((row) => row.split === split && row.model_id === "cmamba_v_reproduced");
                const s5Row = data.direction.find((row) => row.split === split && row.model_id === "s5_full");
                const delta = primaryDirection.find((row) => row.split === split);
                return <tr key={split} className="border-b border-[var(--line-subtle)] last:border-0"><td className="px-3 py-2 font-medium">{splitLabel(split)}</td><td className="px-3 py-2 text-right">{estimateWithInterval(cmRow?.estimate_pct, cmRow?.ci_low_pct, cmRow?.ci_high_pct, "%")}</td><td className="px-3 py-2 text-right">{estimateWithInterval(s5Row?.estimate_pct, s5Row?.ci_low_pct, s5Row?.ci_high_pct, "%")}</td><td className="px-3 py-2 text-right">{estimateWithInterval(delta?.estimate_pp, delta?.ci_low_pp, delta?.ci_high_pp, " pp")}</td><td className="px-3 py-2 text-right"><Pill tone="neutral">{fmt(delta?.mcnemar_exact_p, 3)}</Pill></td></tr>;
              })}</tbody>
            </table>
          </ExactValues>
        </div>
      </section>

      <section className="mt-6 border-t border-[var(--line-subtle)] pt-5">
        <h3 className="text-sm font-semibold text-[var(--text-primary)]">Block sensitivity · L={protocol.block_lengths.join(", ")}</h3>
        <div className="mt-3 grid grid-cols-1 gap-4 lg:grid-cols-2">
          <div className="rounded border border-[var(--line-subtle)] bg-[var(--bg-sunken)] p-2">
            <div className="eyebrow px-2 pt-1">ΔRMSE sensitivity</div>
            <Plot figure={sensitivityRmseForestFigure(data)} height={330} ariaLabel="RMSE block-sensitivity forest plot" accessibleSummary="Paired RMSE differences for validation and test at block lengths 5, 7 and 14; every interval is assessed against zero." leftMargin={120} />
          </div>
          <div className="rounded border border-[var(--line-subtle)] bg-[var(--bg-sunken)] p-2">
            <div className="eyebrow px-2 pt-1">ΔDirection sensitivity</div>
            <Plot figure={sensitivityDirectionForestFigure(data)} height={330} ariaLabel="Directional block-sensitivity forest plot" accessibleSummary="Paired directional differences for validation and test at block lengths 5, 7 and 14; every interval is assessed against zero." leftMargin={120} />
          </div>
        </div>
        <ExactValues>
            <table className="w-full text-[12px]">
              <thead><tr className="border-b border-[var(--line-subtle)] text-left">
                <th className="eyebrow px-3 py-2 font-normal">Split</th><th className="eyebrow px-3 py-2 text-right font-normal">Block length</th><th className="eyebrow px-3 py-2 text-right font-normal">ΔRMSE S5−CM-v · 95% CI</th><th className="eyebrow px-3 py-2 text-right font-normal">ΔDirection CM-v−S5 · 95% CI</th><th className="eyebrow px-3 py-2 font-normal">Interpretation</th>
              </tr></thead>
              <tbody className="tnum">{data.sensitivity.map((row) => {
                const includesZero = row.rmse_ci_includes_zero && row.direction_ci_includes_zero;
                return <tr key={`${row.split}-${row.block_length}`} className="border-b border-[var(--line-subtle)] last:border-0"><td className="px-3 py-2 font-medium">{splitLabel(row.split)}</td><td className="px-3 py-2 text-right">{row.block_length}</td><td className="px-3 py-2 text-right">{estimateWithInterval(row.rmse_difference_estimate, row.rmse_ci_low, row.rmse_ci_high)}</td><td className="px-3 py-2 text-right">{estimateWithInterval(row.direction_difference_estimate_pp, row.direction_ci_low_pp, row.direction_ci_high_pp, " pp")}</td><td className="px-3 py-2"><Pill tone={includesZero ? "amber" : "red"}>{includesZero ? "Includes zero" : "Excludes zero"}</Pill></td></tr>;
              })}</tbody>
            </table>
        </ExactValues>
      </section>

      <section className="mt-6 border-t border-[var(--line-subtle)] pt-5">
        <h3 className="text-sm font-semibold text-[var(--text-primary)]">Chronological halves</h3>
        <p className="mt-1 text-[11px] text-[var(--text-muted)]">
          Descriptive views of the same checkpoints, not independent holdouts or separate training runs.
        </p>
        <div className="mt-3 grid grid-cols-1 gap-4 lg:grid-cols-2">
          <div className="rounded border border-[var(--line-subtle)] bg-[var(--bg-sunken)] p-2">
            <div className="eyebrow px-2 pt-1">RMSE by half</div>
            <Plot figure={chronologicalRmseFigure(data)} height={300} ariaLabel="Chronological-half RMSE grouped-dot plot" accessibleSummary="CM-v, S5-Full and persistence RMSE for the first and second halves of validation and test." leftMargin={78} />
          </div>
          <div className="rounded border border-[var(--line-subtle)] bg-[var(--bg-sunken)] p-2">
            <div className="eyebrow px-2 pt-1">Direction by half</div>
            <Plot figure={chronologicalDirectionFigure(data)} height={300} ariaLabel="Chronological-half directional accuracy grouped-dot plot" accessibleSummary="CM-v and S5-Full directional accuracy for the first and second halves of validation and test." leftMargin={78} />
          </div>
        </div>
        <ExactValues>
            <table className="w-full text-[12px]">
              <thead><tr className="border-b border-[var(--line-subtle)] text-left">
                <th className="eyebrow px-3 py-2 font-normal">Period</th><th className="eyebrow px-3 py-2 font-normal">Dates</th><th className="eyebrow px-3 py-2 text-right font-normal">N</th><th className="eyebrow px-3 py-2 text-right font-normal">CM-v RMSE</th><th className="eyebrow px-3 py-2 text-right font-normal">S5-Full RMSE</th><th className="eyebrow px-3 py-2 text-right font-normal">Persistence RMSE</th><th className="eyebrow px-3 py-2 text-right font-normal">CM-v direction</th><th className="eyebrow px-3 py-2 text-right font-normal">S5 direction</th>
              </tr></thead>
              <tbody className="tnum">{data.periods.map((row) => (
                <tr key={row.period} className="border-b border-[var(--line-subtle)] last:border-0"><td className="px-3 py-2 font-medium">{row.period}</td><td className="px-3 py-2 text-[var(--text-muted)]">{row.date_from} → {row.date_to}</td><td className="px-3 py-2 text-right">{row.samples}</td><td className="px-3 py-2 text-right">{fmt(row.cmamba_v_rmse, 2)}</td><td className="px-3 py-2 text-right">{fmt(row.s5_full_rmse, 2)}</td><td className="px-3 py-2 text-right">{fmt(row.persistence_rmse, 2)}</td><td className="px-3 py-2 text-right">{pct(row.cmamba_v_direction_pct)}</td><td className="px-3 py-2 text-right">{pct(row.s5_full_direction_pct)}</td></tr>
              ))}</tbody>
            </table>
        </ExactValues>
      </section>
    </Panel>
  );
}

export function ReproduceScreen() {
  const { reproduce } = useConsole();
  const result = reproduce.data;
  const chart = useMemo(() => metricFigure(result?.controlled_local ?? []), [result?.controlled_local]);
  const paperChart = useMemo(() => paperParameterRmseFigure(result?.paper_reported ?? []), [result?.paper_reported]);

  if (reproduce.error) return <ErrorBanner message={reproduce.error} />;
  if (reproduce.loading && !result) {
    return <div className="grid gap-4"><Skeleton className="h-[110px] rounded-lg" /><Skeleton className="h-[430px] rounded-lg" /></div>;
  }

  const ready =
    result?.final_evidence_status === "READY" &&
    result.reproduction_350d.status === "READY" &&
    result.forecast_robustness.status === "READY";
  const test = result?.controlled_local.filter((row) => row.split === "test") ?? [];
  const cm = test.find((row) => row.model_id === "cmamba_v_reproduced");
  const s5 = test.find((row) => row.model_id === "s5_full");
  const naive = test.find((row) => row.model_id === "naive_persistence");

  return (
    <>
      <PageHeader
        title="Evaluation"
        description={result ? `The original ${result.reproduction_350d.samples ?? "—"}-date CM-v reproduction is kept separate from the ${result.controlled_local[0]?.samples ?? "—"}-date controlled CM-v, S5-Full and persistence comparison. Only identical local target dates are used in paired tests.` : "Loading the frozen evaluation scopes and paired evidence."}
      />

      {!result && <Callout tone="neutral">Loading evaluation evidence…</Callout>}
      {result && !ready && (
        <Callout tone="amber" title="Evaluation evidence is not ready">
          {result.final_evidence_error ?? result.errors.join(" ")}
        </Callout>
      )}

      {result && ready && (
        <>
          <ReproductionPanel data={result.reproduction_350d} controlledSamples={cm?.samples} />

          <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
            <StatTile label="Reproduced CM-v · test RMSE" value={fmt(cm?.RMSE, 2)} sub={`${cm?.samples ?? "—"} aligned dates`} tone="blue" />
            <StatTile label="S5-Full · test RMSE" value={fmt(s5?.RMSE, 2)} sub={`${s5?.samples ?? "—"} aligned dates`} tone="brand" />
            <StatTile label="Persistence · test RMSE" value={fmt(naive?.RMSE, 2)} sub={`${naive?.samples ?? "—"} aligned dates · no parameters`} />
          </div>

          <Panel eyebrow="Controlled local comparison" hint="All three rows use identical targets inside each split; lower RMSE, MAE and MAPE are better.">
            <Plot figure={chart} height={350} ariaLabel="Controlled validation and test RMSE comparison" accessibleSummary="Validation and test RMSE for reproduced CM-v, S5-Full and persistence on aligned dates." />
            <ExactValues>
              <table className="w-full text-[12px]">
                <thead><tr className="border-b border-[var(--line-subtle)] text-left">
                  <th className="eyebrow px-4 py-2 font-normal">Split</th><th className="eyebrow px-4 py-2 font-normal">Model</th><th className="eyebrow px-4 py-2 text-right font-normal">N</th><th className="eyebrow px-4 py-2 text-right font-normal">RMSE</th><th className="eyebrow px-4 py-2 text-right font-normal">MAE</th><th className="eyebrow px-4 py-2 text-right font-normal">MAPE</th><th className="eyebrow px-4 py-2 text-right font-normal">Direction</th>
                </tr></thead>
                <tbody className="tnum">
                  {result.controlled_local.map((row) => (
                    <tr key={`${row.split}-${row.model_id}`} className="border-b border-[var(--line-subtle)] last:border-0">
                      <td className="px-4 py-2 uppercase text-[var(--text-muted)]">{row.split}</td><td className="px-4 py-2 font-medium text-[var(--text-primary)]">{row.display_name}</td><td className="px-4 py-2 text-right">{row.samples}</td><td className="px-4 py-2 text-right">{fmt(row.RMSE, 2)}</td><td className="px-4 py-2 text-right">{fmt(row.MAE, 2)}</td><td className="px-4 py-2 text-right">{pct(row.MAPE_pct)}</td><td className="px-4 py-2 text-right"><DirectionValue row={row} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </ExactValues>
            <p className="mt-3 text-[11px] text-[var(--text-muted)]">
              Persistence predicts the current Close, so an upward or downward direction is not defined.
            </p>
          </Panel>

          <Panel eyebrow="Paired statistical tests" hint="DM uses squared error with HAC lag 1. Wilcoxon uses absolute error. p < 0.05 is highlighted, but significance does not imply practical superiority.">
            <p className="text-[12px] leading-relaxed text-[var(--text-secondary)]">
              Pairing is restricted to identical local target dates. The primary uncertainty view below adds time-aware intervals to these point-test diagnostics.
            </p>
            <ExactValues>
              <table className="w-full text-[12px]">
                <thead><tr className="border-b border-[var(--line-subtle)] text-left">
                  <th className="eyebrow px-4 py-2 font-normal">Split</th><th className="eyebrow px-4 py-2 font-normal">Comparison</th><th className="eyebrow px-4 py-2 text-right font-normal">DM p</th><th className="eyebrow px-4 py-2 font-normal">Lower MSE</th><th className="eyebrow px-4 py-2 text-right font-normal">Wilcoxon p</th><th className="eyebrow px-4 py-2 font-normal">Lower MAE</th>
                </tr></thead>
                <tbody className="tnum">
                  {result.paired_tests.map((row) => (
                    <tr key={`${row.split}-${row.comparison}`} className="border-b border-[var(--line-subtle)] last:border-0">
                      <td className="px-4 py-2 uppercase text-[var(--text-muted)]">{row.split}</td><td className="px-4 py-2 text-[var(--text-secondary)]">{prettyModel(row.model_a)} vs {prettyModel(row.model_b)}</td><td className="px-4 py-2 text-right"><Pill tone={row.dm_significant_5pct ? "green" : "neutral"}>{fmt(row.dm_p_value, 3)}</Pill></td><td className="px-4 py-2">{prettyModel(row.lower_mse_model)}</td><td className="px-4 py-2 text-right"><Pill tone={row.wilcoxon_significant_5pct ? "green" : "neutral"}>{fmt(row.wilcoxon_p_value, 3)}</Pill></td><td className="px-4 py-2">{prettyModel(row.lower_mae_model)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </ExactValues>
          </Panel>

          <RobustnessPanel data={result.forecast_robustness} />

          <Panel eyebrow="Paper-reported reference · CryptoMamba-v paper Table 3" hint="These are published aggregate values. They are not mixed into local paired tests because per-date paper errors are unavailable.">
            <div className="eyebrow mb-2">Parameter count vs paper-reported RMSE</div>
            <p className="mb-2 text-[11px] text-[var(--text-muted)]">The parameter axis uses a logarithmic scale so the full model range remains readable.</p>
            <Plot figure={paperChart} height={360} ariaLabel="Paper parameter count versus RMSE scatter" accessibleSummary="Paper-reported parameter count and RMSE for all Table 3 models, with CryptoMamba-v highlighted." leftMargin={64} />
            <ExactValues>
              <table className="w-full text-[12px]">
                <thead><tr className="border-b border-[var(--line-subtle)] text-left"><th className="eyebrow px-4 py-2 font-normal">Model</th><th className="eyebrow px-4 py-2 text-right font-normal">Parameters</th><th className="eyebrow px-4 py-2 text-right font-normal">RMSE</th><th className="eyebrow px-4 py-2 text-right font-normal">MAE</th><th className="eyebrow px-4 py-2 text-right font-normal">MAPE</th><th className="eyebrow px-4 py-2 font-normal">Source</th></tr></thead>
                <tbody className="tnum">{result.paper_reported.map((row) => (
                  <tr key={row.model_id} className="border-b border-[var(--line-subtle)] last:border-0"><td className="px-4 py-2 font-medium">{row.display_name}</td><td className="px-4 py-2 text-right">{row.parameter_count.toLocaleString("en-US")}</td><td className="px-4 py-2 text-right">{fmt(row.RMSE, 1)}</td><td className="px-4 py-2 text-right">{fmt(row.MAE, 1)}</td><td className="px-4 py-2 text-right">{pct(row.MAPE_pct, 3)}</td><td className="px-4 py-2 text-[var(--text-muted)]">{row.source_table}</td></tr>
                ))}</tbody>
              </table>
            </ExactValues>
          </Panel>
        </>
      )}
    </>
  );
}
