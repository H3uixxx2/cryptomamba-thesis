import { useMemo, type ReactNode } from "react";

import { Plot } from "@/components/Plot";
import { Callout, ErrorBanner, PageHeader, Panel, Pill, StatTile } from "@/components/primitives";
import { Skeleton } from "@/components/ui/skeleton";
import type {
  ControlledForecastMetricRow,
  ForecastRobustness,
  PlotlyFigure,
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
      xaxis: { title: "ΔRMSE · CM-T − CM-v", zeroline: false },
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
      xaxis: { title: "ΔDirection · CM-v − CM-T (pp)", zeroline: false },
      shapes: [zeroReferenceLine()],
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

function RobustnessPanel({ data }: { data: ForecastRobustness }) {
  const protocol = data.protocol;
  if (!protocol) {
    return <Callout tone="amber">The robustness protocol is unavailable.</Callout>;
  }
  const primaryRmse = data.rmse_difference;
  const primaryDirection = data.direction_difference;
  const allIntervalsCrossZero =
    primaryRmse.length > 0 &&
    primaryDirection.length > 0 &&
    primaryRmse.every((row) => row.ci_low <= 0 && row.ci_high >= 0) &&
    primaryDirection.every((row) => row.ci_low_pp <= 0 && row.ci_high_pp >= 0);

  return (
    <Panel
      eyebrow="Frozen-forecast robustness"
      hint={`${protocol.resamples.toLocaleString("en-US")} frozen-forecast resamples use a paired non-circular moving-block bootstrap. Primary block length: ${protocol.primary_block_length}.`}
      actions={<Pill tone={allIntervalsCrossZero ? "neutral" : "amber"}>{allIntervalsCrossZero ? "All intervals cross zero" : "Review intervals"}</Pill>}
    >
      <Callout tone={allIntervalsCrossZero ? "amber" : "red"} title="Bounded statistical conclusion">
        {allIntervalsCrossZero
          ? <>At block length {protocol.primary_block_length}, every paired RMSE and directional 95% interval crosses zero. Crossing zero does not prove equivalence.</>
          : <>At least one paired interval does not cross zero; inspect the values below before stating a conclusion.</>} This is not retraining, multi-seed, or future-regime evidence.
      </Callout>

      <section className="mt-5">
        <p className="text-[11px] text-[var(--text-muted)]">
          Horizontal bars are paired 95% intervals; the dashed vertical line is zero difference. Negative ΔRMSE favors CryptoMamba-T; positive ΔDirection favors CM-v.
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
                <th className="eyebrow px-3 py-2 font-normal">Split</th><th className="eyebrow px-3 py-2 text-right font-normal">CM-v RMSE · 95% CI</th><th className="eyebrow px-3 py-2 text-right font-normal">CryptoMamba-T RMSE · 95% CI</th><th className="eyebrow px-3 py-2 text-right font-normal">Persistence RMSE · 95% CI</th><th className="eyebrow px-3 py-2 text-right font-normal">ΔRMSE CM-T−CM-v · 95% CI</th>
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
                <th className="eyebrow px-3 py-2 font-normal">Split</th><th className="eyebrow px-3 py-2 text-right font-normal">CM-v direction · 95% CI</th><th className="eyebrow px-3 py-2 text-right font-normal">CryptoMamba-T direction · 95% CI</th><th className="eyebrow px-3 py-2 text-right font-normal">ΔDirection CM-v−CM-T · 95% CI</th><th className="eyebrow px-3 py-2 text-right font-normal">Exact McNemar p</th>
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
    </Panel>
  );
}

export function ReproduceScreen() {
  const { reproduce } = useConsole();
  const result = reproduce.data;
  const chart = useMemo(() => metricFigure(result?.controlled_local ?? []), [result?.controlled_local]);

  if (reproduce.error) return <ErrorBanner message={reproduce.error} />;
  if (reproduce.loading && !result) {
    return <div className="grid gap-4"><Skeleton className="h-[110px] rounded-lg" /><Skeleton className="h-[430px] rounded-lg" /></div>;
  }

  const ready =
    result?.final_evidence_status === "READY" &&
    result.forecast_robustness.status === "READY";
  const test = result?.controlled_local.filter((row) => row.split === "test") ?? [];
  const cm = test.find((row) => row.model_id === "cmamba_v_reproduced");
  const s5 = test.find((row) => row.model_id === "s5_full");
  const naive = test.find((row) => row.model_id === "naive_persistence");

  return (
    <>
      <PageHeader
        title="Evaluation"
        description={result ? `Controlled comparison of CM-v, CryptoMamba-T and persistence on ${result.controlled_local[0]?.samples ?? "—"} identical local target dates.` : "Loading the frozen evaluation scopes and paired evidence."}
      />

      {!result && <Callout tone="neutral">Loading evaluation evidence…</Callout>}
      {result && !ready && (
        <Callout tone="amber" title="Evaluation evidence is not ready">
          {result.final_evidence_error ?? result.errors.join(" ")}
        </Callout>
      )}

      {result && ready && (
        <>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
            <StatTile label="Reproduced CM-v · test RMSE" value={fmt(cm?.RMSE, 2)} sub={`${cm?.samples ?? "—"} aligned dates`} tone="blue" />
            <StatTile label="CryptoMamba-T · test RMSE" value={fmt(s5?.RMSE, 2)} sub={`${s5?.samples ?? "—"} aligned dates`} tone="brand" />
            <StatTile label="Persistence · test RMSE" value={fmt(naive?.RMSE, 2)} sub={`${naive?.samples ?? "—"} aligned dates · no parameters`} />
          </div>

          <Panel eyebrow="Controlled local comparison" hint="All three rows use identical targets inside each split; lower RMSE, MAE and MAPE are better.">
            <Plot figure={chart} height={350} ariaLabel="Controlled validation and test RMSE comparison" accessibleSummary="Validation and test RMSE for reproduced CM-v, CryptoMamba-T and persistence on aligned dates." />
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
        </>
      )}
    </>
  );
}
