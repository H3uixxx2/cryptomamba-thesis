import { useEffect, useState } from "react";
import { ChevronLeft, ChevronRight } from "lucide-react";

import { Plot } from "@/components/Plot";
import { ReplayChart } from "@/components/ReplayChart";
import { Callout, ErrorBanner, Field, Num, PageHeader, Panel, Pill, StatTile, toneForRoi } from "@/components/primitives";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import type { PlotlyFigure } from "@/lib/api";
import { fmt, pct, prettyModel, signedPct, usd } from "@/lib/format";
import { useConsole } from "@/store";

const MODEL_ORDER: Record<string, number> = { cmamba_v_reproduced: 0, s5_full: 1, naive_persistence: 2 };
const STRATEGY_ORDER: Record<string, number> = { vanilla: 0, smart: 1, smart_w_short: 2 };
const FEE_CHART_MODELS = [
  { modelId: "cmamba_v_reproduced", title: "Reproduced CM-v" },
  { modelId: "s5_full", title: "CMamba-T / S5-Full" },
] as const;
const FEE_SERIES_COLOURS: Record<string, string> = {
  buy_hold: "#e0a82e",
  vanilla: "#5b8cff",
  smart: "#37b37e",
  smart_w_short: "#9d72f2",
};

function modelLabel(value: string): string {
  if (value === "cmamba_v_reproduced") return "Reproduced CM-v";
  if (value === "s5_full") return "CMamba-T / S5-Full";
  if (value === "naive_persistence") return "Naive persistence";
  return prettyModel(value);
}

function replayStatusLabel(status: string): string {
  if (status === "VERIFIED") return "Within local replay criterion";
  if (status === "NOT_MATCHED") return "Does not match";
  return status;
}

function FinalEvidence() {
  const { backtest } = useConsole();
  const data = backtest.data;
  if (backtest.error) return <ErrorBanner message={backtest.error} />;
  if (!data) return <Callout tone="neutral">Loading trading results…</Callout>;
  if (data.final_evidence_status !== "READY") {
    return <Callout tone="amber" title="Final trading evidence is not ready">{data.final_evidence_error ?? data.error}</Callout>;
  }

  const testCorrected = data.corrected.filter((row) => row.split === "test");
  const referenceCost = data.corrected_metadata.thesis_reference_cost_pct;
  const referenceRowsWithRepeatedBuyHold = testCorrected.filter((row) => row.transaction_cost_pct === referenceCost);
  const firstBuyHoldIndex = referenceRowsWithRepeatedBuyHold.findIndex((row) => row.strategy === "buy_hold");
  const corrected = referenceRowsWithRepeatedBuyHold.filter(
    (row, index) => row.strategy !== "buy_hold" || index === firstBuyHoldIndex,
  );
  const paper = data.paper_replay.filter((row) => row.split === "test");
  const buyHold = corrected.find((row) => row.strategy === "buy_hold");
  const s5Smart = corrected.find((row) => row.model_id === "s5_full" && row.strategy === "smart");
  const cmSmart = corrected.find((row) => row.model_id === "cmamba_v_reproduced" && row.strategy === "smart");
  const forecastSeries = Array.from(
    new Map(testCorrected.filter((row) => row.strategy !== "buy_hold").map((row) => [
      `${row.model_id}:${row.strategy}`,
      { modelId: row.model_id, strategy: row.strategy },
    ])).values(),
  ).sort((left, right) =>
    (MODEL_ORDER[left.modelId] ?? 99) - (MODEL_ORDER[right.modelId] ?? 99) ||
    (STRATEGY_ORDER[left.strategy] ?? 99) - (STRATEGY_ORDER[right.strategy] ?? 99));
  const metricAtCost = (modelId: string | null, strategy: string, cost: number) => testCorrected.find(
    (row) => row.strategy === strategy && row.transaction_cost_pct === cost && (modelId === null || row.model_id === modelId),
  );
  const declaredFeeScenarios = Array.from(new Set(
    (data.corrected_metadata.transaction_cost_pct_scenarios ?? []).filter(Number.isFinite),
  )).sort((left, right) => left - right);
  const isCompleteFeeScenario = (cost: number) => {
    const buyHoldRowsAtCost = testCorrected.filter(
      (row) => row.strategy === "buy_hold" && row.transaction_cost_pct === cost,
    );
    const forecastRowsAtCost = testCorrected.filter(
      (row) => row.strategy !== "buy_hold" && row.transaction_cost_pct === cost,
    );
    return Boolean(forecastSeries.length && buyHoldRowsAtCost.length &&
      buyHoldRowsAtCost.every((row) => Number.isFinite(row.final_equity) &&
        row.final_equity === buyHoldRowsAtCost[0].final_equity) &&
      forecastRowsAtCost.length === forecastSeries.length &&
      forecastRowsAtCost.every((row) => Number.isFinite(row.final_equity)) &&
      forecastSeries.every((series) => forecastRowsAtCost.some(
        (row) => row.model_id === series.modelId && row.strategy === series.strategy,
      )));
  };
  const feeScenarios = declaredFeeScenarios.filter(isCompleteFeeScenario);
  const hasCompleteFeeMatrix = declaredFeeScenarios.length > 0 && feeScenarios.length === declaredFeeScenarios.length;
  const feeScenarioLabels = feeScenarios.map((cost) => `${cost}%`);
  const referenceCostLabel = referenceCost == null ? "—" : `${referenceCost}%`;
  const buyHoldBeatsEveryForecastStrategy = hasCompleteFeeMatrix && feeScenarios.every((cost) => {
    const buyHoldAtCost = metricAtCost(null, "buy_hold", cost);
    const forecastRowsAtCost = testCorrected.filter(
      (row) => row.strategy !== "buy_hold" && row.transaction_cost_pct === cost,
    );
    return Boolean(buyHoldAtCost && forecastRowsAtCost.length === forecastSeries.length &&
      forecastRowsAtCost.every((row) => buyHoldAtCost.final_equity > row.final_equity));
  });
  const modelForecastSeries = (modelId: string) => forecastSeries.filter((series) => series.modelId === modelId);
  const feeSensitivityFigure = (modelId: string): PlotlyFigure | undefined => {
    if (!feeScenarios.length) return undefined;
    const strategyTraces = modelForecastSeries(modelId).map((series) => ({
      type: "scatter",
      mode: "lines+markers",
      name: prettyModel(series.strategy),
      x: feeScenarios,
      y: feeScenarios.map((cost) => metricAtCost(modelId, series.strategy, cost)?.final_equity),
      line: { color: FEE_SERIES_COLOURS[series.strategy], width: 2.25 },
      marker: { color: FEE_SERIES_COLOURS[series.strategy], size: 6 },
      hovertemplate: "Cost: %{x}%<br>Final equity: %{y:.2f}<extra>%{fullData.name}</extra>",
    }));
    return {
      data: [
        {
          type: "scatter",
          mode: "lines+markers",
          name: "Buy & Hold (shared)",
          x: feeScenarios,
          y: feeScenarios.map((cost) => metricAtCost(null, "buy_hold", cost)?.final_equity),
          line: { color: FEE_SERIES_COLOURS.buy_hold, width: 4 },
          marker: { color: FEE_SERIES_COLOURS.buy_hold, size: 8 },
          hovertemplate: "Cost: %{x}%<br>Final equity: %{y:.2f}<extra>%{fullData.name}</extra>",
        },
        ...strategyTraces,
      ],
      layout: {
        height: 280,
        showlegend: false,
        hovermode: "x unified",
        xaxis: { title: "Transaction cost", tickmode: "array", tickvals: feeScenarios, ticktext: feeScenarioLabels },
        yaxis: { title: "Final equity" },
      },
    };
  };
  const feeSensitivitySummary = (modelId: string) => {
    const describe = (label: string, strategy: string, shared = false) =>
      `${label}: ${feeScenarios.map((cost) =>
        `${cost}% cost ${fmt(metricAtCost(shared ? null : modelId, strategy, cost)?.final_equity, 2)}`).join(", ")}`;
    return [
      describe("Buy & Hold shared benchmark", "buy_hold", true),
      ...modelForecastSeries(modelId).map((series) => describe(prettyModel(series.strategy), series.strategy)),
    ].join(". ");
  };

  return (
    <div className="flex flex-col gap-4">
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <StatTile label="Reference cost" value={referenceCostLabel} sub="applied on traded notional" tone="blue" />
        <StatTile label="Buy & hold · test" value={fmt(buyHold?.final_equity, 2)} sub={`shared market benchmark · initial equity ${fmt(buyHold?.initial_equity, 0)}`} />
        <StatTile label="CM-v Smart · test" value={fmt(cmSmart?.final_equity, 2)} sub={`${cmSmart?.number_of_trades ?? "—"} trades`} tone="brand" />
        <StatTile label="S5-Full Smart · test" value={fmt(s5Smart?.final_equity, 2)} sub={`${s5Smart?.number_of_trades ?? "—"} trades`} tone="green" />
      </div>

      <Tabs defaultValue="corrected" className="w-full">
        <TabsList className="bg-[var(--bg-subtle)]">
          <TabsTrigger value="corrected">Corrected self-financing</TabsTrigger>
          <TabsTrigger value="paper">Paper replay</TabsTrigger>
        </TabsList>
        <TabsContent value="corrected" className="mt-4">
          <div className="flex flex-col gap-4">
          <Panel
            eyebrow="Fee sensitivity · corrected self-financing · test"
            hint="Final equity is shown across the tested transaction costs. Buy & Hold is a model-independent market benchmark, so it is displayed once rather than repeated for each forecast model."
            actions={<Pill tone="blue">{feeScenarioLabels.join(" · ") || "No complete cost rows"}</Pill>}
          >
            {buyHoldBeatsEveryForecastStrategy && (
              <Callout tone="blue" title="Result across all tested transaction costs">
                Buy & Hold finishes above every forecast-driven model-strategy row at all tested costs: {feeScenarioLabels.join(", ")}.
              </Callout>
            )}
            <div className="mt-4 grid min-w-0 gap-4 lg:grid-cols-2">
              {FEE_CHART_MODELS.map(({ modelId, title }) => {
                const strategies = modelForecastSeries(modelId);
                return (
                  <section key={modelId} className="min-w-0 overflow-hidden rounded-md border border-[var(--line-subtle)] bg-[var(--bg-sunken)] p-3">
                    <h3 className="text-sm font-semibold text-[var(--text-primary)]">{title}</h3>
                    <p className="mt-1 text-[11px] text-[var(--text-muted)]">Final equity by transaction cost · shared benchmark highlighted</p>
                    <div className="mt-3 flex flex-wrap gap-x-4 gap-y-2 text-[10px] text-[var(--text-secondary)]" aria-hidden="true">
                      {[{ strategy: "buy_hold", label: "Buy & Hold (shared)" }, ...strategies.map((series) => ({ strategy: series.strategy, label: prettyModel(series.strategy) }))].map((series) => (
                        <span key={series.strategy} className="inline-flex items-center gap-1.5">
                          <span className="h-2 w-2 rounded-full" style={{ backgroundColor: FEE_SERIES_COLOURS[series.strategy] }} />
                          {series.label}
                        </span>
                      ))}
                    </div>
                    <Plot
                      figure={feeSensitivityFigure(modelId)}
                      height={280}
                      emptyLabel="No complete fee scenarios."
                      ariaLabel={`${title} corrected fee sensitivity`}
                      accessibleSummary={feeSensitivitySummary(modelId)}
                    />
                  </section>
                );
              })}
            </div>
            <details className="mt-4 rounded-md border border-[var(--line-subtle)] bg-[var(--bg-sunken)]">
              <summary className="cursor-pointer px-4 py-3 text-xs font-medium text-[var(--text-primary)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--signal-blue)]">
                View exact values
              </summary>
              <div className="overflow-x-auto border-t border-[var(--line-subtle)]">
                <table className="w-full min-w-[680px] text-[12px]">
                  <caption className="sr-only">Exact corrected final equity by model, strategy and transaction cost</caption>
                  <thead><tr className="border-b border-[var(--line-subtle)] text-left"><th className="eyebrow px-4 py-2 font-normal">Model / benchmark</th><th className="eyebrow px-4 py-2 font-normal">Strategy</th>{feeScenarios.map((cost) => <th key={cost} className="eyebrow px-4 py-2 text-right font-normal">Final equity · {cost}%{cost === referenceCost ? " · thesis" : ""}</th>)}</tr></thead>
                  <tbody className="tnum">
                    <tr className="border-b border-[var(--line-subtle)] bg-[var(--bg-subtle)]"><td className="px-4 py-2 font-medium">Shared market benchmark</td><td className="px-4 py-2 text-[var(--text-secondary)]">Buy & Hold</td>{feeScenarios.map((cost) => <td key={cost} className="px-4 py-2 text-right font-semibold">{fmt(metricAtCost(null, "buy_hold", cost)?.final_equity, 2)}</td>)}</tr>
                    {forecastSeries.map((series) => (
                      <tr key={`${series.modelId}-${series.strategy}`} className="border-b border-[var(--line-subtle)] last:border-0"><td className="px-4 py-2 font-medium">{modelLabel(series.modelId)}</td><td className="px-4 py-2 text-[var(--text-secondary)]">{prettyModel(series.strategy)}</td>{feeScenarios.map((cost) => <td key={cost} className="px-4 py-2 text-right">{fmt(metricAtCost(series.modelId, series.strategy, cost)?.final_equity, 2)}</td>)}</tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </details>
          </Panel>
          <Panel
            eyebrow={`Corrected self-financing · test · thesis reference cost ${referenceCostLabel}`}
            hint="Cash, position and equity reconcile at every step. Fees are charged on traded notional; target close is used only for mark-to-market and outcome metrics."
            actions={<Pill tone="green">Primary trading evidence</Pill>}
          >
            <details className="rounded-md border border-[var(--line-subtle)] bg-[var(--bg-sunken)]">
              <summary className="cursor-pointer px-4 py-3 text-xs font-medium text-[var(--text-primary)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--signal-blue)]">
                View reference-cost metrics
              </summary>
              <div className="overflow-x-auto border-t border-[var(--line-subtle)]">
                <table className="w-full min-w-[1050px] text-[12px]">
                <thead><tr className="border-b border-[var(--line-subtle)] text-left"><th className="eyebrow px-4 py-2 font-normal">Model / benchmark</th><th className="eyebrow px-4 py-2 font-normal">Strategy</th><th className="eyebrow px-4 py-2 text-right font-normal">Final equity</th><th className="eyebrow px-4 py-2 text-right font-normal">ROI</th><th className="eyebrow px-4 py-2 text-right font-normal">MDD</th><th className="eyebrow px-4 py-2 text-right font-normal">Sharpe</th><th className="eyebrow px-4 py-2 text-right font-normal">Trades</th><th className="eyebrow px-4 py-2 text-right font-normal">Turnover</th><th className="eyebrow px-4 py-2 text-right font-normal">Fees</th><th className="eyebrow px-4 py-2 text-right font-normal">Reconciliation</th></tr></thead>
                <tbody className="tnum">{corrected.map((row) => (
                  <tr key={`${row.model_id}-${row.strategy}`} className="border-b border-[var(--line-subtle)] last:border-0"><td className="px-4 py-2 font-medium">{row.strategy === "buy_hold" ? "Shared market benchmark" : modelLabel(row.model_id)}</td><td className="px-4 py-2 text-[var(--text-secondary)]">{prettyModel(row.strategy)}</td><td className="px-4 py-2 text-right">{fmt(row.final_equity, 2)}</td><td className="px-4 py-2 text-right"><Num color={toneForRoi(row.ROI_pct)}>{signedPct(row.ROI_pct)}</Num></td><td className="px-4 py-2 text-right">{pct(row.max_drawdown_pct)}</td><td className="px-4 py-2 text-right">{fmt(row.Sharpe, 2)}</td><td className="px-4 py-2 text-right">{row.number_of_trades}</td><td className="px-4 py-2 text-right">{fmt(row.turnover_notional, 2)}</td><td className="px-4 py-2 text-right">{fmt(row.total_fees, 3)}</td><td className="px-4 py-2 text-right">{row.reconciliation_error.toExponential(1)}</td></tr>
                ))}</tbody>
                </table>
              </div>
            </details>
          </Panel>
          </div>
        </TabsContent>
        <TabsContent value="paper" className="mt-4">
          <Panel
            eyebrow="Paper engine replay · test split · zero transaction cost"
            hint="This panel exists only to audit the paper's published trading claim. It is not mixed with the corrected self-financing results."
            actions={<Pill tone="amber">Reproduction audit</Pill>}
          >
            <div className="-mx-4 overflow-x-auto">
              <table className="w-full min-w-[850px] text-[12px]">
                <thead><tr className="border-b border-[var(--line-subtle)] text-left"><th className="eyebrow px-4 py-2 font-normal">Checkpoint</th><th className="eyebrow px-4 py-2 font-normal">Strategy</th><th className="eyebrow px-4 py-2 text-right font-normal">Published balance</th><th className="eyebrow px-4 py-2 text-right font-normal">Replayed balance</th><th className="eyebrow px-4 py-2 text-right font-normal">Balance gap</th><th className="eyebrow px-4 py-2 text-right font-normal">MDD</th><th className="eyebrow px-4 py-2 font-normal">Status</th></tr></thead>
                <tbody className="tnum">{paper.map((row) => (
                  <tr key={`${row.result_type}-${row.trade_mode}`} className="border-b border-[var(--line-subtle)] last:border-0"><td className="px-4 py-2">{/official/i.test(row.result_type) ? "Official paper checkpoint" : "Reproduced checkpoint"}</td><td className="px-4 py-2 text-[var(--text-secondary)]">{prettyModel(row.trade_mode)}</td><td className="px-4 py-2 text-right">{fmt(row.paper_final_balance, 2)}</td><td className="px-4 py-2 text-right">{fmt(row.final_balance, 2)}</td><td className="px-4 py-2 text-right">{signedPct(row.balance_gap_pct)}</td><td className="px-4 py-2 text-right">{pct(row.max_drawdown_pct)}</td><td className="px-4 py-2"><Pill tone={row.status === "VERIFIED" ? "green" : "amber"}>{replayStatusLabel(row.status)}</Pill></td></tr>
                ))}</tbody>
              </table>
            </div>
          </Panel>
        </TabsContent>
      </Tabs>

      <Panel eyebrow="Engine and protocol">
        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
          <Field label="Engine">{data.corrected_metadata.engine ?? "—"}</Field>
          <Field label="Execution / mark">{data.corrected_metadata.execution_basis ?? "—"} / {data.corrected_metadata.mark_basis ?? "—"}</Field>
          <Field label="Cost scenarios">{data.corrected_metadata.transaction_cost_pct_scenarios?.join("% · ") ?? "—"}%</Field>
        </div>
      </Panel>
    </div>
  );
}

function HistoricalReplay() {
  const { replay, backtest } = useConsole();
  const data = replay.data;
  const [cursor, setCursor] = useState(0);

  useEffect(() => setCursor(0), [data]);
  if (replay.error) return <ErrorBanner message={replay.error} />;
  if (!data) return <Callout tone="neutral">Loading historical replay rows…</Callout>;
  if (data.status !== "READY" || !data.rows.length) return <Callout tone="amber" title="Replay not ready">{data.error ?? "No replay rows."}</Callout>;

  const row = data.rows[Math.min(cursor, data.rows.length - 1)];
  const directionCorrect = (row.predicted_close > row.current_close) === (row.target_close > row.current_close);
  const select = (value: number) => setCursor(Math.max(0, Math.min(data.rows.length - 1, value)));

  return (
    <div className="flex flex-col gap-4">
      <Panel eyebrow="Historical decision replay" hint="Interactive presentation of the persisted paper-engine timeline. It does not recompute orders or replace corrected trading evidence." actions={<Pill tone="blue">Exploratory replay</Pill>}>
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          <div className="flex flex-col gap-2"><Label>Checkpoint</Label><Select value={backtest.resultType} onValueChange={backtest.setResultType}><SelectTrigger aria-label="Replay checkpoint"><SelectValue /></SelectTrigger><SelectContent>{data.available.result_types.map((value) => <SelectItem key={value} value={value}>{value === "retrained_checkpoint" ? "Reproduced" : "Official paper"}</SelectItem>)}</SelectContent></Select></div>
          <div className="flex flex-col gap-2"><Label>Split</Label><Select value={backtest.split} onValueChange={backtest.setSplit}><SelectTrigger aria-label="Replay split"><SelectValue /></SelectTrigger><SelectContent>{data.available.splits.map((value) => <SelectItem key={value} value={value}>{value}</SelectItem>)}</SelectContent></Select></div>
          <div className="flex flex-col gap-2"><Label>Strategy</Label><Select value={replay.strategy} onValueChange={replay.setStrategy}><SelectTrigger aria-label="Replay strategy"><SelectValue /></SelectTrigger><SelectContent>{data.available.strategies.map((value) => <SelectItem key={value} value={value}>{prettyModel(value)}</SelectItem>)}</SelectContent></Select></div>
          <div className="flex flex-col gap-2"><Label>Transaction cost</Label><Select value={String(backtest.cost)} onValueChange={(value) => backtest.setCost(Number(value))}><SelectTrigger aria-label="Replay transaction cost"><SelectValue /></SelectTrigger><SelectContent>{data.available.costs.map((value) => <SelectItem key={value} value={String(value)}>{value}%</SelectItem>)}</SelectContent></Select></div>
        </div>
        <div className="mt-5 grid items-center gap-3 md:grid-cols-[auto_minmax(200px,1fr)_auto]">
          <Button variant="outline" size="icon" aria-label="Previous replay day" disabled={cursor === 0} onClick={() => select(cursor - 1)}><ChevronLeft className="h-4 w-4" /></Button>
          <div><Slider value={[cursor]} min={0} max={data.rows.length - 1} step={1} onValueChange={([value]) => select(value)} aria-label="Replay date" /><div className="mt-2 flex justify-between font-mono text-[10px] text-[var(--text-muted)]"><span>{data.rows[0].decision_date}</span><span>{cursor + 1}/{data.rows.length} · {row.decision_date} → {row.outcome_date}</span><span>{data.rows.at(-1)?.outcome_date}</span></div></div>
          <Button variant="outline" size="icon" aria-label="Next replay day" disabled={cursor === data.rows.length - 1} onClick={() => select(cursor + 1)}><ChevronRight className="h-4 w-4" /></Button>
        </div>
      </Panel>

      <div className="grid items-start gap-4 xl:grid-cols-[minmax(0,1fr)_340px]">
        <Panel eyebrow="BTC/USD · forecast · historical actions" bodyClassName="p-0"><ReplayChart candles={data.candles} rows={data.rows} cursor={cursor} phase="outcome" onSelectStep={select} /></Panel>
        <Panel eyebrow={`${row.decision_date} → ${row.outcome_date}`}>
          <div className="mb-4 flex gap-2"><Pill tone={row.action === "BUY" ? "green" : row.action === "SELL" ? "red" : "amber"}>{row.action}</Pill><Pill tone={directionCorrect ? "green" : "red"}>{directionCorrect ? "Correct direction" : "Wrong direction"}</Pill></div>
          <dl className="grid grid-cols-2 gap-4 text-xs"><div><dt className="eyebrow">Decision close</dt><dd className="tnum mt-1">{usd(row.current_close)}</dd></div><div><dt className="eyebrow">Forecast T+1</dt><dd className="tnum mt-1 text-[var(--signal-blue)]">{usd(row.predicted_close)}</dd></div><div><dt className="eyebrow">Actual T+1</dt><dd className="tnum mt-1">{usd(row.target_close)}</dd></div><div><dt className="eyebrow">Forecast return</dt><dd className="tnum mt-1">{signedPct(row.predicted_return_pct)}</dd></div><div><dt className="eyebrow">Portfolio</dt><dd className="tnum mt-1">{usd(row.portfolio_value)}</dd></div><div><dt className="eyebrow">Drawdown</dt><dd className="tnum mt-1">{pct(Math.abs(row.drawdown_pct))}</dd></div></dl>
        </Panel>
      </div>
    </div>
  );
}

function OneDayDemo() {
  const { trading } = useConsole();
  if (trading.error) return <ErrorBanner message={trading.error} />;
  if (!trading.basis) return <Callout tone="neutral">Run or select a prediction before opening the one-day scenario.</Callout>;
  return (
    <div className="flex flex-col gap-4">
      <Panel eyebrow="One-day scenario" hint={`Prediction source: ${trading.basis.source}. This is an explanatory what-if, not historical performance evidence.`}>
        <div className="grid items-end gap-4 md:grid-cols-2 xl:grid-cols-4"><div className="flex flex-col gap-2"><Label>Cash (USD)</Label><Input type="number" value={trading.capital} onChange={(event) => trading.setCapital(Number(event.target.value))} /></div><div className="flex flex-col gap-2"><Label>BTC held</Label><Input type="number" step="0.01" value={trading.btc} onChange={(event) => trading.setBtc(Number(event.target.value))} /></div><div className="flex flex-col gap-2"><Label>Realised next-day move · {signedPct(trading.move)}</Label><Slider value={[trading.move]} min={-15} max={15} step={0.25} onValueChange={([value]) => trading.setMove(value)} onValueCommit={([value]) => trading.commitMove(value)} /></div><Field label="Forecast">{usd(trading.basis.current)} → {usd(trading.basis.predicted)}</Field></div>
      </Panel>
      <Panel eyebrow="Strategy response" bodyClassName="p-0"><Table><TableHeader><TableRow><TableHead className="pl-5">Strategy</TableHead><TableHead>Action</TableHead><TableHead>Size</TableHead><TableHead className="text-right">End value</TableHead><TableHead className="pr-5 text-right">ROI</TableHead></TableRow></TableHeader><TableBody>{(trading.sim?.rows ?? []).map((row) => <TableRow key={row.strategy}><TableCell className="pl-5 font-medium">{row.strategy}</TableCell><TableCell>{row.action}</TableCell><TableCell className="font-mono text-xs">{row.trade_size}</TableCell><TableCell className="text-right"><Num>{usd(row.end_value)}</Num></TableCell><TableCell className="pr-5 text-right"><Num color={toneForRoi(row.roi_pct)}>{signedPct(row.roi_pct)}</Num></TableCell></TableRow>)}</TableBody></Table></Panel>
    </div>
  );
}

export function TradingScreen() {
  return (
    <>
      <PageHeader title="Trading" description="Corrected self-financing results are primary. The paper-engine replay and one-day scenario remain separate explanatory surfaces." />
      <Tabs defaultValue="evidence" className="w-full">
        <TabsList className="w-full max-w-full justify-start overflow-x-auto bg-[var(--bg-subtle)]"><TabsTrigger value="evidence">Evidence</TabsTrigger><TabsTrigger value="replay">Historical Replay</TabsTrigger><TabsTrigger value="demo">One-Day What-If</TabsTrigger></TabsList>
        <TabsContent value="evidence" className="mt-4"><FinalEvidence /></TabsContent>
        <TabsContent value="replay" className="mt-4"><HistoricalReplay /></TabsContent>
        <TabsContent value="demo" className="mt-4"><OneDayDemo /></TabsContent>
      </Tabs>
    </>
  );
}
