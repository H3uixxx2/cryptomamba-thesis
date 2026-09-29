import { useMemo, useState, type ReactNode } from "react";

import { Plot } from "@/components/Plot";
import { Callout, Panel, Pill, type Tone } from "@/components/primitives";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { fetchRuns, type AblationComparison, type AblationModel, type AblationPtRow, type AblationTradingRow, type FamilyComparisonRow, type PaperReplayRunRow, type PlotlyFigure, type RunMetricRow, type RunTradingRow, type RunsResponse } from "@/lib/api";
import { fmt, pct, prettyModel } from "@/lib/format";
import { useApiOnce } from "@/lib/useApiOnce";

/* Per-seed and ablation evidence. Everything shown is read from the checksum-verified `runs/`
 * part of the evidence bundle by GET /api/runs; nothing is computed here except display layout. */

const HEAD = "eyebrow px-3 py-2 font-normal";
const HEAD_R = `${HEAD} text-right`;
const MODEL_ORDER = ["cmamba_v_reproduced", "s5_full", "naive_persistence"];
const SPLIT_LABEL: Record<string, string> = { val: "Validation", test: "Test" };

function modelLabel(id: string): string {
  if (id === "naive_persistence" || id === "naive") return "Persistence";
  if (id === "cmamba_v_reproduced" || id === "cmv") return "CryptoMamba-v";
  if (id === "s5_full" || id === "cmt") return "CryptoMamba-T";
  return prettyModel(id);
}

function interval(lo: number | null | undefined, hi: number | null | undefined, digits = 2): string {
  if (lo == null || hi == null) return "—";
  return `[${fmt(lo, digits)}; ${fmt(hi, digits)}]`;
}

function VerdictPill({ verdict }: { verdict: string }) {
  const tone: Tone = verdict.includes("same direction") ? "amber" : verdict.includes("not established") ? "neutral" : "green";
  return <Pill tone={tone}>{verdict.replace(/^\(descriptive\) /, "")}</Pill>;
}

function Details({ summary, open, children }: { summary: string; open?: boolean; children: ReactNode }) {
  return (
    <details open={open} className="mt-4 rounded border border-[var(--line-subtle)] bg-[var(--bg-sunken)]">
      <summary className="cursor-pointer px-3 py-2 font-mono text-[11px] font-medium text-[var(--text-secondary)]">{summary}</summary>
      <div className="overflow-x-auto border-t border-[var(--line-subtle)] p-2">{children}</div>
    </details>
  );
}

function useRuns() {
  return useApiOnce<RunsResponse>(fetchRuns, "runs");
}

function RunsState({ children }: { children: (runs: RunsResponse) => ReactNode }) {
  const { data, error, loading } = useRuns();
  if (loading) return <Callout tone="neutral">Loading the training-run evidence…</Callout>;
  if (error) return <Callout tone="amber" title="Training-run evidence is not available">{error}</Callout>;
  if (!data || data.status !== "READY") {
    return <Callout tone="amber" title="Training-run evidence is not ready">{data?.message ?? "The evidence bundle did not verify."}</Callout>;
  }
  return <>{children(data)}</>;
}

/* ------------------------------ Seeds panel ------------------------------ */

function seedFigure(rows: RunMetricRow[], split: string): PlotlyFigure | undefined {
  const learned = ["cmamba_v_reproduced", "s5_full"];
  const members = rows.filter((r) => r.split === split && r.row_kind === "member" && learned.includes(r.model_id));
  const naive = rows.find((r) => r.split === split && r.row_kind === "member" && r.model_id === "naive_persistence");
  if (!members.length) return undefined;
  const labels = learned.map(modelLabel);
  const mean = (id: string) => {
    const values = members.filter((r) => r.model_id === id).map((r) => r.RMSE);
    return values.reduce((a, b) => a + b, 0) / values.length;
  };
  return {
    data: [
      naive
        ? { type: "scatter", mode: "lines", name: "Persistence", x: labels, y: labels.map(() => naive.RMSE), line: { color: "#8795a8", width: 1.5, dash: "dash" }, hovertemplate: "Persistence %{y:.2f}<extra></extra>" }
        : null,
      { type: "scatter", mode: "markers", name: "Seed", x: members.map((r) => modelLabel(r.model_id)), y: members.map((r) => r.RMSE), text: members.map((r) => `seed ${r.member}`), marker: { size: 10, color: "#4d8dff", opacity: 0.75 }, hovertemplate: "%{text}<br>RMSE %{y:.2f}<extra></extra>" },
      { type: "scatter", mode: "markers", name: "Mean of seeds", x: labels, y: learned.map(mean), marker: { symbol: "line-ew-open", size: 26, line: { width: 3, color: "#17191d" } }, hovertemplate: "Mean %{y:.2f}<extra></extra>" },
    ].filter(Boolean),
    layout: { height: 260, yaxis: { title: "RMSE (USD)" }, xaxis: { type: "category" }, legend: { orientation: "h", y: 1.18, x: 1, xanchor: "right" } },
  };
}

function SeedMetricsTable({ rows }: { rows: RunMetricRow[] }) {
  const line = (r: RunMetricRow, label: string, strong = false) => (
    <tr key={`${r.split}-${r.model_id}-${r.member}-${r.row_kind}`} className={`border-b border-[var(--line-subtle)] last:border-0 ${strong ? "bg-[var(--bg-subtle)]" : ""}`}>
      <td className="px-3 py-1.5 text-[var(--text-muted)]">{SPLIT_LABEL[r.split] ?? r.split}</td>
      <td className="px-3 py-1.5 font-medium">{modelLabel(r.model_id)}</td>
      <td className="px-3 py-1.5">{label}</td>
      <td className="px-3 py-1.5 text-right">{fmt(r.RMSE, 2)}</td>
      <td className="px-3 py-1.5 text-right">{fmt(r.MAE, 2)}</td>
      <td className="px-3 py-1.5 text-right">{fmt(r.MAPE_pct, 3)}%</td>
      <td className="px-3 py-1.5 text-right">{r.directional_accuracy_pct == null ? "—" : pct(r.directional_accuracy_pct, 2)}</td>
      <td className="px-3 py-1.5 text-right">{r.predicted_up_pct == null ? "—" : pct(r.predicted_up_pct, 1)}</td>
    </tr>
  );
  const ordered: ReactNode[] = [];
  for (const split of ["val", "test"]) {
    for (const id of MODEL_ORDER) {
      const inSplit = rows.filter((r) => r.split === split && r.model_id === id);
      if (id === "naive_persistence") {
        const one = inSplit.find((r) => r.row_kind === "member");
        if (one) ordered.push(line(one, "—"));
        continue;
      }
      for (const r of inSplit.filter((x) => x.row_kind === "member").sort((a, b) => Number(a.member) - Number(b.member))) ordered.push(line(r, `Seed ${r.member}`));
      for (const [kind, label] of [["family_mean", "Mean"], ["family_min", "Minimum"], ["family_max", "Maximum"]] as const) {
        const r = inSplit.find((x) => x.row_kind === kind);
        if (r) ordered.push(line(r, label, kind === "family_mean"));
      }
    }
  }
  return (
    <table className="w-full min-w-[760px] text-[12px]">
      <thead><tr className="border-b border-[var(--line-subtle)] text-left">
        <th className={HEAD}>Split</th><th className={HEAD}>Model</th><th className={HEAD}>Run</th><th className={HEAD_R}>RMSE</th><th className={HEAD_R}>MAE</th><th className={HEAD_R}>MAPE</th><th className={HEAD_R}>Direction</th><th className={HEAD_R}>Predicted up</th>
      </tr></thead>
      <tbody className="tnum">{ordered}</tbody>
    </table>
  );
}

function pairLabel(row: FamilyComparisonRow): string {
  return `${modelLabel(row.model_a)} − ${modelLabel(row.model_b)}`;
}

function ComparisonsTable({ rows }: { rows: FamilyComparisonRow[] }) {
  return (
    <table className="w-full min-w-[820px] text-[12px]">
      <thead><tr className="border-b border-[var(--line-subtle)] text-left">
        <th className={HEAD}>Split</th><th className={HEAD}>Difference (RMSE, USD)</th><th className={HEAD_R}>Mean of seeds</th><th className={HEAD_R}>Seed 23</th><th className={HEAD_R}>Seed 24</th><th className={HEAD_R}>Seed 25</th><th className={HEAD_R}>95% interval</th><th className={HEAD}>Reading</th>
      </tr></thead>
      <tbody className="tnum">{rows.map((r) => (
        <tr key={`${r.split}-${r.pair}`} className="border-b border-[var(--line-subtle)] last:border-0">
          <td className="px-3 py-2 text-[var(--text-muted)]">{SPLIT_LABEL[r.split] ?? r.split}</td>
          <td className="px-3 py-2 font-medium">{pairLabel(r)}</td>
          <td className="px-3 py-2 text-right font-semibold">{fmt(r.estimate, 2)}</td>
          <td className="px-3 py-2 text-right">{fmt(r.member_23, 2)}</td>
          <td className="px-3 py-2 text-right">{fmt(r.member_24, 2)}</td>
          <td className="px-3 py-2 text-right">{fmt(r.member_25, 2)}</td>
          <td className="px-3 py-2 text-right">{interval(r.ci7_lo, r.ci7_hi)}</td>
          <td className="px-3 py-2"><VerdictPill verdict={r.verdict} /></td>
        </tr>
      ))}</tbody>
    </table>
  );
}

export function SeedsPanel() {
  return (
    <RunsState>
      {(runs) => (
        <Panel
          eyebrow="Training runs · seeds 23, 24, 25"
          hint="CryptoMamba-v and CryptoMamba-T were each trained three times with the same code, data and GPU. Every figure above is the seed-23 checkpoint; this panel shows all three runs and how far apart they land."
          actions={<Pill tone="blue">3 seeds per model</Pill>}
        >
          <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
            {(["val", "test"] as const).map((split) => (
              <div key={split} className="rounded border border-[var(--line-subtle)] bg-[var(--bg-sunken)] p-2">
                <div className="eyebrow px-2 pt-1">{SPLIT_LABEL[split]} · RMSE by seed</div>
                <Plot figure={seedFigure(runs.metrics, split)} height={260} ariaLabel={`${SPLIT_LABEL[split]} RMSE for each seed`} accessibleSummary="RMSE of each of three seeds for CryptoMamba-v and CryptoMamba-T, with the seed mean and the persistence value." />
              </div>
            ))}
          </div>
          <Callout tone="neutral" title="How the seed differences are read">
            A difference between two models is called <b>established</b> only when the 95% bootstrap interval of the seed-averaged RMSE difference excludes zero and all three per-seed differences have the same sign. Three seeds bound the spread between training runs; they do not estimate its distribution.
          </Callout>
          <Details summary="View per-seed metrics" open><SeedMetricsTable rows={runs.metrics} /></Details>
          <Details summary="View seed-averaged differences and their intervals" open><ComparisonsTable rows={runs.comparisons} /></Details>
        </Panel>
      )}
    </RunsState>
  );
}

/* ----------------------------- Ablation panel ---------------------------- */

function AblationComparisonsTable({ rows, names }: { rows: AblationComparison[]; names: Record<string, string> }) {
  return (
    <table className="w-full min-w-[980px] text-[12px]">
      <thead><tr className="border-b border-[var(--line-subtle)] text-left">
        <th className={HEAD}>#</th><th className={HEAD}>What changes</th><th className={HEAD}>Split</th><th className={HEAD_R}>Mean of seeds</th><th className={HEAD_R}>Seed 23 / 24 / 25</th><th className={HEAD_R}>95% interval</th><th className={HEAD_R}>DM p &lt; 0.05</th><th className={HEAD}>Reading</th>
      </tr></thead>
      <tbody className="tnum">{rows.map((c) => (
        <tr key={`${c.number}-${c.split}`} className="border-b border-[var(--line-subtle)] last:border-0">
          <td className="px-3 py-2 text-[var(--text-muted)]">{c.number}</td>
          <td className="px-3 py-2"><span className="font-medium">{names[c.before] ?? c.before}</span> → <span className="font-medium">{names[c.after] ?? c.after}</span><div className="text-[10.5px] text-[var(--text-muted)]">{c.label}</div></td>
          <td className="px-3 py-2 text-[var(--text-muted)]">{SPLIT_LABEL[c.split] ?? c.split}</td>
          <td className="px-3 py-2 text-right font-semibold">{fmt(c.estimate, 2)}</td>
          <td className="px-3 py-2 text-right">{c.per_seed.map((v) => fmt(v, 2)).join(" / ")}</td>
          <td className="px-3 py-2 text-right">{interval(c.ci["7"]?.[0], c.ci["7"]?.[1])}</td>
          <td className="px-3 py-2 text-right">{c.tests.filter((t) => t.dm_p < 0.05).length}/{c.tests.length}</td>
          <td className="px-3 py-2"><VerdictPill verdict={c.verdict} /></td>
        </tr>
      ))}</tbody>
    </table>
  );
}

export function AblationPanel() {
  return (
    <RunsState>
      {(runs) => {
        const { models, naive, comparisons, trading, pt } = runs.ablation;
        const names = Object.fromEntries(models.map((m) => [m.id, m.name]));
        return (
          <Panel
            eyebrow="Ablation · what separates CryptoMamba-T from CryptoMamba-v"
            hint="CryptoMamba-T differs from CryptoMamba-v in five things at once. Five intermediate models change them one at a time, each trained with the same three seeds. Each comparison is the model after minus the model before."
            actions={<Pill tone="blue">6 comparisons</Pill>}
          >
            <Callout tone="amber" title="Reading the result">
              The comparisons do not attribute CryptoMamba-T's lower error to the time-axis scan or the 60-day window. Changing only the output of CryptoMamba-v to a change from today's close lowers validation RMSE by as much, and the models with the lowest error are those whose forecasts vary least, that is, closest to persistence.
            </Callout>
            <Details summary="View every ablation model (RMSE, USD)" open>
              <table className="w-full min-w-[820px] text-[12px]">
                <thead><tr className="border-b border-[var(--line-subtle)] text-left">
                  <th className={HEAD}>Model</th><th className={HEAD}>What differs from CryptoMamba-v</th><th className={HEAD_R}>Validation · mean [min; max]</th><th className={HEAD_R}>Test · mean [min; max]</th><th className={HEAD_R}>Runs</th>
                </tr></thead>
                <tbody className="tnum">
                  {models.map((m) => (
                    <tr key={m.id} className="border-b border-[var(--line-subtle)]">
                      <td className="px-3 py-2 font-medium">{m.name}</td>
                      <td className="px-3 py-2 text-[var(--text-secondary)]">{m.note}</td>
                      {(["val", "test"] as const).map((split) => (
                        <td key={split} className="px-3 py-2 text-right">{m.splits[split] ? `${fmt(m.splits[split].rmse_mean, 2)} ${interval(m.splits[split].rmse_min, m.splits[split].rmse_max)}` : "—"}</td>
                      ))}
                      <td className="px-3 py-2 text-right">{m.splits.val?.runs.length ?? 0}</td>
                    </tr>
                  ))}
                  <tr className="bg-[var(--bg-subtle)]">
                    <td className="px-3 py-2 font-medium">Persistence</td><td className="px-3 py-2 text-[var(--text-secondary)]">No parameters; forecast = today's close</td>
                    <td className="px-3 py-2 text-right">{fmt(naive.val?.rmse, 2)}</td><td className="px-3 py-2 text-right">{fmt(naive.test?.rmse, 2)}</td><td className="px-3 py-2 text-right">—</td>
                  </tr>
                </tbody>
              </table>
            </Details>
            <Details summary="View the six comparisons" open><AblationComparisonsTable rows={comparisons} names={names} /></Details>
            <Details summary="View every ablation run: forecast, direction and trading"><AblationRunsTable models={models} trading={trading} pt={pt} /></Details>
          </Panel>
        );
      }}
    </RunsState>
  );
}

/** One row per training run: forecast error, direction and the paper's three strategies on the test split. */
function AblationRunsTable({ models, trading, pt }: { models: AblationModel[]; trading: AblationTradingRow[]; pt: AblationPtRow[] }) {
  const trade = (id: string, seed: number, strategy: string) =>
    trading.find((r) => r.model_id === `${id} s${seed}` && r.split === "test" && r.strategy === strategy && r.transaction_cost_pct === 0.2 && r.borrow_cost_bps_per_day === 0);
  const ptP = (id: string, seed: number) => pt.find((r) => r.run === `${id} s${seed}` && r.split === "test")?.pt_p_two_sided;
  const buyHold = trading.find((r) => r.strategy === "buy_hold" && r.split === "test" && r.transaction_cost_pct === 0.2);
  return (
    <div>
      <table className="w-full min-w-[1080px] text-[12px]">
        <thead><tr className="border-b border-[var(--line-subtle)] text-left">
          <th className={HEAD}>Model</th><th className={HEAD}>Seed</th>
          <th className={HEAD_R}>RMSE val</th><th className={HEAD_R}>Direction val %</th>
          <th className={HEAD_R}>RMSE test</th><th className={HEAD_R}>Direction test %</th><th className={HEAD_R}>Up forecasts test %</th><th className={HEAD_R}>PT p (test)</th>
          <th className={HEAD_R}>Vanilla</th><th className={HEAD_R}>Smart</th><th className={HEAD_R}>Extended Smart</th><th className={HEAD_R}>Vanilla trades</th>
        </tr></thead>
        <tbody className="tnum">
          {buyHold && (
            <tr className="border-b border-[var(--line-subtle)] bg-[var(--bg-subtle)]">
              <td className="px-3 py-2 font-medium">Buy & Hold</td><td className="px-3 py-2 text-[var(--text-muted)]">—</td>
              <td colSpan={7} className="px-3 py-2 text-right text-[var(--text-muted)]">reference · test · cost 0.2%</td>
              <td colSpan={3} className="px-3 py-2 text-right font-semibold">{fmt(buyHold.final_equity, 2)}</td><td className="px-3 py-2 text-right">1</td>
            </tr>
          )}
          {models.flatMap((m) => {
            const seeds = (m.splits.test?.runs ?? []).map((r) => r.seed);
            return seeds.map((seed, i) => {
              const val = m.splits.val?.runs.find((r) => r.seed === seed);
              const test = m.splits.test?.runs.find((r) => r.seed === seed);
              return (
                <tr key={`${m.id}-${seed}`} className="border-b border-[var(--line-subtle)] last:border-0">
                  <td className="px-3 py-2 font-medium">{i === 0 ? m.name : ""}</td>
                  <td className="px-3 py-2 text-[var(--text-muted)]">{seed}</td>
                  <td className="px-3 py-2 text-right">{fmt(val?.rmse, 2)}</td><td className="px-3 py-2 text-right">{fmt(val?.dir, 2)}</td>
                  <td className="px-3 py-2 text-right">{fmt(test?.rmse, 2)}</td><td className="px-3 py-2 text-right">{fmt(test?.dir, 2)}</td><td className="px-3 py-2 text-right">{fmt(test?.up, 2)}</td>
                  <td className="px-3 py-2 text-right">{ptP(m.id, seed) == null ? "—" : fmt(ptP(m.id, seed), 3)}</td>
                  <td className="px-3 py-2 text-right">{fmt(trade(m.id, seed, "vanilla")?.final_equity, 2)}</td>
                  <td className="px-3 py-2 text-right">{fmt(trade(m.id, seed, "smart")?.final_equity, 2)}</td>
                  <td className="px-3 py-2 text-right">{fmt(trade(m.id, seed, "smart_w_short")?.final_equity, 2)}</td>
                  <td className="px-3 py-2 text-right">{trade(m.id, seed, "vanilla")?.number_of_trades ?? "—"}</td>
                </tr>
              );
            });
          })}
        </tbody>
      </table>
      <p className="px-3 pt-2 text-[11px] text-[var(--text-muted)]">Errors are on the 304 dates every model can forecast. PT is the two-sided Pesaran–Timmermann p-value; it is not defined for a model that always forecasts a rise. Trading is the corrected simulator at 0.2% cost with fills at the signal-day close, start 100.</p>
    </div>
  );
}

/* --------------------------- Paper replay by seed -------------------------- */

const REPLAY_MODES = ["vanilla", "smart", "smart_w_short"];
const REPLAY_MEMBERS = ["official", "23", "24", "25"];

function ReplayTable({ rows, field, digits, title }: { rows: PaperReplayRunRow[]; field: "final_balance" | "max_drawdown_pct"; digits: number; title: string }) {
  const value = (split: string, mode: string, member: string) => rows.find((r) => r.split === split && r.trade_mode === mode && String(r.member) === member)?.[field];
  const mean = (split: string, mode: string) => {
    const v = ["23", "24", "25"].map((m) => value(split, mode, m)).filter((x): x is number => typeof x === "number");
    return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null;
  };
  return (
    <div className="mt-4">
      <div className="eyebrow px-1 pb-1">{title}</div>
      <table className="w-full min-w-[720px] text-[12px]">
        <thead><tr className="border-b border-[var(--line-subtle)] text-left">
          <th className={HEAD}>Split</th><th className={HEAD}>Strategy</th><th className={HEAD_R}>Official checkpoint</th>
          <th className={HEAD_R}>Seed 23</th><th className={HEAD_R}>Seed 24</th><th className={HEAD_R}>Seed 25</th><th className={HEAD_R}>Mean of seeds</th>
        </tr></thead>
        <tbody className="tnum">
          {(["val", "test"] as const).flatMap((split) => REPLAY_MODES.map((mode, i) => (
            <tr key={`${split}-${mode}`} className="border-b border-[var(--line-subtle)] last:border-0">
              <td className="px-3 py-2 text-[var(--text-muted)]">{i === 0 ? SPLIT_LABEL[split] : ""}</td>
              <td className="px-3 py-2 font-medium">{prettyModel(mode)}</td>
              {REPLAY_MEMBERS.map((member) => <td key={member} className="px-3 py-2 text-right">{fmt(value(split, mode, member), digits)}</td>)}
              <td className="px-3 py-2 text-right font-semibold">{fmt(mean(split, mode), digits)}</td>
            </tr>
          )))}
        </tbody>
      </table>
    </div>
  );
}

export function PaperReplayBySeedPanel() {
  return (
    <RunsState>
      {(runs) => (
        <Panel
          eyebrow="Paper engine replay · every training run · zero transaction cost"
          hint="The paper's released trading code on each CryptoMamba-v run and on the official checkpoint, over the 350 dates of each split, start 100 and risk 2%."
          actions={<Pill tone="amber">Reproduction audit</Pill>}
        >
          <div className="overflow-x-auto">
            <ReplayTable rows={runs.paper_replay} field="final_balance" digits={2} title="Final balance" />
            <ReplayTable rows={runs.paper_replay} field="max_drawdown_pct" digits={2} title="Maximum drawdown (%)" />
          </div>
        </Panel>
      )}
    </RunsState>
  );
}

/* --------------------------- Trading by seed tab -------------------------- */

const PERIODS: Array<{ id: string; label: string }> = [
  { id: "thesis304_val", label: "Validation · 304 days" },
  { id: "thesis304_test", label: "Test · 304 days" },
  { id: "holdout_full", label: "After 09/2024 · 685 days" },
  { id: "holdout_A", label: "After 09/2024 · A, rising · 365 days" },
  { id: "holdout_B", label: "After 09/2024 · B, falling · 320 days" },
];
const EXECUTIONS = [{ id: "close", label: "Fill at signal-day close" }, { id: "next_open", label: "Fill at next day's open" }];
const COSTS = [0, 0.1, 0.2, 0.3, 0.5];
const STRATEGIES = ["vanilla", "smart", "smart_w_short"];

/** A drawdown is a non-negative depth; a run that never trades reports -0, which would print "-0.00%". */
function drawdown(value: number | null | undefined): number | null {
  return value == null ? null : Math.abs(value);
}

function tradingRunLabel(model: string): string {
  const match = /^(cmv|cmt)_s(\d+)$/.exec(model);
  if (match) return `${modelLabel(match[1])} · seed ${match[2]}`;
  if (model === "cmv_official") return "Official CM-v checkpoint";
  return modelLabel(model);
}

function runOrder(model: string): number {
  const match = /^(cmv|cmt)_s(\d+)$/.exec(model);
  if (!match) return model === "cmv_official" ? 0 : 99;
  return (match[1] === "cmv" ? 10 : 20) + Number(match[2]) - 23;
}

export function SeedTradingPanel() {
  const [period, setPeriod] = useState("thesis304_test");
  const [execution, setExecution] = useState("close");
  const [cost, setCost] = useState(0.2);
  const [borrow, setBorrow] = useState(0);
  const [detail, setDetail] = useState("smart");
  return (
    <RunsState>
      {(runs) => <SeedTradingTable rows={runs.trading} period={period} setPeriod={setPeriod} execution={execution} setExecution={setExecution} cost={cost} setCost={setCost} borrow={borrow} setBorrow={setBorrow} detail={detail} setDetail={setDetail} />}
    </RunsState>
  );
}

function SeedTradingTable({ rows, period, setPeriod, execution, setExecution, cost, setCost, borrow, setBorrow, detail, setDetail }: {
  rows: RunTradingRow[]; period: string; setPeriod: (v: string) => void; execution: string; setExecution: (v: string) => void; cost: number; setCost: (v: number) => void;
  borrow: number; setBorrow: (v: number) => void; detail: string; setDetail: (v: string) => void;
}) {
  const inCell = useMemo(
    () => rows.filter((r) => r.period === period && r.execution === execution && r.cost_pct === cost),
    [rows, period, execution, cost],
  );
  const borrowOptions = useMemo(
    () => [...new Set(inCell.filter((r) => r.strategy === "smart_w_short").map((r) => r.borrow_bps_day))].sort((a, b) => a - b),
    [inCell],
  );
  const activeBorrow = borrowOptions.includes(borrow) ? borrow : 0;
  // Borrow cost only applies to Extended Smart's short leg; every other strategy is shown at zero borrow.
  const selected = useMemo(
    () => inCell.filter((r) => r.borrow_bps_day === (r.strategy === "smart_w_short" ? activeBorrow : 0)),
    [inCell, activeBorrow],
  );
  const models = useMemo(
    () => [...new Set(selected.filter((r) => r.strategy !== "buy_hold" && !r.model.endsWith("thesis") && r.model !== "naive").map((r) => r.model))].sort((a, b) => runOrder(a) - runOrder(b)),
    [selected],
  );
  const benchmark = selected.find((r) => r.strategy === "buy_hold");
  const at = (model: string, strategy: string) => selected.find((r) => r.model === model && r.strategy === strategy);

  return (
    <Panel
      eyebrow="Corrected self-financing · every training run"
      hint="The same simulator as the main tab, for each seed of each model. Vanilla, Smart and Extended Smart are the paper's three strategies; Buy & Hold is the model-independent benchmark."
      actions={<Pill tone="blue">Initial equity 100</Pill>}
    >
      <div className="grid gap-4 md:grid-cols-4">
        <div className="flex flex-col gap-2"><Label>Period</Label>
          <Select value={period} onValueChange={setPeriod}><SelectTrigger aria-label="Period"><SelectValue /></SelectTrigger><SelectContent>{PERIODS.map((p) => <SelectItem key={p.id} value={p.id}>{p.label}</SelectItem>)}</SelectContent></Select></div>
        <div className="flex flex-col gap-2"><Label>Order fill</Label>
          <Select value={execution} onValueChange={setExecution}><SelectTrigger aria-label="Order fill"><SelectValue /></SelectTrigger><SelectContent>{EXECUTIONS.map((p) => <SelectItem key={p.id} value={p.id}>{p.label}</SelectItem>)}</SelectContent></Select></div>
        <div className="flex flex-col gap-2"><Label>Cost per trade</Label>
          <Select value={String(cost)} onValueChange={(v) => setCost(Number(v))}><SelectTrigger aria-label="Cost per trade"><SelectValue /></SelectTrigger><SelectContent>{COSTS.map((c) => <SelectItem key={c} value={String(c)}>{c}% of traded value</SelectItem>)}</SelectContent></Select></div>
        <div className="flex flex-col gap-2"><Label>Borrow cost (Extended Smart)</Label>
          <Select value={String(activeBorrow)} onValueChange={(v) => setBorrow(Number(v))} disabled={borrowOptions.length < 2}><SelectTrigger aria-label="Borrow cost"><SelectValue /></SelectTrigger><SelectContent>{(borrowOptions.length ? borrowOptions : [0]).map((b) => <SelectItem key={b} value={String(b)}>{b} bps per day on the short</SelectItem>)}</SelectContent></Select></div>
      </div>
      {period.startsWith("holdout") && (
        <Callout tone="amber" title="After 09/2024 is descriptive">
          These dates were not used to choose any model, setting or rule, but they are a single stretch of history that includes a long fall. A result above Buy & Hold in the falling stretch mostly reflects holding less BTC, not forecasting skill.
        </Callout>
      )}
      <div className="mt-4 overflow-x-auto">
        {selected.length === 0 ? (
          <Callout tone="amber">No rows for this combination.</Callout>
        ) : (
          <table className="w-full min-w-[900px] text-[12px]">
            <thead><tr className="border-b border-[var(--line-subtle)] text-left">
              <th className={HEAD}>Run</th>
              {STRATEGIES.map((s) => <th key={s} className={HEAD_R}>{prettyModel(s)} · final equity</th>)}
              {STRATEGIES.map((s) => <th key={`m-${s}`} className={HEAD_R}>{prettyModel(s)} · MDD</th>)}
              <th className={HEAD_R}>Trades (Smart)</th>
            </tr></thead>
            <tbody className="tnum">
              {benchmark && (
                <tr className="border-b border-[var(--line-subtle)] bg-[var(--bg-subtle)]">
                  <td className="px-3 py-2 font-medium">Buy & Hold</td>
                  <td className="px-3 py-2 text-right font-semibold" colSpan={3}>{fmt(benchmark.final_equity, 2)}</td>
                  <td className="px-3 py-2 text-right" colSpan={3}>{pct(drawdown(benchmark.mdd_pct))}</td>
                  <td className="px-3 py-2 text-right">{benchmark.trades}</td>
                </tr>
              )}
              {models.map((model) => (
                <tr key={model} className="border-b border-[var(--line-subtle)] last:border-0">
                  <td className="px-3 py-2 font-medium">{tradingRunLabel(model)}</td>
                  {STRATEGIES.map((s) => <td key={s} className="px-3 py-2 text-right">{fmt(at(model, s)?.final_equity, 2)}</td>)}
                  {STRATEGIES.map((s) => <td key={`m-${s}`} className="px-3 py-2 text-right">{pct(drawdown(at(model, s)?.mdd_pct))}</td>)}
                  <td className="px-3 py-2 text-right">{at(model, "smart")?.trades ?? "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
      <p className="mt-3 text-[11px] text-[var(--text-muted)]">Borrow cost is charged only on Extended Smart's short position and is available at 0.2% cost; at every other cost it is zero. Persistence never trades, so it ends at 100 under every strategy and is left out.</p>
      <details className="mt-4 rounded border border-[var(--line-subtle)] bg-[var(--bg-sunken)]">
        <summary className="cursor-pointer px-3 py-2 font-mono text-[11px] font-medium text-[var(--text-secondary)]">Risk-adjusted measures, fees and trades for one strategy</summary>
        <div className="border-t border-[var(--line-subtle)] p-3">
          <div className="mb-3 flex max-w-xs flex-col gap-2"><Label>Strategy</Label>
            <Select value={detail} onValueChange={setDetail}><SelectTrigger aria-label="Strategy"><SelectValue /></SelectTrigger><SelectContent>{STRATEGIES.map((s) => <SelectItem key={s} value={s}>{prettyModel(s)}</SelectItem>)}</SelectContent></Select></div>
          <div className="overflow-x-auto">
            <table className="w-full min-w-[820px] text-[12px]">
              <thead><tr className="border-b border-[var(--line-subtle)] text-left">
                <th className={HEAD}>Run</th><th className={HEAD_R}>Final equity</th><th className={HEAD_R}>MDD</th><th className={HEAD_R}>Sharpe</th><th className={HEAD_R}>Sortino</th><th className={HEAD_R}>Calmar</th><th className={HEAD_R}>Trades</th><th className={HEAD_R}>Fees</th>
              </tr></thead>
              <tbody className="tnum">
                {[...(benchmark ? [{ label: "Buy & Hold", row: benchmark }] : []), ...models.map((m) => ({ label: tradingRunLabel(m), row: at(m, detail) }))].map(({ label, row }) => (
                  <tr key={label} className="border-b border-[var(--line-subtle)] last:border-0">
                    <td className="px-3 py-2 font-medium">{label}</td>
                    <td className="px-3 py-2 text-right">{fmt(row?.final_equity, 2)}</td><td className="px-3 py-2 text-right">{pct(drawdown(row?.mdd_pct))}</td>
                    <td className="px-3 py-2 text-right">{fmt(row?.sharpe, 2)}</td><td className="px-3 py-2 text-right">{fmt(row?.sortino, 2)}</td><td className="px-3 py-2 text-right">{fmt(row?.calmar, 2)}</td>
                    <td className="px-3 py-2 text-right">{row?.trades ?? "—"}</td><td className="px-3 py-2 text-right">{fmt(row?.fees, 2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="mt-2 text-[11px] text-[var(--text-muted)]">Calmar is not defined for a run whose drawdown is zero.</p>
        </div>
      </details>
    </Panel>
  );
}
