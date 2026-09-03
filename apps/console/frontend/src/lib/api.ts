/* Typed client for the console API.
 *
 * Every number rendered by this app comes from these endpoints; the backend
 * reuses the validated cryptomamba_ui logic. Nothing here computes model or
 * data results — the frontend is presentation only.
 */

export type PlotlyFigure = { data: unknown[]; layout?: Record<string, unknown> };

async function getJSON<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  let body: unknown = null;
  try {
    body = await res.json();
  } catch {
    /* non-JSON response (e.g. a proxy error page) */
  }
  if (!res.ok) {
    const detail =
      (body as { detail?: string; error?: string } | null)?.detail ??
      (body as { error?: string } | null)?.error;
    throw new Error(detail || `${res.status} ${res.statusText}`);
  }
  return body as T;
}

/* ------------------------------- Data ---------------------------------- */

export interface DataMetric {
  label: string;
  value: string;
  unit: string;
  range: string;
  color: string;
}

export interface SplitRow {
  split: string;
  rows: number;
  from: string;
  to: string;
}

export interface PipelineStage {
  stage: string;
  artifact: string;
  status: string;
}

export interface DataProcessingSummary {
  raw_rows: number;
  daily_rows: number;
  processed_rows: number;
  split_strategy: "paper_date" | "chronological_ratio" | string;
}

export interface DataCandle {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  split: string;
}

export interface PredictionPayloadPreview {
  prediction_date: string;
  risk: number;
  candles: Array<Omit<DataCandle, "split">>;
}

export interface DataWindowEvidence {
  window: { start: string; end: string; size: number; rows: DataCandle[] };
  tensor: {
    shape: number[];
    feature_order: string[];
    seq_len: number;
    volume_rule: string;
    model_config: string;
  };
  prediction_payload: PredictionPayloadPreview;
}

export interface DataResponse extends DataWindowEvidence {
  source: { label: string; detail: string; strategy: string };
  processing: DataProcessingSummary;
  metrics: DataMetric[];
  splits: SplitRow[];
  pipeline: PipelineStage[];
  candles: DataCandle[];
}

export const fetchData = () => getJSON<DataResponse>("/api/data?mode=paper");

export function uploadCsv(file: File) {
  const fd = new FormData();
  fd.append("file", file);
  return getJSON<DataResponse>("/api/data/upload", { method: "POST", body: fd });
}

export function previewDataWindow(candles: DataCandle[], predictionDate: string) {
  return getJSON<DataWindowEvidence>("/api/data/window", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ candles, prediction_date: predictionDate, risk: 2 }),
  });
}

/* ----------------------------- Reproduce -------------------------------- */





export interface Reproduction350dMetricRow {
  result_type: "official_checkpoint" | "retrained_checkpoint";
  display_name: string;
  samples: number;
  date_from: string;
  date_to: string;
  RMSE: number;
  MAE: number;
  MAPE_pct: number;
  RMSE_gap_pct: number;
  MAE_gap_pct: number;
  MAPE_gap_pct: number;
  tolerance_pct: number;
  status: string;
}

export interface Reproduction350d {
  status: string;
  error?: string | null;
  split?: "test";
  samples?: number;
  date_from?: string | null;
  date_to?: string | null;
  criterion?: { threshold_pct: number; label: string };
  paper_reference?: {
    model_id: string;
    display_name: string;
    RMSE: number;
    MAE: number;
    MAPE_pct: number;
    source_label: string;
    source_table: string;
    source_url: string;
  };
  rows: Reproduction350dMetricRow[];
  evidence?: {
    source_artifact: string;
    source_sha256: string;
    source_bytes: number;
    manifest_artifact: string;
    evidence_package_sha256: string;
  };
}

export interface RobustnessRmseRow {
  split: "val" | "test";
  model_id: ControlledForecastMetricRow["model_id"];
  estimate: number;
  ci_low: number;
  ci_high: number;
}

export interface RobustnessRmseDifferenceRow {
  split: "val" | "test";
  comparison: string;
  estimate: number;
  ci_low: number;
  ci_high: number;
}

export interface RobustnessDirectionRow {
  split: "val" | "test";
  model_id: "cmamba_v_reproduced" | "s5_full";
  estimate_pct: number;
  ci_low_pct: number;
  ci_high_pct: number;
}

export interface RobustnessDirectionDifferenceRow {
  split: "val" | "test";
  comparison: string;
  estimate_pp: number;
  ci_low_pp: number;
  ci_high_pp: number;
  model_a_only_correct: number;
  model_b_only_correct: number;
  mcnemar_exact_p: number;
}

export interface RobustnessPeriodRow {
  period: string;
  samples: number;
  date_from: string;
  date_to: string;
  cmamba_v_rmse: number;
  s5_full_rmse: number;
  persistence_rmse: number;
  cmamba_v_direction_pct: number;
  s5_full_direction_pct: number;
}

export interface RobustnessSensitivityRow {
  split: "val" | "test";
  block_length: 5 | 7 | 14;
  rmse_difference_estimate: number;
  rmse_ci_low: number;
  rmse_ci_high: number;
  rmse_ci_includes_zero: boolean;
  direction_difference_estimate_pp: number;
  direction_ci_low_pp: number;
  direction_ci_high_pp: number;
  direction_ci_includes_zero: boolean;
}

export interface ForecastRobustness {
  status: string;
  error?: string | null;
  protocol?: {
    method: string;
    resamples: number;
    primary_block_length: number;
    block_lengths: number[];
    random_seed: number;
    confidence_level: number;
    retraining: boolean;
  };
  rmse: RobustnessRmseRow[];
  rmse_difference: RobustnessRmseDifferenceRow[];
  direction: RobustnessDirectionRow[];
  direction_difference: RobustnessDirectionDifferenceRow[];
  periods: RobustnessPeriodRow[];
  sensitivity: RobustnessSensitivityRow[];
  evidence?: {
    source_artifact: string;
    source_sha256: string;
    source_rows: number;
    samples_per_model_split: number;
    evidence_package_sha256: string;
  };
}

export interface ReproduceResponse {
  status: string;
  final_evidence_status: string;
  final_evidence_error?: string | null;
  final_evidence_sha256?: string;
  paper_reported: PaperReportedMetricRow[];
  controlled_local: ControlledForecastMetricRow[];
  paired_tests: PairedTestRow[];
  reproduction_350d: Reproduction350d;
  forecast_robustness: ForecastRobustness;
  paired_test_definitions?: {
    diebold_mariano: string;
    wilcoxon: string;
    scope: string;
  };
  errors: string[];
}

export interface PaperReportedMetricRow {
  model_id: string;
  display_name: string;
  RMSE: number;
  MAE: number;
  MAPE_pct: number;
  parameter_count: number;
  source_label: string;
  source_table: string;
  source_url: string;
}

export interface ControlledForecastMetricRow {
  model_id: "cmamba_v_reproduced" | "s5_full" | "naive_persistence";
  display_name: string;
  split: "val" | "test";
  samples: number;
  date_from: string;
  date_to: string;
  MSE: number;
  RMSE: number;
  MAE: number;
  MAPE_pct: number;
  directional_accuracy_pct: number | null;
  directional_coverage_pct: number;
  predicted_up_pct: number;
  parameter_count: number;
  checkpoint_sha256: string | null;
}

export interface PairedTestRow {
  split: "val" | "test";
  model_a: string;
  model_b: string;
  comparison: string;
  samples: number;
  dm_loss: "squared_error";
  dm_hac_lag: number;
  dm_statistic: number;
  dm_p_value: number;
  wilcoxon_loss: "absolute_error";
  wilcoxon_statistic: number;
  wilcoxon_p_value: number;
  lower_mse_model: string;
  lower_mae_model: string;
  dm_significant_5pct: boolean;
  wilcoxon_significant_5pct: boolean;
}

export const fetchReproduce = () => getJSON<ReproduceResponse>("/api/reproduce");

/* --------------------------- Architecture study -------------------------- */

export interface CorrectedTradingMetricRow {
  model_id: "cmamba_v_reproduced" | "s5_full" | "naive_persistence";
  split: "val" | "test";
  strategy: string;
  transaction_cost_pct: number;
  borrow_cost_bps_per_day: number;
  initial_equity: number;
  final_equity: number;
  ROI_pct: number;
  max_drawdown_pct: number;
  Sharpe: number;
  turnover_notional: number;
  number_of_trades: number;
  total_fees: number;
  total_borrow_cost: number;
  reconciliation_error: number;
}

export interface ArchitectureResponse {
  status: string;
  message?: string;
  final_evidence_sha256?: string;
  model?: {
    display_name: string;
    window_days: number;
    parameter_count: number;
    checkpoint_sha256: string;
    source_commit: string;
    prediction_mode: string;
  };
  parameter_comparison?: {
    reproduced_cm_v: number;
    s5_full: number;
    s5_to_cm_v_pct: number;
  };
  source_execution?: {
    input: string;
    feature_order: string[];
    normalization: string;
    scan_axis: string;
    stages: string[];
  };
  training?: {
    config?: string;
    commit?: string;
    branch?: string;
    gpu?: string;
    torch?: string;
    mamba_ssm?: string;
    seed?: number;
    best_epoch?: string;
    training_seconds?: number;
  };
  controlled_metrics?: ControlledForecastMetricRow[];
  paired_tests?: PairedTestRow[];
  corrected_trading?: CorrectedTradingMetricRow[];
  reference_cost_pct?: number;
}

export const fetchArchitecture = () => getJSON<ArchitectureResponse>("/api/architecture");

/* ------------------------------ Predict --------------------------------- */

export type PredictMode = "checkpoint" | "historical" | "live";
export type CheckpointModelId = "cmamba_v_reproduced" | "s5_full";

export interface CheckpointModelOption {
  id: CheckpointModelId;
  label: string;
  window_days: 14 | 60;
  checkpoint_sha256: string;
}

export interface PredictSetup {
  default_mode: "checkpoint";
  checkpoint_models: CheckpointModelOption[];
  checkpoint_available: boolean;
  checkpoint_error: string | null;
  min_date: string;
  max_date: string;
  default_date: string;
  last_close: number;
  last_date: string;
  model_train_horizon: string;
  offline_available: boolean;
  offline_dates: string[];
  offline_min_date: string | null;
  offline_max_date: string | null;
  offline_default_date: string | null;
  api_url_default: string;
}

export interface PredictCandle {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface ForecastVariant {
  id: "raw";
  label: string;
  source: string;
  prediction_date: string;
  predicted_close: number;
  move_pct: number;
  error_pct?: number;
  direction_correct?: boolean;
}

export interface PredictResult {
  available?: boolean;
  error?: string;
  inference_type: string;
  model_id: string;
  prediction_date: string;
  last_close: number;
  predicted_close: number;
  move_pct: number;
  expected_return_pct?: number;
  window_days?: number;
  vanilla_action?: string;
  smart_action?: string;
  smart_pct?: number | null;
  checkpoint_sha256?: string;
  source_commit?: string;
  ood: boolean;
  window: { start: string; end: string; size: number; rows: PredictCandle[] };
  charts: { candle: PlotlyFigure };
  forecast_variants?: ForecastVariant[];
  provenance?: Record<string, string>;
  actual_close?: number;
  error_pct?: number;
  actual_return_pct?: number;
  direction_correct?: boolean;
}

export const fetchPredictSetup = () => getJSON<PredictSetup>("/api/predict");

export const runCheckpointPrediction = (body: {
  model_id: CheckpointModelId;
  candles: PredictCandle[];
  prediction_date: string;
}) =>
  getJSON<PredictResult>("/api/predict/checkpoint", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

export const fetchOfflinePrediction = (date?: string) =>
  getJSON<PredictResult>("/api/predict/offline" + (date ? `?date=${encodeURIComponent(date)}` : ""));

export const runLivePrediction = (body: { api_url: string; prediction_date?: string; risk: number }) =>
  getJSON<PredictResult>("/api/predict/live", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

/* ------------------------------ Trading --------------------------------- */

export interface SimulateRow {
  strategy: string;
  action: string;
  action_color: string;
  trade_size: string;
  end_value: number;
  pnl: number;
  roi_pct: number;
}

export interface SimulateResponse {
  start_value: number;
  assumed_close: number;
  move_pct: number;
  rows: SimulateRow[];
  charts: { roi: PlotlyFigure };
}

export const runSimulate = (body: {
  current: number;
  predicted: number;
  capital: number;
  btc: number;
  risk: number;
  realized_move: number;
}) =>
  getJSON<SimulateResponse>("/api/trading/simulate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

export interface BacktestMetricRow {
  strategy: string;
  transaction_cost_pct: number;
  initial_balance: number;
  final_balance: number;
  ROI_pct: number;
  max_drawdown_pct: number;
  Sharpe: number;
  number_of_trades: number;
  win_rate_pct: number;
  [k: string]: unknown;
}

export interface RegimeRow {
  regime: string;
  strategy: string;
  transaction_cost_pct: number;
  samples: number;
  RMSE: number;
  MAPE_pct: number;
  ROI_pct: number;
  max_drawdown_pct: number;
  [k: string]: unknown;
}

export interface BacktestResponse {
  status: string;
  error?: string;
  result_type: string;
  split: string;
  ref_cost: number;
  available: { result_types: string[]; splits: string[]; costs: number[]; strategies: string[] };
  metrics: BacktestMetricRow[];
  regime: RegimeRow[];
  metadata: {
    regime_window_days?: number;
    regime_up_pct?: number;
    regime_down_pct?: number;
    [k: string]: unknown;
  };
  charts: Record<string, PlotlyFigure | undefined>;
  final_evidence_status: string;
  final_evidence_error?: string | null;
  final_evidence_sha256?: string;
  paper_replay: PaperReplayMetricRow[];
  corrected: CorrectedTradingMetricRow[];
  corrected_metadata: CorrectedTradingMetadata;
}

export interface PaperReplayMetricRow {
  result_type: string;
  model: string;
  checkpoint: string;
  source_commit: string;
  split: "val" | "test";
  trade_mode: string;
  initial_balance: number;
  risk_pct: number;
  final_balance: number;
  max_drawdown_pct: number;
  paper_final_balance: number;
  paper_max_drawdown_pct: number;
  balance_gap_pct: number;
  mdd_gap_pp: number;
  protocol: string;
  status: string;
}

export interface CorrectedTradingMetadata {
  engine?: string;
  execution_basis?: string;
  mark_basis?: string;
  same_close_execution?: boolean;
  thesis_reference_cost_pct?: number;
  risk_pct?: number;
  transaction_cost_pct_scenarios?: number[];
  target_close_used_only_for?: string[];
}

export const fetchBacktest = (params: { result_type: string; split: string; ref_cost: number }) =>
  getJSON<BacktestResponse>(
    "/api/trading/backtest?" +
      new URLSearchParams({
        result_type: params.result_type,
        split: params.split,
        ref_cost: String(params.ref_cost),
      }),
  );

export interface TradingReplayRow {
  step: number;
  date: string;
  decision_date: string;
  outcome_date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  window_start_date: string;
  window_end_date: string;
  current_close: number;
  target_close: number;
  predicted_close: number;
  predicted_return_pct: number;
  actual_return_pct: number;
  action: "BUY" | "SELL" | "HOLD";
  cash: number;
  position_btc: number;
  portfolio_value: number;
  drawdown_pct: number;
  trade_value: number;
  transaction_cost: number;
}

export interface TradingReplayResponse {
  status: string;
  error?: string;
  result_type: string;
  split: string;
  strategy: string;
  ref_cost: number;
  available: BacktestResponse["available"];
  summary: BacktestMetricRow & {
    curve_last_value: number;
    terminal_valuation_delta: number;
  };
  candles: Array<{
    date: string;
    open: number;
    high: number;
    low: number;
    close: number;
    volume: number;
  }>;
  rows: TradingReplayRow[];
  provenance: {
    evidence_kind: string;
    order_execution: boolean;
    checkpoint: string;
    model: string;
    source_commit: string;
    protocol: string;
    artifacts: string[];
    valuation_basis: string;
  };
}

export const fetchTradingReplay = (params: {
  result_type: string;
  split: string;
  strategy: string;
  ref_cost: number;
}) =>
  getJSON<TradingReplayResponse>(
    "/api/trading/replay?" +
      new URLSearchParams({
        result_type: params.result_type,
        split: params.split,
        strategy: params.strategy,
        ref_cost: String(params.ref_cost),
      }),
  );

/* -------------------------------- Health -------------------------------- */

