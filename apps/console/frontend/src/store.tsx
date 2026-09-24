import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";

import * as api from "@/lib/api";

/* Single store for all five screens. Screens read state and call actions; they
 * never fetch on their own. Each payload is cached after its first load, so
 * navigating between screens does not re-hit the backend and per-screen state
 * survives (an uploaded CSV stays loaded, a prediction stays on screen). */

export type ScreenId = "data" | "reproduce" | "predict" | "trading";

export const SCREENS: { id: ScreenId; no: string; label: string; title: string }[] = [
  { id: "data", no: "01", label: "Data", title: "Dataset" },
  { id: "reproduce", no: "02", label: "Evaluation", title: "Model Evaluation" },
  { id: "predict", no: "03", label: "Predict", title: "Prediction" },
  { id: "trading", no: "04", label: "Trading", title: "Trading" },
];

function msg(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

function nextUtcDate(value: string): string {
  const date = new Date(`${value}T00:00:00Z`);
  date.setUTCDate(date.getUTCDate() + 1);
  return date.toISOString().slice(0, 10);
}

function useConsoleStore() {
  const [screen, setScreen] = useState<ScreenId>("data");

  // ---- Data ----
  const [paperData, setPaperData] = useState<api.DataResponse | null>(null);
  const [uploadedData, setUploadedData] = useState<api.DataResponse | null>(null);
  const [dataMode, setDataMode] = useState<"paper" | "upload">("paper");
  const [dataLoading, setDataLoading] = useState(false);
  const [dataError, setDataError] = useState("");
  const [dataWindowLoading, setDataWindowLoading] = useState(false);
  const [dataWindowError, setDataWindowError] = useState("");
  const [uploadName, setUploadName] = useState("");
  const dataRequestId = useRef(0);
  const dataWindowRequestId = useRef(0);

  // ---- Reproduce ----
  const [repro, setRepro] = useState<api.ReproduceResponse | null>(null);
  const [reproLoading, setReproLoading] = useState(false);
  const [reproError, setReproError] = useState("");

  // ---- Predict ----
  const [setup, setSetup] = useState<api.PredictSetup | null>(null);
  const [prediction, setPrediction] = useState<api.PredictResult | null>(null);
  const [predictMode, setPredictMode] = useState<api.PredictMode>("checkpoint");
  const [checkpointResults, setCheckpointResults] =
    useState<Partial<Record<api.CheckpointModelId, api.PredictResult>>>({});
  const [checkpointErrors, setCheckpointErrors] =
    useState<Partial<Record<api.CheckpointModelId, string>>>({});
  const [predictLoading, setPredictLoading] = useState(false);
  const [predictError, setPredictError] = useState("");
  const [offlineUnavailable, setOfflineUnavailable] = useState(false);
  const [offlineDate, setOfflineDate] = useState("");
  const [predictDate, setPredictDate] = useState("");
  const [apiUrl, setApiUrl] = useState("");
  const [risk, setRisk] = useState(2);
  const predictRequestId = useRef(0);
  const predictModeRef = useRef<api.PredictMode>("checkpoint");

  // ---- Historical backtest ----
  const [backtest, setBacktest] = useState<api.BacktestResponse | null>(null);
  const [backtestLoading, setBacktestLoading] = useState(false);
  const [backtestError, setBacktestError] = useState("");
  const [btResultType, setBtResultType] = useState("retrained_checkpoint");
  const [btSplit, setBtSplit] = useState("test");
  const [btCost, setBtCost] = useState(0.1);
  const [replay, setReplay] = useState<api.TradingReplayResponse | null>(null);
  const [replayLoading, setReplayLoading] = useState(false);
  const [replayError, setReplayError] = useState("");
  const [replayStrategy, setReplayStrategy] = useState("smart");
  const backtestRequestId = useRef(0);
  const replayRequestId = useRef(0);

  const data = dataMode === "upload" ? uploadedData : paperData;

  const showPaper = useCallback(async () => {
    const requestId = ++dataRequestId.current;
    ++dataWindowRequestId.current;
    setDataWindowLoading(false);
    setDataWindowError("");
    setDataMode("paper");
    setDataError("");
    if (paperData) {
      setDataLoading(false);
      return;
    }
    setDataLoading(true);
    try {
      const response = await api.fetchData();
      if (requestId === dataRequestId.current) setPaperData(response);
    } catch (e) {
      if (requestId === dataRequestId.current) setDataError(msg(e));
    } finally {
      if (requestId === dataRequestId.current) setDataLoading(false);
    }
  }, [paperData]);

  const uploadCsv = useCallback(async (file: File) => {
    const requestId = ++dataRequestId.current;
    ++dataWindowRequestId.current;
    setDataWindowLoading(false);
    setDataWindowError("");
    setDataLoading(true);
    setDataError("");
    try {
      const res = await api.uploadCsv(file);
      if (requestId === dataRequestId.current) {
        setUploadedData(res);
        setUploadName(file.name);
        setDataMode("upload");
      }
    } catch (e) {
      if (requestId === dataRequestId.current) setDataError(msg(e));
    } finally {
      if (requestId === dataRequestId.current) setDataLoading(false);
    }
  }, []);

  const showUpload = useCallback(() => {
    ++dataRequestId.current;
    ++dataWindowRequestId.current;
    setDataError("");
    setDataLoading(false);
    setDataWindowLoading(false);
    setDataWindowError("");
    if (uploadedData) setDataMode("upload");
  }, [uploadedData]);

  const clearUpload = useCallback(() => {
    setUploadedData(null);
    setUploadName("");
    void showPaper();
  }, [showPaper]);

  const selectDataWindow = useCallback(async (candles: api.DataCandle[], predictionDate: string) => {
    const requestId = ++dataWindowRequestId.current;
    const mode = dataMode;
    setDataWindowLoading(true);
    setDataWindowError("");
    try {
      const evidence = await api.previewDataWindow(candles, predictionDate);
      if (requestId !== dataWindowRequestId.current) return;
      const merge = (current: api.DataResponse | null): api.DataResponse | null =>
        current ? { ...current, ...evidence } : current;
      if (mode === "paper") setPaperData(merge);
      else setUploadedData(merge);
    } catch (e) {
      if (requestId === dataWindowRequestId.current) setDataWindowError(msg(e));
    } finally {
      if (requestId === dataWindowRequestId.current) setDataWindowLoading(false);
    }
  }, [dataMode]);

  const cancelDataWindowSelection = useCallback(() => {
    ++dataWindowRequestId.current;
    setDataWindowLoading(false);
    setDataWindowError("");
  }, []);

  const loadReproduce = useCallback(async () => {
    setReproLoading(true);
    setReproError("");
    try {
      setRepro(await api.fetchReproduce());
    } catch (e) {
      setReproError(msg(e));
    } finally {
      setReproLoading(false);
    }
  }, []);

  const loadOffline = useCallback(async (date?: string) => {
    const requestId = ++predictRequestId.current;
    setPredictLoading(true);
    setPredictError("");
    setPrediction(null);
    try {
      const res = await api.fetchOfflinePrediction(date);
      if (requestId !== predictRequestId.current || predictModeRef.current !== "historical") return;
      if (res && res.available) {
        setPrediction(res);
        setOfflineUnavailable(false);
        if (res.prediction_date) setOfflineDate(res.prediction_date);
      } else {
        setPrediction(null);
        setOfflineUnavailable(true);
      }
    } catch (e) {
      if (requestId === predictRequestId.current && predictModeRef.current === "historical") {
        setPrediction(null);
        setPredictError(msg(e));
      }
    } finally {
      if (requestId === predictRequestId.current && predictModeRef.current === "historical") {
        setPredictLoading(false);
      }
    }
  }, []);

  const loadPredict = useCallback(async () => {
    setPredictLoading(true);
    setPredictError("");
    try {
      const s = await api.fetchPredictSetup();
      setSetup(s);
      setApiUrl((current) => current || s.api_url_default || "");
      setPredictDate((current) => current || s.default_date);
      setOfflineDate((current) => current || s.offline_default_date || "");
    } catch (e) {
      setPredictError(msg(e));
    } finally {
      setPredictLoading(false);
    }
  }, []);

  const runCheckpoint = useCallback(async () => {
    if (!data) {
      setPredictError("Load or upload a dataset before running checkpoint inference.");
      return;
    }
    if (!predictDate) {
      setPredictError("Select a prediction date.");
      return;
    }
    const models = setup?.checkpoint_models ?? [];
    const available = data.candles.filter((candle) => candle.date < predictDate).length;
    const candles = data.candles.map(({ date, open, high, low, close, volume }) => ({
      date, open, high, low, close, volume,
    }));

    const requestId = ++predictRequestId.current;
    setPredictLoading(true);
    setPredictError("");
    setPrediction(null);
    setCheckpointResults({});
    setCheckpointErrors({});
    setOfflineUnavailable(false);

    const runnable = models.filter((model) => available >= model.window_days);
    const skipped = models.filter((model) => available < model.window_days);
    if (requestId === predictRequestId.current && predictModeRef.current === "checkpoint") {
      setCheckpointErrors(
        Object.fromEntries(
          skipped.map((model) => [
            model.id,
            `Requires ${model.window_days} prior daily candles; ${available} are available.`,
          ]),
        ),
      );
    }
    if (!runnable.length) {
      if (requestId === predictRequestId.current && predictModeRef.current === "checkpoint") {
        setPredictLoading(false);
      }
      return;
    }

    const outcomes = await Promise.allSettled(
      runnable.map((model) =>
        api.runCheckpointPrediction({ model_id: model.id, prediction_date: predictDate, candles }),
      ),
    );
    if (requestId !== predictRequestId.current || predictModeRef.current !== "checkpoint") return;

    const nextResults: Partial<Record<api.CheckpointModelId, api.PredictResult>> = {};
    const nextErrors: Partial<Record<api.CheckpointModelId, string>> = {};
    outcomes.forEach((outcome, index) => {
      const modelId = runnable[index].id;
      if (outcome.status === "fulfilled") nextResults[modelId] = outcome.value;
      else nextErrors[modelId] = msg(outcome.reason);
    });
    setCheckpointResults(nextResults);
    setCheckpointErrors((current) => ({ ...current, ...nextErrors }));
    // Trading's one-day demo and the OOD banner key off the single canonical
    // (CM-v) result; CryptoMamba-T is comparison-only on this screen.
    setPrediction(nextResults.cmamba_v_reproduced ?? null);
    setPredictLoading(false);
  }, [data, predictDate, setup]);

  const runLive = useCallback(async () => {
    if (!apiUrl) {
      setPredictError("Enter the prediction service URL before running a live forecast.");
      return;
    }
    const requestId = ++predictRequestId.current;
    setPredictLoading(true);
    setPredictError("");
    setPrediction(null);
    try {
      const res = await api.runLivePrediction({ api_url: apiUrl, prediction_date: predictDate, risk });
      if (requestId === predictRequestId.current && predictModeRef.current === "live") {
        setPrediction(res);
      }
    } catch (e) {
      if (requestId === predictRequestId.current && predictModeRef.current === "live") {
        setPrediction(null);
        setPredictError(msg(e));
      }
    } finally {
      if (requestId === predictRequestId.current && predictModeRef.current === "live") {
        setPredictLoading(false);
      }
    }
  }, [apiUrl, predictDate, risk]);

  useEffect(() => {
    const lastDate = data?.candles.at(-1)?.date;
    if (!lastDate) return;
    setPredictDate(nextUtcDate(lastDate));
    if (predictModeRef.current === "checkpoint") {
      ++predictRequestId.current;
      setPredictLoading(false);
      setPredictError("");
      setPrediction(null);
      setCheckpointResults({});
      setCheckpointErrors({});
    }
  }, [data]);

  const loadBacktest = useCallback(
    async (override?: Partial<{ result_type: string; split: string; ref_cost: number }>) => {
      const requestId = ++backtestRequestId.current;
      setBacktestLoading(true);
      setBacktestError("");
      try {
        const res = await api.fetchBacktest({
          result_type: override?.result_type ?? btResultType,
          split: override?.split ?? btSplit,
          ref_cost: override?.ref_cost ?? btCost,
        });
        if (requestId === backtestRequestId.current) setBacktest(res);
        if (requestId === backtestRequestId.current && res.status === "READY") {
          // Keep the controls in sync with what the server actually served
          // (it snaps unknown values to the nearest available option).
          setBtResultType(res.result_type);
          setBtSplit(res.split);
          setBtCost(res.ref_cost);
        }
      } catch (e) {
        if (requestId === backtestRequestId.current) setBacktestError(msg(e));
      } finally {
        if (requestId === backtestRequestId.current) setBacktestLoading(false);
      }
    },
    [btResultType, btSplit, btCost],
  );

  const loadReplay = useCallback(
    async (override?: Partial<{ result_type: string; split: string; strategy: string; ref_cost: number }>) => {
      const requestId = ++replayRequestId.current;
      setReplayLoading(true);
      setReplayError("");
      try {
        const res = await api.fetchTradingReplay({
          result_type: override?.result_type ?? btResultType,
          split: override?.split ?? btSplit,
          strategy: override?.strategy ?? replayStrategy,
          ref_cost: override?.ref_cost ?? btCost,
        });
        if (requestId === replayRequestId.current) setReplay(res);
      } catch (e) {
        if (requestId === replayRequestId.current) setReplayError(msg(e));
      } finally {
        if (requestId === replayRequestId.current) setReplayLoading(false);
      }
    },
    [btResultType, btSplit, replayStrategy, btCost],
  );

  // Lazy-load each screen on first open. Reproduce is the exception: the sidebar
  // renders its readiness badge on every screen, so it is fetched at startup.
  const booted = useRef(false);
  useEffect(() => {
    if (booted.current) return;
    booted.current = true;
    void showPaper();
    void loadReproduce();
  }, [showPaper, loadReproduce]);

  const go = useCallback(
    (id: ScreenId) => {
      setScreen(id);
      if (id === "predict" && !setup && !predictLoading) void loadPredict();
      if (id === "trading") {
        if (!backtest && !backtestLoading) void loadBacktest();
        if (!replay && !replayLoading) void loadReplay();
      }
    },
    [
      setup,
      predictLoading,
      loadPredict,
      backtest,
      backtestLoading,
      loadBacktest,
      replay,
      replayLoading,
      loadReplay,
    ],
  );

  return useMemo(
    () => ({
      screen,
      go,
      data: {
        active: data,
        paper: paperData,
        uploaded: uploadedData,
        mode: dataMode,
        loading: dataLoading,
        error: dataError,
        windowLoading: dataWindowLoading,
        windowError: dataWindowError,
        uploadName,
        hasUpload: Boolean(uploadedData),
        showPaper,
        showUpload,
        uploadCsv,
        clearUpload,
        selectWindow: selectDataWindow,
        cancelWindowSelection: cancelDataWindowSelection,
      },
      reproduce: { data: repro, loading: reproLoading, error: reproError, reload: loadReproduce },
      predict: {
        setup,
        result: prediction,
        checkpointResults,
        checkpointErrors,
        mode: predictMode,
        loading: predictLoading,
        error: predictError,
        offlineUnavailable,
        offlineDate,
        predictDate,
        apiUrl,
        risk,
        setMode: (m: api.PredictMode) => {
          predictModeRef.current = m;
          ++predictRequestId.current;
          setPredictLoading(false);
          setPredictMode(m);
          setPredictError("");
          setPrediction(null);
          setCheckpointResults({});
          setCheckpointErrors({});
          setOfflineUnavailable(false);
          if (m === "historical") void loadOffline(offlineDate || undefined);
        },
        setOfflineDate: (d: string) => {
          setOfflineDate(d);
          void loadOffline(d);
        },
        stepOfflineDate: (delta: number) => {
          const dates = setup?.offline_dates ?? [];
          if (!dates.length) return;
          let i = dates.indexOf(offlineDate);
          if (i < 0) i = 0;
          i = Math.min(dates.length - 1, Math.max(0, i + delta));
          setOfflineDate(dates[i]);
          void loadOffline(dates[i]);
        },
        setPredictDate: (value: string) => {
          setPredictDate(value);
          if (predictModeRef.current === "checkpoint") {
            setPrediction(null);
            setCheckpointResults({});
            setCheckpointErrors({});
          }
        },
        setApiUrl,
        setRisk,
        runCheckpoint,
        runLive,
      },
      backtest: {
        data: backtest,
        loading: backtestLoading,
        error: backtestError,
        resultType: btResultType,
        split: btSplit,
        cost: btCost,
        setResultType: (v: string) => {
          setBtResultType(v);
          void loadBacktest({ result_type: v });
          void loadReplay({ result_type: v });
        },
        setSplit: (v: string) => {
          setBtSplit(v);
          void loadBacktest({ split: v });
          void loadReplay({ split: v });
        },
        setCost: (v: number) => {
          setBtCost(v);
          void loadBacktest({ ref_cost: v });
          void loadReplay({ ref_cost: v });
        },
      },
      replay: {
        data: replay,
        loading: replayLoading,
        error: replayError,
        strategy: replayStrategy,
        setStrategy: (v: string) => {
          setReplayStrategy(v);
          void loadReplay({ strategy: v });
        },
        reload: loadReplay,
      },
    }),
    [
      screen, go,
      data, paperData, dataMode, dataLoading, dataError, dataWindowLoading, dataWindowError, uploadName, uploadedData,
      showPaper, showUpload, uploadCsv, clearUpload, selectDataWindow, cancelDataWindowSelection,
      repro, reproLoading, reproError, loadReproduce,
      setup, prediction, checkpointResults, checkpointErrors, predictMode, predictLoading, predictError, offlineUnavailable, offlineDate,
      predictDate, apiUrl, risk, loadOffline, runCheckpoint, runLive,
      backtest, backtestLoading, backtestError, btResultType, btSplit, btCost, loadBacktest,
      replay, replayLoading, replayError, replayStrategy, loadReplay,
    ],
  );
}

type Store = ReturnType<typeof useConsoleStore>;

const ConsoleContext = createContext<Store | null>(null);

export function ConsoleProvider({ children }: { children: ReactNode }) {
  const store = useConsoleStore();
  return <ConsoleContext.Provider value={store}>{children}</ConsoleContext.Provider>;
}

export function useConsole(): Store {
  const store = useContext(ConsoleContext);
  if (!store) throw new Error("useConsole must be used inside <ConsoleProvider>");
  return store;
}
