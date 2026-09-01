import { useEffect, useMemo, useRef } from "react";
import {
  AreaSeries,
  CandlestickSeries,
  ColorType,
  CrosshairMode,
  LineSeries,
  createChart,
  createSeriesMarkers,
  type CandlestickData,
  type IChartApi,
  type ISeriesApi,
  type ISeriesMarkersPluginApi,
  type LineData,
  type MouseEventHandler,
  type SeriesMarker,
  type Time,
} from "lightweight-charts";

import type { TradingReplayResponse, TradingReplayRow } from "@/lib/api";

export type ReplayPhase = "context" | "decision" | "outcome";

interface ReplayChartProps {
  candles: TradingReplayResponse["candles"];
  rows: TradingReplayRow[];
  cursor: number;
  phase: ReplayPhase;
  onSelectStep: (step: number) => void;
}

const asTime = (value: string) => value as Time;

export function ReplayChart({ candles, rows, cursor, phase, onSelectStep }: ReplayChartProps) {
  const host = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candleRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
  const forecastRef = useRef<ISeriesApi<"Line"> | null>(null);
  const equityRef = useRef<ISeriesApi<"Area"> | null>(null);
  const markersRef = useRef<ISeriesMarkersPluginApi<Time> | null>(null);
  const selectRef = useRef(onSelectStep);
  selectRef.current = onSelectStep;

  const selected = rows[cursor];
  const visibleDecisionCount = phase === "context" ? cursor : cursor + 1;
  const candleCutoff = phase === "outcome" ? selected?.outcome_date : selected?.decision_date;

  const candleData = useMemo<CandlestickData<Time>[]>(
    () =>
      candles
        .filter((candle) => !candleCutoff || candle.date <= candleCutoff)
        .map((candle) => ({
          time: asTime(candle.date),
          open: candle.open,
          high: candle.high,
          low: candle.low,
          close: candle.close,
        })),
    [candles, candleCutoff],
  );
  const forecastData = useMemo<LineData<Time>[]>(
    () =>
      rows.slice(0, visibleDecisionCount).map((row) => ({
        time: asTime(row.outcome_date),
        value: row.predicted_close,
      })),
    [rows, visibleDecisionCount],
  );
  const equityData = useMemo<LineData<Time>[]>(
    () =>
      rows.slice(0, visibleDecisionCount).map((row) => ({
        time: asTime(row.decision_date),
        value: row.portfolio_value,
      })),
    [rows, visibleDecisionCount],
  );
  const markers = useMemo<SeriesMarker<Time>[]>(() => {
    const trades: SeriesMarker<Time>[] = [];
    for (const row of rows.slice(0, visibleDecisionCount)) {
      if (row.action === "BUY") {
        trades.push({
          time: asTime(row.decision_date),
          position: "belowBar",
          color: "#16834f",
          shape: "arrowUp",
          text: "BUY",
        });
      } else if (row.action === "SELL") {
        trades.push({
          time: asTime(row.decision_date),
          position: "aboveBar",
          color: "#c33b32",
          shape: "arrowDown",
          text: "SELL",
        });
      }
    }
    if (selected && phase === "outcome") {
      trades.push({
        time: asTime(selected.outcome_date),
        position: "aboveBar",
        color: "#4d8dff",
        shape: "circle",
        text: "T+1",
      });
    }
    return trades;
  }, [rows, selected, phase, visibleDecisionCount]);

  useEffect(() => {
    const element = host.current;
    if (!element) return;

    const chart = createChart(element, {
      autoSize: true,
      height: 500,
      layout: {
        background: { type: ColorType.Solid, color: "#111923" },
        textColor: "#8795a8",
        fontFamily: '"JetBrains Mono", ui-monospace, monospace',
        attributionLogo: true,
        panes: {
          separatorColor: "rgba(174,185,199,0.12)",
          separatorHoverColor: "rgba(77,141,255,0.35)",
          enableResize: true,
        },
      },
      grid: {
        vertLines: { color: "rgba(174,185,199,0.06)" },
        horzLines: { color: "rgba(174,185,199,0.08)" },
      },
      crosshair: { mode: CrosshairMode.MagnetOHLC },
      rightPriceScale: { borderColor: "rgba(174,185,199,0.18)" },
      timeScale: {
        borderColor: "rgba(174,185,199,0.18)",
        timeVisible: false,
        secondsVisible: false,
        rightOffset: 3,
      },
    });
    const candleSeries = chart.addSeries(CandlestickSeries, {
      title: "BTC/USD",
      upColor: "#16834f",
      downColor: "#c33b32",
      borderVisible: false,
      wickUpColor: "#16834f",
      wickDownColor: "#c33b32",
    });
    const forecastSeries = chart.addSeries(LineSeries, {
      title: "CryptoMamba-v forecast",
      color: "#4d8dff",
      lineWidth: 2,
      priceLineVisible: false,
      lastValueVisible: true,
    });
    const equitySeries = chart.addSeries(
      AreaSeries,
      {
        title: "Portfolio @ decision close",
        lineColor: "#9a7cff",
        topColor: "rgba(118,86,197,0.30)",
        bottomColor: "rgba(118,86,197,0.02)",
        lineWidth: 2,
        priceLineVisible: false,
      },
      1,
    );
    const markerApi = createSeriesMarkers(candleSeries, []);

    const clickHandler: MouseEventHandler<Time> = (event) => {
      if (event.time === undefined) return;
      const date = String(event.time);
      const step = rows.findIndex((row) => row.outcome_date === date || row.decision_date === date);
      if (step >= 0) selectRef.current(step);
    };
    chart.subscribeClick(clickHandler);

    chartRef.current = chart;
    candleRef.current = candleSeries;
    forecastRef.current = forecastSeries;
    equityRef.current = equitySeries;
    markersRef.current = markerApi;

    return () => {
      chart.unsubscribeClick(clickHandler);
      markerApi.detach();
      chart.remove();
      chartRef.current = null;
      candleRef.current = null;
      forecastRef.current = null;
      equityRef.current = null;
      markersRef.current = null;
    };
  }, [rows]);

  useEffect(() => {
    candleRef.current?.setData(candleData);
    forecastRef.current?.setData(forecastData);
    equityRef.current?.setData(equityData);
    markersRef.current?.setMarkers(markers);

    const chart = chartRef.current;
    if (!chart || !selected || candleData.length === 0) return;
    const from = candleData[Math.max(0, candleData.length - 75)].time;
    const to = asTime(phase === "context" ? selected.decision_date : selected.outcome_date);
    chart.timeScale().setVisibleRange({ from, to });
  }, [candleData, equityData, forecastData, markers, phase, selected]);

  return (
    <div
      ref={host}
      className="h-[500px] w-full"
      role="img"
      aria-label="BTC replay chart with historical candles, forecast, trade signals and portfolio value"
    />
  );
}
