import { useEffect, useMemo, useRef, useState } from "react";
import {
  CandlestickSeries,
  ColorType,
  CrosshairMode,
  HistogramSeries,
  createChart,
  createSeriesMarkers,
  type CandlestickData,
  type HistogramData,
  type IChartApi,
  type ISeriesApi,
  type ISeriesMarkersPluginApi,
  type MouseEventHandler,
  type SeriesMarker,
  type Time,
} from "lightweight-charts";

export interface DataExplorerCandle {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  split: string;
}

export interface DataExplorerChartProps {
  candles: DataExplorerCandle[];
  selectedDate?: string | null;
  windowStart?: string | null;
  windowEnd?: string | null;
  focusRange?: { from: string; to: string } | null;
  onSelectDate: (date: string) => void;
  height?: number;
}

type OrderedMarker = SeriesMarker<Time> & { order: number };

const asTime = (date: string) => date as Time;
const EDGE_PADDING_RATIO = 0.08;

function timeToDate(time: Time): string {
  if (typeof time === "string") return time;
  if (typeof time === "number") return new Date(time * 1_000).toISOString().slice(0, 10);
  return `${time.year}-${String(time.month).padStart(2, "0")}-${String(time.day).padStart(2, "0")}`;
}

function splitLabel(split: string): string {
  const normalized = split.trim().toLowerCase();
  if (normalized === "train") return "TRAIN";
  if (normalized === "validation" || normalized === "val") return "VALIDATION";
  if (normalized === "test") return "TEST";
  return split.trim().toUpperCase();
}

function displayDate(date: string): string {
  const [year, month, day] = date.split("-");
  return `${day}/${month}/${year}`;
}

function displayPrice(value: number): string {
  return value.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function displayVolume(value: number): string {
  return `${(value / 1_000_000_000).toLocaleString("en-US", { maximumFractionDigits: 2 })}B`;
}

export function DataExplorerChart({
  candles,
  selectedDate = null,
  windowStart = null,
  windowEnd = null,
  focusRange = null,
  onSelectDate,
  height = 520,
}: DataExplorerChartProps) {
  const host = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candleRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
  const volumeRef = useRef<ISeriesApi<"Histogram"> | null>(null);
  const markersRef = useRef<ISeriesMarkersPluginApi<Time> | null>(null);
  const candlesRef = useRef(candles);
  const selectRef = useRef(onSelectDate);
  const selectedDateRef = useRef(selectedDate);
  const windowEndRef = useRef(windowEnd);
  const [inspectedCandle, setInspectedCandle] = useState<DataExplorerCandle | null>(candles.at(-1) ?? null);

  candlesRef.current = candles;
  selectRef.current = onSelectDate;
  selectedDateRef.current = selectedDate;
  windowEndRef.current = windowEnd;

  const candleDates = useMemo(() => new Set(candles.map((candle) => candle.date)), [candles]);
  const candleDatesRef = useRef(candleDates);
  candleDatesRef.current = candleDates;

  const candleData = useMemo<CandlestickData<Time>[]>(
    () =>
      candles.map((candle) => {
        const isModelInput =
          windowStart !== null && windowEnd !== null && candle.date >= windowStart && candle.date <= windowEnd;
        const rising = candle.close >= candle.open;
        return {
          time: asTime(candle.date),
          open: candle.open,
          high: candle.high,
          low: candle.low,
          close: candle.close,
          color: rising ? (isModelInput ? "#16834f" : "#70a991") : isModelInput ? "#c33b32" : "#c9908a",
          borderColor: isModelInput ? "#d18b17" : rising ? "#4f9277" : "#b97972",
          wickColor: isModelInput ? "#d18b17" : rising ? "#4f9277" : "#b97972",
        };
      }),
    [candles, windowEnd, windowStart],
  );

  const volumeData = useMemo<HistogramData<Time>[]>(
    () =>
      candles.map((candle) => {
        const isModelInput =
          windowStart !== null && windowEnd !== null && candle.date >= windowStart && candle.date <= windowEnd;
        return {
          time: asTime(candle.date),
          value: candle.volume,
          color:
            candle.close >= candle.open
              ? isModelInput
                ? "rgba(22,131,79,0.55)"
                : "rgba(79,146,119,0.28)"
              : isModelInput
                ? "rgba(195,59,50,0.55)"
                : "rgba(185,121,114,0.28)",
        };
      }),
    [candles, windowEnd, windowStart],
  );

  const splitBoundaryMarkers = useMemo<OrderedMarker[]>(() => {
    const next: OrderedMarker[] = [];
    let previousSplit: string | null = null;

    for (const candle of candles) {
      if (candle.split !== previousSplit) {
        next.push({
          time: asTime(candle.date),
          position: "aboveBar",
          color: "#9a7cff",
          shape: "square",
          text: splitLabel(candle.split),
          size: 0.7,
          order: 0,
        });
        previousSplit = candle.split;
      }
    }

    return next;
  }, [candles]);

  const windowMarkers = useMemo<OrderedMarker[]>(() => {
    const next: OrderedMarker[] = [];
    if (windowStart && candleDates.has(windowStart)) {
      next.push({
        time: asTime(windowStart),
        position: "belowBar",
        color: "#d18b17",
        shape: "arrowUp",
        text: "DAY 1",
        size: 0.8,
        order: 1,
      });
    }
    if (windowEnd && candleDates.has(windowEnd)) {
      next.push({
        time: asTime(windowEnd),
        position: "belowBar",
        color: "#d18b17",
        shape: "arrowUp",
        text: "DAY 14",
        size: 0.8,
        order: 2,
      });
    }

    return next;
  }, [candleDates, windowEnd, windowStart]);

  const selectedMarker = useMemo<OrderedMarker[]>(() => {
    if (selectedDate && candleDates.has(selectedDate)) {
      return [
        {
          time: asTime(selectedDate),
          position: "aboveBar",
          color: "#4d8dff",
          shape: "circle",
          text: "TARGET",
          size: 1,
          order: 3,
        },
      ];
    }

    return [];
  }, [candleDates, selectedDate]);

  const markers = useMemo<SeriesMarker<Time>[]>(
    () =>
      [...splitBoundaryMarkers, ...windowMarkers, ...selectedMarker]
        .sort((left, right) => {
          const dateOrder = String(left.time).localeCompare(String(right.time));
          return dateOrder === 0 ? left.order - right.order : dateOrder;
        })
        .map(({ order: _order, ...marker }) => marker),
    [selectedMarker, splitBoundaryMarkers, windowMarkers],
  );

  useEffect(() => {
    const element = host.current;
    if (!element) return;

    const chart = createChart(element, {
      autoSize: true,
      height,
      layout: {
        background: { type: ColorType.Solid, color: "#111923" },
        textColor: "#8795a8",
        fontFamily: '"JetBrains Mono", ui-monospace, monospace',
        attributionLogo: true,
      },
      grid: {
        vertLines: { color: "rgba(174,185,199,0.06)" },
        horzLines: { color: "rgba(174,185,199,0.08)" },
      },
      crosshair: { mode: CrosshairMode.MagnetOHLC },
      handleScroll: {
        mouseWheel: true,
        pressedMouseMove: true,
        horzTouchDrag: true,
        vertTouchDrag: false,
      },
      handleScale: {
        axisPressedMouseMove: true,
        mouseWheel: true,
        pinch: true,
      },
      rightPriceScale: { borderColor: "rgba(174,185,199,0.18)" },
      timeScale: {
        borderColor: "rgba(174,185,199,0.18)",
        timeVisible: false,
        secondsVisible: false,
        rightOffset: 3,
        minBarSpacing: 0.1,
      },
    });
    const candleSeries = chart.addSeries(CandlestickSeries, {
      title: "BTC/USD OHLC",
      upColor: "#16834f",
      downColor: "#c33b32",
      borderVisible: true,
      wickUpColor: "#16834f",
      wickDownColor: "#c33b32",
    });
    const volumeSeries = chart.addSeries(HistogramSeries, {
      title: "Volume",
      priceFormat: { type: "volume" },
      priceScaleId: "",
      priceLineVisible: false,
      lastValueVisible: false,
    });
    volumeSeries.priceScale().applyOptions({
      scaleMargins: { top: 0.8, bottom: 0 },
    });
    const markerApi = createSeriesMarkers(candleSeries, []);

    const clickHandler: MouseEventHandler<Time> = (event) => {
      if (event.time === undefined) return;
      const date = timeToDate(event.time);
      if (candleDatesRef.current.has(date)) {
        selectRef.current(date);
      }
    };
    const crosshairHandler: MouseEventHandler<Time> = (event) => {
      const available = candlesRef.current;
      if (event.time === undefined) {
        setInspectedCandle(available.at(-1) ?? null);
        return;
      }
      const date = timeToDate(event.time);
      setInspectedCandle(available.find((candle) => candle.date === date) ?? available.at(-1) ?? null);
    };
    chart.subscribeClick(clickHandler);
    chart.subscribeCrosshairMove(crosshairHandler);

    chartRef.current = chart;
    candleRef.current = candleSeries;
    volumeRef.current = volumeSeries;
    markersRef.current = markerApi;

    return () => {
      chart.unsubscribeClick(clickHandler);
      chart.unsubscribeCrosshairMove(crosshairHandler);
      markerApi.detach();
      chart.remove();
      chartRef.current = null;
      candleRef.current = null;
      volumeRef.current = null;
      markersRef.current = null;
    };
  }, [height]);

  useEffect(() => {
    setInspectedCandle(candles.at(-1) ?? null);
  }, [candles]);

  useEffect(() => {
    candleRef.current?.setData(candleData);
    volumeRef.current?.setData(volumeData);
  }, [candleData, volumeData]);

  useEffect(() => {
    const chart = chartRef.current;
    if (!chart || candles.length === 0 || focusRange) return;
    const timeScale = chart.timeScale();
    const EDGE_PADDING_BARS = Math.max(12, candles.length * EDGE_PADDING_RATIO);
    timeScale.setVisibleLogicalRange({
      from: -EDGE_PADDING_BARS,
      to: candles.length - 1 + EDGE_PADDING_BARS,
    });
  }, [candles, focusRange]);

  useEffect(() => {
    markersRef.current?.setMarkers(markers);
  }, [markers]);

  useEffect(() => {
    const chart = chartRef.current;
    if (!chart || !focusRange || candleData.length === 0) return;
    chart.timeScale().setVisibleRange({
      from: asTime(focusRange.from),
      to: asTime(focusRange.to),
    });
  }, [candleData.length, focusRange]);

  const selectAdjacent = (direction: -1 | 1) => {
    const available = candlesRef.current;
    if (available.length === 0) return;
    const selected = selectedDateRef.current;
    if (!selected && windowEndRef.current) {
      if (direction < 0) selectRef.current(windowEndRef.current);
      return;
    }
    const anchor = selected ?? windowEndRef.current;
    const current = anchor ? available.findIndex((candle) => candle.date === anchor) : -1;
    const fallback = direction > 0 ? 0 : available.length - 1;
    const next = current < 0 ? fallback : Math.min(available.length - 1, Math.max(0, current + direction));
    selectRef.current(available[next].date);
  };

  return (
    <figure className="w-full" aria-label="BTC/USD data explorer">
      {inspectedCandle && (
        <div
          className="flex flex-wrap items-center gap-x-4 gap-y-1 border-b border-[var(--border-subtle)] px-3 py-2 font-mono text-[11px]"
          aria-label="Inspected OHLCV candle"
        >
          <span className="font-semibold text-[var(--text-primary)]">{displayDate(inspectedCandle.date)}</span>
          <span><span className="text-[var(--text-muted)]">Open</span> {displayPrice(inspectedCandle.open)}</span>
          <span><span className="text-[var(--text-muted)]">High</span> {displayPrice(inspectedCandle.high)}</span>
          <span><span className="text-[var(--text-muted)]">Low</span> {displayPrice(inspectedCandle.low)}</span>
          <span><span className="text-[var(--text-muted)]">Close</span> {displayPrice(inspectedCandle.close)}</span>
          <span><span className="text-[var(--text-muted)]">Volume</span> {displayVolume(inspectedCandle.volume)}</span>
        </div>
      )}
      <div
        ref={host}
        className="w-full rounded-md outline-none focus-visible:ring-2 focus-visible:ring-[var(--brand-accent)] focus-visible:ring-offset-2"
        style={{ height }}
        role="group"
        tabIndex={0}
        aria-label="BTC/USD price chart. Select a target date with a click or the arrow keys."
        onKeyDown={(event) => {
          if (event.key === "ArrowLeft") {
            event.preventDefault();
            selectAdjacent(-1);
          } else if (event.key === "ArrowRight") {
            event.preventDefault();
            selectAdjacent(1);
          }
        }}
      />
      <figcaption className="sr-only">
        Gold borders mark the 14-day input. Purple markers show dataset splits; blue marks the target date.
      </figcaption>
    </figure>
  );
}
