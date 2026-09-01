import { useEffect, useMemo, useRef } from "react";
import {
  CandlestickSeries,
  ColorType,
  CrosshairMode,
  LineSeries,
  LineStyle,
  createChart,
  type CandlestickData,
  type IChartApi,
  type ISeriesApi,
  type LineData,
  type Time,
} from "lightweight-charts";

import type { PredictCandle } from "@/lib/api";
import { usd } from "@/lib/format";

interface ForecastChartProps {
  candles: PredictCandle[];
  predictionDate: string;
  predictedClose: number;
  actualClose?: number;
  variantLabel: string;
}

const asTime = (value: string) => value as Time;

export function ForecastChart({
  candles,
  predictionDate,
  predictedClose,
  actualClose,
  variantLabel,
}: ForecastChartProps) {
  const host = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candleRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
  const forecastRef = useRef<ISeriesApi<"Line"> | null>(null);
  const actualRef = useRef<ISeriesApi<"Line"> | null>(null);

  const candleData = useMemo<CandlestickData<Time>[]>(
    () =>
      candles.map((candle) => ({
        time: asTime(candle.date),
        open: candle.open,
        high: candle.high,
        low: candle.low,
        close: candle.close,
      })),
    [candles],
  );
  const lastCandle = candles[candles.length - 1];
  const forecastData = useMemo<LineData<Time>[]>(
    () =>
      lastCandle
        ? [
            { time: asTime(lastCandle.date), value: lastCandle.close },
            { time: asTime(predictionDate), value: predictedClose },
          ]
        : [],
    [lastCandle, predictedClose, predictionDate],
  );
  const actualData = useMemo<LineData<Time>[]>(
    () =>
      lastCandle && actualClose != null
        ? [
            { time: asTime(lastCandle.date), value: lastCandle.close },
            { time: asTime(predictionDate), value: actualClose },
          ]
        : [],
    [actualClose, lastCandle, predictionDate],
  );

  useEffect(() => {
    const element = host.current;
    if (!element) return;

    const chart = createChart(element, {
      autoSize: true,
      height: 420,
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
      handleScroll: { mouseWheel: true, pressedMouseMove: true },
      handleScale: { axisPressedMouseMove: true, mouseWheel: true, pinch: true },
      rightPriceScale: { borderColor: "rgba(174,185,199,0.18)" },
      timeScale: {
        borderColor: "rgba(174,185,199,0.18)",
        rightOffset: 2,
        fixLeftEdge: true,
        timeVisible: false,
        secondsVisible: false,
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
      title: "Forecast",
      color: "#4d8dff",
      lineWidth: 3,
      priceLineVisible: false,
      lastValueVisible: true,
    });
    const actualSeries = chart.addSeries(LineSeries, {
      title: "Observed close",
      color: "#9a7cff",
      lineWidth: 2,
      lineStyle: LineStyle.Dashed,
      priceLineVisible: false,
      lastValueVisible: true,
    });

    chartRef.current = chart;
    candleRef.current = candleSeries;
    forecastRef.current = forecastSeries;
    actualRef.current = actualSeries;

    return () => {
      chart.remove();
      chartRef.current = null;
      candleRef.current = null;
      forecastRef.current = null;
      actualRef.current = null;
    };
  }, []);

  useEffect(() => {
    candleRef.current?.setData(candleData);
    forecastRef.current?.applyOptions({ title: variantLabel });
    forecastRef.current?.setData(forecastData);
    actualRef.current?.setData(actualData);
    chartRef.current?.timeScale().fitContent();
  }, [actualData, candleData, forecastData, variantLabel]);

  return (
    <figure className="w-full">
      <div
        ref={host}
        className="h-[420px] w-full"
        role="img"
        aria-label={`${variantLabel} forecast for ${predictionDate}: ${usd(predictedClose)}${actualClose != null ? `; observed close ${usd(actualClose)}` : ""}`}
      />
      <figcaption className="sr-only">
        {`${candles.length} daily Bitcoin candles followed by the selected next-day forecast${
          actualClose != null ? " and the observed historical close" : ""
        }.`}
      </figcaption>
    </figure>
  );
}
