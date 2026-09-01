import { useEffect, useMemo, useRef } from "react";
import Plotly from "plotly.js-dist-min";

import type { PlotlyFigure } from "@/lib/api";

/* The server ships full Plotly figure JSON produced by the validated charts.py.
 * We keep its model/strategy colours and only apply the console's dark surface,
 * axes and hover treatment. */

const CONFIG = {
  displayModeBar: false,
  displaylogo: false,
  responsive: false, // width is driven explicitly, see themed()
  scrollZoom: false,
  doubleClick: false,
  showTips: false,
};

const AXIS = {
  gridcolor: "rgba(174,185,199,0.09)",
  zerolinecolor: "rgba(174,185,199,0.20)",
  linecolor: "rgba(174,185,199,0.22)",
  tickfont: { color: "#8795a8", size: 10 },
  title: { font: { color: "#aeb9c7", size: 10.5 } },
};

function themed(
  figure: PlotlyFigure,
  height: number,
  width: number,
  leftMargin: number,
  rightMargin: number,
): Record<string, unknown> {
  const source = (figure.layout ?? {}) as Record<string, unknown>;
  const layout: Record<string, unknown> = {
    ...source,
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: "#aeb9c7", family: '"JetBrains Mono", ui-monospace, monospace', size: 10.5 },
    margin: { l: leftMargin, r: rightMargin, t: 10, b: 38 },
    legend: {
      ...((source.legend as Record<string, unknown>) ?? {}),
      font: { color: "#aeb9c7", size: 10.5 },
      bgcolor: "rgba(0,0,0,0)",
    },
    hoverlabel: {
      bgcolor: "#18222e",
      bordercolor: "#3a4b60",
      font: { color: "#edf2f7", family: '"JetBrains Mono", ui-monospace, monospace', size: 11 },
    },
    // Cards already carry a header; the figure's own title would duplicate it.
    title: "",
    autosize: false,
  };
  layout.xaxis = { ...AXIS, ...((source.xaxis as Record<string, unknown>) ?? {}) };
  layout.yaxis = { ...AXIS, ...((source.yaxis as Record<string, unknown>) ?? {}) };
  // Width is driven explicitly from the measured container. Plotly's autosize
  // measures unreliably inside CSS grid/flex cards (it falls back to its 700px
  // default), which overflows the column on narrow viewports.
  layout.width = width;
  layout.height = (source.height as number) ?? height;
  return layout;
}

interface PlotProps {
  figure?: PlotlyFigure;
  /** Fallback height when the server figure does not carry one. */
  height?: number;
  /** Message shown when the backend omitted this (optional) chart. */
  emptyLabel?: string;
  /** Optional room for long category labels or outside value labels. */
  leftMargin?: number;
  rightMargin?: number;
  /** Accessible name and current-value summary for the otherwise visual Plotly surface. */
  ariaLabel?: string;
  accessibleSummary?: string;
}

export function Plot({
  figure,
  height = 320,
  emptyLabel = "Chart not available.",
  leftMargin = 54,
  rightMargin = 16,
  ariaLabel,
  accessibleSummary,
}: PlotProps) {
  const ref = useRef<HTMLDivElement>(null);
  const data = useMemo(() => (figure?.data ? (figure.data as unknown[]) : undefined), [figure]);

  useEffect(() => {
    const el = ref.current;
    if (!el || !data || !figure) return;

    let disposed = false;
    const draw = () =>
      Plotly.react(el, data, themed(figure, height, el.clientWidth, leftMargin, rightMargin), CONFIG);
    void draw();

    // Self-heal sizing: the card may still be laying out (or hidden in an
    // inactive tab) on first draw, so redraw at the real width whenever the
    // container's size actually changes.
    let last = el.clientWidth;
    const redrawIfResized = () => {
      if (disposed) return;
      const next = el.clientWidth;
      if (next > 0 && next !== last) {
        last = next;
        void draw();
      }
    };
    const observer = new ResizeObserver(() => {
      if (el.offsetParent !== null) redrawIfResized();
    });
    observer.observe(el);

    // ResizeObserver callbacks are throttled while the tab is in the
    // background, so also redraw on the plain window resize event.
    window.addEventListener("resize", redrawIfResized);

    return () => {
      disposed = true;
      observer.disconnect();
      window.removeEventListener("resize", redrawIfResized);
      Plotly.purge(el);
    };
  }, [data, figure, height, leftMargin, rightMargin]);

  if (!data) {
    return (
      <div
        className="flex items-center justify-center rounded border border-dashed border-[var(--line)] text-xs text-[var(--text-muted)]"
        style={{ height }}
      >
        {emptyLabel}
      </div>
    );
  }

  return (
    <div className="w-full">
      <div
        ref={ref}
        className="w-full"
        style={{ minHeight: height }}
        role={ariaLabel ? "img" : undefined}
        aria-label={ariaLabel}
      />
      {accessibleSummary && <p className="sr-only">{accessibleSummary}</p>}
    </div>
  );
}
