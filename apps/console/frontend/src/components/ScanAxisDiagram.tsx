/* Side-by-side view of the one change CMamba-T makes: which axis the selective
 * scan runs along. Everything drawn here is structure, not measurement — the
 * parameter counts are passed in from the artifact-backed API. */

const FEATURES = ["Timestamp", "Open", "High", "Low", "Close", "Volume"];

function Block({
  title,
  sub,
  tone = "block",
}: {
  title: string;
  sub?: string;
  tone?: "block" | "head" | "out";
}) {
  const style =
    tone === "head"
      ? "border-[#b79a34] bg-[#3a3213]"
      : tone === "out"
        ? "border-[#6a9a3a] bg-[#1d2c15]"
        : "border-[#b24a45] bg-[#3a1f1e]";
  return (
    <div className={`rounded-md border px-3 py-2 text-center ${style}`}>
      <div className="text-[12px] font-semibold text-[var(--text-primary)]">{title}</div>
      {sub && <div className="mt-0.5 font-mono text-[10px] text-[var(--text-muted)]">{sub}</div>}
    </div>
  );
}

function Arrow() {
  return <div className="mx-auto h-4 w-px bg-[var(--text-muted)]" aria-hidden />;
}

/** CryptoMamba-v: 6 feature rows, the scan runs DOWN the feature axis. */
function FeatureAxisGrid() {
  return (
    <svg viewBox="0 0 300 104" className="w-full" role="img"
         aria-label="Input tensor with six feature rows; the scan runs down the feature axis">
      {FEATURES.map((name, r) => (
        <g key={name}>
          <text x="66" y={17 + r * 16} textAnchor="end" fontSize="8.5" fill="#8a8e99">{name}</text>
          {Array.from({ length: 14 }).map((_, c) => (
            <rect key={c} x={74 + c * 15} y={7 + r * 16} width="14" height="13"
                  fill={name === "Close" ? "#3a1f1e" : "#1b2431"}
                  stroke={name === "Close" ? "#b24a45" : "#31405a"} strokeWidth="0.8" />
          ))}
        </g>
      ))}
      <line x1="292" y1="10" x2="292" y2="100" stroke="#b24a45" strokeWidth="2"
            markerEnd="url(#arrowDown)" />
      <defs>
        <marker id="arrowDown" markerWidth="7" markerHeight="7" refX="3.5" refY="6"
                orient="auto"><path d="M0,0 L7,0 L3.5,7 z" fill="#b24a45" /></marker>
        <marker id="arrowRight" markerWidth="7" markerHeight="7" refX="6" refY="3.5"
                orient="auto"><path d="M0,0 L0,7 L7,3.5 z" fill="#b24a45" /></marker>
      </defs>
    </svg>
  );
}

/** CMamba-T: one token per day, the scan runs ALONG the time axis. */
function TimeAxisGrid() {
  return (
    <svg viewBox="0 0 300 104" className="w-full" role="img"
         aria-label="Input tensor with sixty daily tokens; the scan runs along the time axis">
      {["O", "H", "L", "C", "V"].map((name, r) => (
        <g key={name}>
          <text x="30" y={17 + r * 16} textAnchor="end" fontSize="8.5" fill="#8a8e99">{name}</text>
          {Array.from({ length: 60 }).map((_, c) => (
            <rect key={c} x={36 + c * 4.2} y={7 + r * 16} width="3.6" height="13"
                  fill={c === 59 ? "#3a1f1e" : "#1b2431"}
                  stroke={c === 59 ? "#b24a45" : "#31405a"} strokeWidth="0.45" />
          ))}
        </g>
      ))}
      <line x1="36" y1="96" x2="288" y2="96" stroke="#b24a45" strokeWidth="2"
            markerEnd="url(#arrowRight2)" />
      <defs>
        <marker id="arrowRight2" markerWidth="7" markerHeight="7" refX="6" refY="3.5"
                orient="auto"><path d="M0,0 L0,7 L7,3.5 z" fill="#b24a45" /></marker>
      </defs>
    </svg>
  );
}

export function ScanAxisDiagram({
  paramsOriginal,
  paramsProposed,
}: {
  paramsOriginal?: number;
  paramsProposed?: number;
}) {
  const n = (v?: number) => (v ? v.toLocaleString("en-US") : "—");
  return (
    <div className="grid gap-5 lg:grid-cols-2">
      <div className="min-w-0 space-y-2">
        <div className="flex items-baseline justify-between gap-2">
          <h3 className="text-[13px] font-semibold text-[var(--text-primary)]">CryptoMamba-v (original)</h3>
          <span className="tnum font-mono text-[11px] text-[var(--text-muted)]">{n(paramsOriginal)} params</span>
        </div>
        <FeatureAxisGrid />
        <p className="text-[11px] font-medium text-[#c9635c]">
          Scan runs down the 6 feature tokens · the 14-day window is the model width
        </p>
        <Block title="12 × CMBlock — 3 stages" sub="d: 14 → 16 → 32 → 1" />
        <Arrow />
        <Block title="Permute + Linear 6 → 1" sub="mixes the 6 features" tone="head" />
        <Arrow />
        <Block title="ŷ — next-day Close price" tone="out" />
        <p className="text-[11px] leading-relaxed text-[var(--text-muted)]">
          The window length <em>is</em> the model width, so looking further back inflates every stage.
        </p>
      </div>

      <div className="min-w-0 space-y-2">
        <div className="flex items-baseline justify-between gap-2">
          <h3 className="text-[13px] font-semibold text-[var(--text-primary)]">CMamba-T / S5-Full</h3>
          <span className="tnum font-mono text-[11px] text-[var(--text-muted)]">{n(paramsProposed)} params</span>
        </div>
        <TimeAxisGrid />
        <p className="text-[11px] font-medium text-[#c9635c]">
          Causal scan runs along the 60 day tokens · window is independent of model width
        </p>
        <Block title="Embed Linear 5 → 32" sub="one token per day · drop Timestamp · log1p(V/1e9)" />
        <Arrow />
        <Block title="4 × CMBlock" sub="d=32 · d_state=16 · MLP×2" />
        <Arrow />
        <Block title="last token → LayerNorm → Linear 32 → 1" tone="head" />
        <Arrow />
        <Block title="ŷ — relative return, rebuilt around the last Close" tone="out" />
        <p className="text-[11px] leading-relaxed text-[var(--text-muted)]">
          The window is the scan length, so 14 → 60 days costs zero extra parameters.
        </p>
      </div>
    </div>
  );
}
