/* Display formatting. Values are never rounded before they reach here — the
 * backend is the source of truth; this only controls how many digits show. */

const EM_DASH = "—";

export function fmt(value: unknown, digits = 2): string {
  if (value === null || value === undefined || value === "") return EM_DASH;
  const n = Number(value);
  if (Number.isNaN(n)) return String(value);
  return n.toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

export function pct(value: unknown, digits = 2): string {
  if (value === null || value === undefined || value === "") return EM_DASH;
  return `${fmt(value, digits)}%`;
}

export function signedPct(value: unknown, digits = 2): string {
  if (value === null || value === undefined || value === "") return EM_DASH;
  const n = Number(value);
  if (Number.isNaN(n)) return String(value);
  // Sign comes from the *rounded* text, so -0.0008 at 2 digits reads "-0.00%",
  // never "+-0.00%".
  const text = fmt(n, digits);
  return `${text.startsWith("-") ? "" : "+"}${text}%`;
}

export function usd(value: unknown, digits = 2): string {
  if (value === null || value === undefined || value === "") return EM_DASH;
  const n = Number(value);
  if (Number.isNaN(n)) return String(value);
  return `$${fmt(n, digits)}`;
}

/** Human label for a model/strategy key coming from the artifacts. */
export function prettyModel(key: string): string {
  const map: Record<string, string> = {
    naive_persistence: "Naive persistence",
    cmamba_v_reproduced: "Reproduced CM-v",
    s5_full: "CMamba-T / S5-Full",
    lstm: "LSTM",
    gru: "GRU",
    itransformer: "iTransformer",
    "CryptoMamba-v": "CryptoMamba-v",
    buy_hold: "Buy & Hold",
    vanilla: "Vanilla",
    smart: "Smart",
    smart_w_short: "Extended Smart",
    official_checkpoint: "Official checkpoint",
    retrained_checkpoint: "Reproduced checkpoint",
  };
  return map[key] ?? key;
}
