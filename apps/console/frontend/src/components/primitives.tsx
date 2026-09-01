import type { ReactNode } from "react";

import { cn } from "@/lib/utils";

/* Small presentation primitives shared by every screen. They exist so screens
 * stay declarative and the visual language stays consistent across 5 pages. */

export type Tone = "green" | "amber" | "blue" | "red" | "neutral" | "brand";

const TONE: Record<Tone, { text: string; bg: string; dot: string; border: string }> = {
  green: { text: "text-[var(--signal-green)]", bg: "bg-[var(--signal-green-soft)]", dot: "bg-[var(--signal-green)]", border: "border-[var(--signal-green-border)]" },
  amber: { text: "text-[var(--signal-amber)]", bg: "bg-[var(--signal-amber-soft)]", dot: "bg-[var(--signal-amber)]", border: "border-[var(--signal-amber-border)]" },
  blue: { text: "text-[var(--signal-blue)]", bg: "bg-[var(--signal-blue-soft)]", dot: "bg-[var(--signal-blue)]", border: "border-[var(--signal-blue-border)]" },
  red: { text: "text-[var(--signal-red)]", bg: "bg-[var(--signal-red-soft)]", dot: "bg-[var(--signal-red)]", border: "border-[var(--signal-red-border)]" },
  brand: { text: "text-[var(--brand-accent)]", bg: "bg-[var(--brand-accent-soft)]", dot: "bg-[var(--brand-accent)]", border: "border-[var(--line)]" },
  neutral: { text: "text-[var(--text-secondary)]", bg: "bg-[var(--bg-subtle)]", dot: "bg-[var(--text-muted)]", border: "border-[var(--line)]" },
};

export function Pill({
  tone = "neutral",
  dot = true,
  children,
  className,
}: {
  tone?: Tone;
  dot?: boolean;
  children: ReactNode;
  className?: string;
}) {
  const t = TONE[tone];
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded px-2 py-[3px] font-mono text-[10px] font-semibold uppercase tracking-[0.07em]",
        t.bg,
        t.text,
        className,
      )}
    >
      {dot && <span className={cn("h-1.5 w-1.5 shrink-0 rounded-full", t.dot)} />}
      {children}
    </span>
  );
}

/** Pulsing status dot — used for "this reflects live state" indicators. */
export function LiveDot({ tone = "green" }: { tone?: Tone }) {
  const t = TONE[tone];
  return (
    <span className="relative inline-flex h-2 w-2 shrink-0">
      <span
        className={cn("absolute -inset-1 rounded-full opacity-30", t.dot)}
        style={{ animation: "cm-pulse 2s cubic-bezier(0.16,1,0.3,1) infinite" }}
      />
      <span className={cn("h-2 w-2 rounded-full", t.dot)} />
    </span>
  );
}

export function Panel({
  eyebrow,
  hint,
  actions,
  children,
  className,
  bodyClassName,
}: {
  eyebrow?: ReactNode;
  hint?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
  bodyClassName?: string;
}) {
  return (
    <section
      className={cn(
        "min-w-0 rounded-md border border-[var(--line)] bg-[var(--bg-card)] shadow-[var(--shadow-card)]",
        className,
      )}
    >
      {(eyebrow || actions) && (
        <header className="flex flex-wrap items-start justify-between gap-3 border-b border-[var(--line-subtle)] px-4 py-2.5">
          <div className="min-w-0">
            {eyebrow && <div className="eyebrow">{eyebrow}</div>}
            {hint && <p className="mt-1 max-w-3xl text-[11.5px] leading-relaxed text-[var(--text-muted)]">{hint}</p>}
          </div>
          {actions && <div className="flex shrink-0 items-center gap-2">{actions}</div>}
        </header>
      )}
      <div className={cn("p-4", bodyClassName)}>{children}</div>
    </section>
  );
}

export function StatTile({
  label,
  value,
  unit,
  sub,
  valueColor,
  tone,
}: {
  label: string;
  value: ReactNode;
  unit?: string;
  sub?: ReactNode;
  valueColor?: string;
  tone?: Tone;
}) {
  return (
    <div className="min-w-0 rounded-md border border-[var(--line)] bg-[var(--bg-card)] px-4 py-3 shadow-[var(--shadow-card)]">
      <div className="flex items-center gap-2">
        {tone && <span className={cn("h-1.5 w-1.5 rounded-full", TONE[tone].dot)} />}
        <span className="eyebrow truncate">{label}</span>
      </div>
      <div className="mt-2.5 flex items-baseline gap-1.5">
        <span
          className="tnum truncate text-[23px] font-semibold leading-none tracking-tight"
          style={valueColor ? { color: valueColor } : undefined}
        >
          {value}
        </span>
        {unit && <span className="text-xs text-[var(--text-muted)]">{unit}</span>}
      </div>
      {sub && <div className="mt-2 truncate font-mono text-[10px] text-[var(--text-muted)]">{sub}</div>}
    </div>
  );
}

export function PageHeader({
  title,
  description,
  actions,
}: {
  title: string;
  description?: ReactNode;
  actions?: ReactNode;
}) {
  return (
    <div className="flex flex-wrap items-start justify-between gap-5">
      <div className="max-w-3xl">
        <h1 className="text-[22px] font-semibold leading-tight tracking-tight text-[var(--text-primary)]">{title}</h1>
        {description && (
          <p className="mt-1.5 text-[12.5px] leading-relaxed text-[var(--text-secondary)]">{description}</p>
        )}
      </div>
      {actions && <div className="flex shrink-0 items-center gap-2">{actions}</div>}
    </div>
  );
}

export function Callout({
  tone = "neutral",
  title,
  children,
}: {
  tone?: Tone;
  title?: ReactNode;
  children: ReactNode;
}) {
  const t = TONE[tone];
  return (
    <div className={cn("rounded-md border px-3.5 py-2.5 text-[12px] leading-relaxed", t.bg, t.border)}>
      {title && <div className={cn("mb-1 font-semibold", t.text)}>{title}</div>}
      <div className="text-[var(--text-secondary)]">{children}</div>
    </div>
  );
}

export function ErrorBanner({ message }: { message: string }) {
  return (
    <Callout tone="red" title="Error">
      <span className="font-mono text-xs">{message}</span>
    </Callout>
  );
}

/** Definition item used in provenance / metadata grids. */
export function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="min-w-0 rounded border border-[var(--line)] bg-[var(--bg-sunken)] px-3 py-2.5">
      <div className="eyebrow text-[9.5px]">{label}</div>
      <div className="mt-1.5 break-words font-mono text-[11.5px] text-[var(--text-primary)]">{children}</div>
    </div>
  );
}

/** Right-aligned numeric table cell content. */
export function Num({ children, color }: { children: ReactNode; color?: string }) {
  return (
    <span className="tnum font-mono text-xs" style={color ? { color } : undefined}>
      {children}
    </span>
  );
}

export function toneForRoi(value: number | null | undefined): string | undefined {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return undefined;
  return Number(value) >= 0 ? "var(--signal-green)" : "var(--signal-red)";
}
