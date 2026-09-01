import { ConsoleProvider, SCREENS, useConsole, type ScreenId } from "@/store";
import { cn } from "@/lib/utils";
import { DataScreen } from "@/screens/DataScreen";
import { PredictScreen } from "@/screens/PredictScreen";
import { ReproduceScreen } from "@/screens/ReproduceScreen";
import { ArchitectureScreen } from "@/screens/ArchitectureScreen";
import { TradingScreen } from "@/screens/TradingScreen";

function Brand() {
  return (
    <div className="flex items-center gap-2.5 px-2 pb-5">
      <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-[10px] bg-[var(--brand-accent)] text-base font-bold text-[var(--text-primary)] shadow-none">
        λ
      </span>
      <span className="flex min-w-0 flex-col leading-tight">
        <span className="truncate text-[13px] font-semibold tracking-[0.04em] text-[var(--text-primary)]">
          CRYPTOMAMBA
        </span>
        <span className="truncate font-mono text-[9px] tracking-[0.2em] text-[var(--text-muted)]">
          BTC FORECAST · v
        </span>
      </span>
    </div>
  );
}

function Sidebar() {
  const { screen, go } = useConsole();

  return (
    <aside className="sticky top-0 flex h-screen w-[228px] shrink-0 flex-col border-r border-[var(--line)] bg-[var(--bg-chrome)] px-3 py-4">
      <Brand />

      <div className="eyebrow px-3 pb-2.5">Workflow</div>
      <nav className="flex flex-col gap-0.5">
        {SCREENS.map((s) => {
          const active = s.id === screen;
          return (
            <button
              key={s.id}
              onClick={() => go(s.id as ScreenId)}
              aria-label={`${s.no} ${s.label}`}
              aria-current={active ? "page" : undefined}
              className={cn(
                "group flex w-full items-center gap-2.5 rounded border px-2.5 py-1.5 text-left text-[12.5px] transition-colors",
                active
                  ? "border-[var(--line)] bg-[var(--bg-subtle)] font-medium text-[var(--text-primary)]"
                  : "border-transparent text-[var(--text-secondary)] hover:bg-[var(--bg-hover)] hover:text-[var(--text-primary)]",
              )}
            >
              <span className={cn("w-5 font-mono text-[11px]", active ? "text-[var(--brand-accent)]" : "text-[var(--text-muted)]")}>
                {s.no}
              </span>
              <span className="flex-1 truncate">{s.label}</span>
              <span
                className={cn(
                  "h-1.5 w-1.5 rounded-full transition-colors",
                  active ? "bg-[var(--brand-accent)]" : "bg-transparent",
                )}
              />
            </button>
          );
        })}
      </nav>
    </aside>
  );
}

function TopBar() {
  const { screen } = useConsole();
  const current = SCREENS.find((s) => s.id === screen) ?? SCREENS[0];

  return (
    <header className="sticky top-0 z-10 flex items-center gap-3 border-b border-[var(--line)] bg-[var(--bg-chrome)]/90 px-6 py-2.5 backdrop-blur">
      <div className="flex min-w-0 flex-col leading-tight">
        <span className="font-mono text-[9.5px] uppercase tracking-[0.16em] text-[var(--text-muted)]">
          Workflow · {current.no}
        </span>
        <span className="truncate text-[15px] font-semibold tracking-tight text-[var(--text-primary)]">{current.title}</span>
      </div>
      <div className="flex-1" />
      <div className="text-right font-mono text-[9.5px] uppercase tracking-[0.08em] text-[var(--text-muted)]">
        Local research prototype · Not investment advice
      </div>
    </header>
  );
}

function Screens() {
  const { screen } = useConsole();
  return (
    <main className="mx-auto flex w-full max-w-[1680px] flex-1 flex-col gap-5 px-6 pb-14 pt-6">
      {screen === "data" && <DataScreen />}
      {screen === "reproduce" && <ReproduceScreen />}
      {screen === "predict" && <PredictScreen />}
      {screen === "trading" && <TradingScreen />}
      {screen === "architecture" && <ArchitectureScreen />}
    </main>
  );
}

export default function App() {
  return (
    <ConsoleProvider>
      <div className="flex min-h-screen bg-[var(--bg-page)] text-[var(--text-primary)]">
        <Sidebar />
        <div className="flex min-w-0 flex-1 flex-col">
          <TopBar />
          <Screens />
        </div>
      </div>
    </ConsoleProvider>
  );
}
