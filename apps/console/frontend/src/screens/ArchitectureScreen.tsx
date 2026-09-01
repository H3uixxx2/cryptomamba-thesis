import { useEffect } from "react";

import { Callout, ErrorBanner, Field, PageHeader, Panel, Pill, StatTile } from "@/components/primitives";
import { ScanAxisDiagram } from "@/components/ScanAxisDiagram";
import { Skeleton } from "@/components/ui/skeleton";
import { fmt } from "@/lib/format";
import { useConsole } from "@/store";

export function ArchitectureScreen() {
  const { architecture } = useConsole();
  const data = architecture.data;

  useEffect(() => {
    if (!architecture.data && !architecture.loading) architecture.reload();
  }, [architecture]);

  if (architecture.error) return <ErrorBanner message={architecture.error} />;
  if (!data) return <Skeleton className="h-64 w-full" />;
  if (data.status !== "READY" || !data.model || !data.parameter_comparison) {
    return (
      <>
        <PageHeader title="Architecture" />
        <Callout tone="amber" title="Architecture evidence is not ready">{data.message ?? "Missing final evidence."}</Callout>
      </>
    );
  }

  const val = data.controlled_metrics?.find((row) => row.split === "val");
  const test = data.controlled_metrics?.find((row) => row.split === "test");
  const trainingMinutes = data.training?.training_seconds == null ? undefined : data.training.training_seconds / 60;

  return (
    <>
      <PageHeader
        title="Architecture"
        description="Source-derived CM-v execution is contrasted with the single approved model experiment: CMamba-T / S5-Full. Metrics below come from the same controlled evidence used on Evaluation."
        actions={<Pill tone="green">S5-Full · validation + test</Pill>}
      />

      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <StatTile label="S5-Full validation RMSE" value={fmt(val?.RMSE, 2)} sub="304 aligned dates" tone="blue" />
        <StatTile label="S5-Full test RMSE" value={fmt(test?.RMSE, 2)} sub="304 aligned dates" tone="brand" />
        <StatTile label="Parameter footprint" value={`${data.parameter_comparison.s5_to_cm_v_pct}%`} sub={`${data.model.parameter_count.toLocaleString("en-US")} of ${data.parameter_comparison.reproduced_cm_v.toLocaleString("en-US")}`} tone="green" />
        <StatTile label="Training time" value={trainingMinutes == null ? "—" : `${fmt(trainingMinutes, 1)} min`} sub={`seed ${data.training?.seed ?? "—"} · best epoch ${data.training?.best_epoch ?? "—"}`} />
      </div>

      <Panel
        eyebrow="Paper view versus source-derived execution"
        hint="The paper presents CryptoMamba as a temporal predictor; source inspection shows the released CM-v selective scan traverses feature tokens. S5-Full changes the scan to chronological day tokens."
      >
        <ScanAxisDiagram paramsOriginal={data.parameter_comparison.reproduced_cm_v} paramsProposed={data.parameter_comparison.s5_full} />
      </Panel>

      <div className="grid gap-4 xl:grid-cols-2">
        <Panel eyebrow="S5-Full source contract">
          <div className="space-y-3">
            <Field label="Input tensor">{data.source_execution?.input ?? "—"}</Field>
            <Field label="Features">{data.source_execution?.feature_order.join(" · ") ?? "—"}</Field>
            <Field label="Normalization">{data.source_execution?.normalization ?? "—"}</Field>
            <Field label="Scan axis">{data.source_execution?.scan_axis ?? "—"}</Field>
            <div className="rounded border border-[var(--line)] bg-[var(--bg-sunken)] p-3">
              <div className="eyebrow">Stages</div>
              <ol className="mt-2 list-decimal space-y-1 pl-5 text-xs text-[var(--text-secondary)]">
                {(data.source_execution?.stages ?? []).map((stage) => <li key={stage}>{stage}</li>)}
              </ol>
            </div>
          </div>
        </Panel>

        <Panel eyebrow="Training and runtime context">
          <div className="space-y-3">
            <Field label="Checkpoint selection">seed {data.training?.seed ?? "—"} · best epoch {data.training?.best_epoch ?? "—"}</Field>
            <Field label="Runtime">{data.training?.gpu ?? "—"} · PyTorch {data.training?.torch ?? "—"} · mamba-ssm {data.training?.mamba_ssm ?? "—"}</Field>
            <Field label="Training duration">{trainingMinutes == null ? "—" : `${fmt(trainingMinutes, 1)} minutes`}</Field>
          </div>
        </Panel>
      </div>

    </>
  );
}
