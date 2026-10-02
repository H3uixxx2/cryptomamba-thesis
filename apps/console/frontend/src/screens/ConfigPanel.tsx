import { Callout, Panel, Pill } from "@/components/primitives";
import { fetchConfig, type ConfigResponse } from "@/lib/api";
import { useApiOnce } from "@/lib/useApiOnce";

/* Configuration files behind the two trained models, read verbatim from GET /api/config (allow-listed paths). */

const GROUP_ORDER = ["Data", "CryptoMamba-v", "CryptoMamba-T", "Registry"];

export function ConfigPanel() {
  const { data, error, loading } = useApiOnce<ConfigResponse>(fetchConfig, "config");

  let body;
  if (loading) body = <Callout tone="neutral">Loading the configuration files…</Callout>;
  else if (error) body = <Callout tone="amber" title="Configuration is not available">{error}</Callout>;
  else if (!data || data.status !== "READY") body = <Callout tone="amber" title="Configuration is not ready">{data?.message ?? ""}</Callout>;
  else {
    body = (
      <div className="flex flex-col gap-5">
        <p className="text-[12px] text-[var(--text-secondary)]">
          A training run reads one training file. It names a data configuration and a model; the model name is resolved to a model file through the registry, and the model file names the Python class to build and its parameters.
        </p>
        {GROUP_ORDER.map((group) => {
          const files = data.files.filter((f) => f.group === group);
          if (!files.length) return null;
          return (
            <section key={group}>
              <div className="eyebrow mb-2">{group}</div>
              <div className="flex flex-col gap-2">
                {files.map((f) => (
                  <details key={f.id} className="rounded border border-[var(--line-subtle)] bg-[var(--bg-sunken)]">
                    <summary className="cursor-pointer px-3 py-2 text-[12px] font-medium text-[var(--text-primary)]">
                      {f.title} <span className="ml-2 font-mono text-[10.5px] font-normal text-[var(--text-muted)]">{f.path}</span>
                    </summary>
                    <div className="border-t border-[var(--line-subtle)] px-3 py-2">
                      <p className="mb-2 text-[11.5px] text-[var(--text-muted)]">{f.purpose}</p>
                      <pre className="overflow-x-auto rounded bg-[var(--bg-card)] p-3 font-mono text-[11.5px] leading-relaxed text-[var(--text-primary)]">{f.text}</pre>
                    </div>
                  </details>
                ))}
              </div>
            </section>
          );
        })}
        <section>
          <div className="eyebrow mb-2">Names inside the files</div>
          <table className="w-full max-w-xl text-[12px]">
            <tbody>{data.labels.map((l) => (
              <tr key={l.internal} className="border-b border-[var(--line-subtle)] last:border-0">
                <td className="px-3 py-1.5 font-mono text-[11.5px]">{l.internal}</td><td className="px-3 py-1.5 text-[var(--text-secondary)]">{l.name}</td>
              </tr>
            ))}</tbody>
          </table>
        </section>
        <section>
          <div className="eyebrow mb-2">Set outside the files</div>
          <ul className="list-disc pl-5 text-[12px] text-[var(--text-secondary)]">{data.outside_files.map((t) => <li key={t}>{t}</li>)}</ul>
        </section>
      </div>
    );
  }

  return (
    <Panel eyebrow="Model configuration" hint="The files that define the data split and both trained models, shown exactly as they are in the repository." actions={<Pill tone="blue">Read-only</Pill>}>
      {body}
    </Panel>
  );
}
