import { useMemo, useState } from "react";
import { analyze } from "../lib/compiler";
import { EXAMPLES } from "../lib/examples.generated";
import { runTpu, type TpuRun } from "../lib/tpu";
import TpuSchedule from "./TpuSchedule";
import { tpuExact } from "../lib/verify";
import type { Analysis, GpuRun } from "../lib/types";

type Tab = "passes" | "gpu" | "tpu" | "compare";
const TABS: Array<[Tab, string]> = [
  ["passes", "Passes"],
  ["gpu", "tiny-gpu"],
  ["tpu", "tiny-tpu"],
  ["compare", "Run & compare"],
];

function Code({ children }: { children: string }) {
  return <pre>{children}</pre>;
}

function Badge({ ok, label }: { ok: boolean; label: string }) {
  return <span className={ok ? "ok" : "bad"}>{ok ? "✓ " : "✗ "}{label}</span>;
}

export default function Explorer() {
  const [source, setSource] = useState(EXAMPLES["mm_relu"]);
  const [example, setExample] = useState("mm_relu");
  const [opt, setOpt] = useState(1);
  const [seed, setSeed] = useState(42);
  const [tab, setTab] = useState<Tab>("compare");
  const [target, setTarget] = useState<"gpu" | "tpu">("gpu");
  const [passIndex, setPassIndex] = useState(0);
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("");
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [tpuRun, setTpuRun] = useState<TpuRun | null>(null);
  const [tpuError, setTpuError] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function compile() {
    setBusy(true);
    setError(null);
    setTpuRun(null);
    setTpuError(null);
    setStatus("Compiling (the first run downloads Python, about 10 MB)…");
    try {
      const a = await analyze(source, opt, seed);
      setAnalysis(a);
      setPassIndex(0);
      if (a.ok && a.tpu?.ok && a.tpu.program && a.inputs && a.oracle) {
        setStatus("Running on the tiny-tpu RTL…");
        try {
          const run = await runTpu(a.tpu.program, a.inputs);
          setTpuRun(run);
        } catch (e) {
          setTpuError(String((e as Error).message ?? e));
        }
      }
      setStatus("");
    } catch (e) {
      setError(String((e as Error).message ?? e));
      setStatus("");
    } finally {
      setBusy(false);
    }
  }

  const passes = analysis?.gpu && target === "gpu" ? analysis.gpu.passes : analysis?.tpu?.passes ?? [];
  const exactTpu = useMemo(
    () => (analysis?.tpu?.ok && tpuRun && analysis.oracle ? tpuExact(tpuRun, analysis.tpu, analysis.oracle) : null),
    [analysis, tpuRun],
  );
  const gpuRun: GpuRun | null | undefined = analysis?.gpu?.run;
  const gpuEst = analysis?.gpu?.estimate;

  return (
    <main>
      <header style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap", marginBottom: 12 }}>
        <h1>Compiler Explorer</h1>
        <a href="/" className="muted">what is this?</a>
        <span style={{ flex: 1 }} />
        <select
          value={example}
          onChange={(e) => {
            setExample(e.target.value);
            setSource(EXAMPLES[e.target.value]);
          }}
          aria-label="example"
        >
          {Object.keys(EXAMPLES).map((k) => (
            <option key={k} value={k}>{k}</option>
          ))}
        </select>
        <select value={opt} onChange={(e) => setOpt(Number(e.target.value))} aria-label="optimization level">
          <option value={0}>-O0</option>
          <option value={1}>-O1</option>
          <option value={2}>-O2</option>
        </select>
        <label className="muted">
          seed <input type="number" value={seed} onChange={(e) => setSeed(Number(e.target.value))} style={{ width: 80 }} />
        </label>
        <button className="primary" onClick={compile} disabled={busy}>
          {busy ? "Working…" : "Compile & run"}
        </button>
      </header>

      <div style={{ display: "grid", gridTemplateColumns: "minmax(0,1fr) minmax(0,1.3fr)", gap: 12 }} className="grid">
        <section className="panel">
          <textarea
            value={source}
            onChange={(e) => setSource(e.target.value)}
            spellCheck={false}
            aria-label="source"
            style={{ width: "100%", minHeight: 360, background: "var(--panel)", color: "var(--fg)", border: 0, resize: "vertical" }}
          />
          <p className="muted" style={{ margin: "8px 0 0" }}>
            Language: <code>input</code>, <code>const</code>, <code>output</code>, <code>@</code> (matmul), <code>+</code>
            (row broadcast), <code>relu</code>. Integers only.
          </p>
          {status && <p className="muted">{status}</p>}
          {error && <p className="bad">{error}</p>}
        </section>

        <section className="panel">
          {!analysis && <p className="muted">Press “Compile & run” to see the passes, both chips, and the results.</p>}
          {analysis && !analysis.ok && <pre>{analysis.error}</pre>}
          {analysis?.ok && (
            <>
              <div className="tabs" role="tablist">
                {TABS.map(([id, label]) => (
                  <button key={id} role="tab" aria-selected={tab === id} onClick={() => setTab(id)}>
                    {label}
                  </button>
                ))}
              </div>

              {tab === "passes" && (
                <>
                  <div className="tabs">
                    <button aria-selected={target === "gpu"} onClick={() => { setTarget("gpu"); setPassIndex(0); }}>tiny-gpu</button>
                    <button aria-selected={target === "tpu"} onClick={() => { setTarget("tpu"); setPassIndex(0); }}>tiny-tpu</button>
                  </div>
                  {passes.length === 0 ? (
                    <p className="muted">This chip did not accept the program.</p>
                  ) : (
                    <>
                      <div className="tabs">
                        {passes.map((p, i) => (
                          <button key={i} aria-selected={i === passIndex} onClick={() => setPassIndex(i)}>
                            {p.name}
                          </button>
                        ))}
                      </div>
                      <Code>{passes[passIndex]?.ir ?? ""}</Code>
                      <p className="muted">Each stage is the IR after that pass. Lowering to kernels and tiles is shown as its own stage.</p>
                    </>
                  )}
                </>
              )}

              {tab === "gpu" && (
                <>
                  {!analysis.gpu?.ok ? (
                    <Code>{analysis.gpu?.error ?? ""}</Code>
                  ) : (
                    <>
                      <p>
                        {analysis.gpu.kernels?.length} launch(es), {gpuEst?.registers} registers per thread
                        {gpuEst && gpuEst.shared_loads + gpuEst.shared_stores > 0 && " · uses shared memory"}
                      </p>
                      <Code>{analysis.gpu.asm ?? ""}</Code>
                    </>
                  )}
                </>
              )}

              {tab === "tpu" && (
                <>
                  {!analysis.tpu?.ok ? (
                    <Code>{analysis.tpu?.error ?? ""}</Code>
                  ) : (
                    <>
                      <p>
                        {analysis.tpu.estimate?.passes} chip pass(es) × 15 cycles = {analysis.tpu.estimate?.rtl_cycles} RTL cycles
                        (estimate)
                      </p>
                      <Code>{analysis.tpu.tile ?? ""}</Code>
                      <h3 style={{ fontSize: 15, marginTop: 16 }}>Array, cycle by cycle</h3>
                      {tpuRun ? (
                        <TpuSchedule run={tpuRun} />
                      ) : (
                        <p className="muted">{tpuError ?? "Running on the tiny-tpu RTL…"}</p>
                      )}
                    </>
                  )}
                </>
              )}

              {tab === "compare" && (
                <Compare analysis={analysis} tpuRun={tpuRun} tpuError={tpuError} exactTpu={exactTpu} gpuRun={gpuRun} />
              )}
            </>
          )}
        </section>
      </div>
      <style>{`@media (max-width: 800px) { .grid { grid-template-columns: 1fr !important; } }`}</style>
    </main>
  );
}

function Compare({
  analysis,
  tpuRun,
  tpuError,
  exactTpu,
  gpuRun,
}: {
  analysis: Analysis;
  tpuRun: TpuRun | null;
  tpuError: string | null;
  exactTpu: boolean | null;
  gpuRun: GpuRun | null | undefined;
}) {
  const names = analysis.outputNames ?? [];
  const oracle = analysis.oracle;
  const sample = (x: unknown) => JSON.stringify(x).slice(0, 160);
  return (
    <>
      {analysis.oracleError && <p className="bad">{analysis.oracleError}</p>}
      <table>
        <thead>
          <tr><th>result</th><th>numpy (oracle)</th><th>tiny-gpu</th><th>tiny-tpu</th></tr>
        </thead>
        <tbody>
          {names.map((name, i) => (
            <tr key={name}>
              <td>%{name}</td>
              <td className="muted">{oracle ? sample(oracle[i]) : "—"}</td>
              <td>
                {!analysis.gpu?.ok ? (
                  <span className="muted">rejected</span>
                ) : gpuRun ? (
                  <Badge ok={gpuRun.exact} label={gpuRun.exact ? "bit-exact" : "differs"} />
                ) : (
                  <span className="muted">not run</span>
                )}
              </td>
              <td>
                {!analysis.tpu?.ok ? (
                  <span className="muted">rejected</span>
                ) : tpuError ? (
                  <span className="bad">{tpuError}</span>
                ) : exactTpu === null ? (
                  <span className="muted">not run</span>
                ) : (
                  <Badge ok={exactTpu} label={exactTpu ? "bit-exact" : "differs"} />
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      {!analysis.tpu?.ok && analysis.tpu?.error && (
        <p className="muted" style={{ marginTop: 10 }}>
          tiny-tpu: <span className="bad">{analysis.tpu.error.split("\n")[0]}</span>
        </p>
      )}

      <h3 style={{ marginTop: 16, fontSize: 15 }}>Head to head</h3>
      <table>
        <thead>
          <tr><th>measured</th><th>tiny-gpu</th><th>tiny-tpu</th></tr>
        </thead>
        <tbody>
          <tr>
            <td>cycles</td>
            <td>{gpuRun ? `${gpuRun.cycles}` : "—"}</td>
            <td>{tpuRun ? `${tpuRun.totalCycles} (${tpuRun.passes.length} pass${tpuRun.passes.length === 1 ? "" : "es"})` : "—"}</td>
          </tr>
          <tr>
            <td>instructions</td>
            <td>{gpuRun ? gpuRun.instructions : "—"}</td>
            <td className="muted">not applicable (fixed-function datapath)</td>
          </tr>
          <tr>
            <td>global loads / stores</td>
            <td>{gpuRun ? `${gpuRun.loads} / ${gpuRun.stores}` : "—"}</td>
            <td className="muted">host-side (not counted on the chip)</td>
          </tr>
          <tr>
            <td>host operations</td>
            <td className="muted">none</td>
            <td>{tpuRun ? JSON.stringify(tpuRun.hostOps) : "—"}</td>
          </tr>
        </tbody>
      </table>
      <p className="muted" style={{ marginTop: 8 }}>
        tiny-gpu runs straight-line code in one cycle, so compare instructions as well as cycles.
        The two chips are different machines: this shows what each one does for the same program, not a
        performance verdict.
      </p>
      {gpuRun && !gpuRun.reservedIntact && <p className="bad">A kernel overwrote R5–R7.</p>}
    </>
  );
}
