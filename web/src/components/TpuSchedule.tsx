import { useEffect, useState } from "react";
import { PEGrid } from "./PEGrid";
import type { TpuRun } from "../lib/tpu";

// Plays the RTL's per-cycle trace of each chip pass on the systolic-array grid.
export default function TpuSchedule({ run }: { run: TpuRun }) {
  const [pass, setPass] = useState(0);
  const [cycle, setCycle] = useState(0);
  const [playing, setPlaying] = useState(false);

  const current = run.passes[pass];
  const last = current ? current.states.length - 1 : 0;

  // Advance one cycle per tick while playing; stop at the end of the pass.
  useEffect(() => {
    if (!playing) return;
    const id = window.setInterval(() => {
      setCycle((c) => {
        if (c >= last) {
          setPlaying(false);
          return c;
        }
        return c + 1;
      });
    }, 250);
    return () => window.clearInterval(id);
  }, [playing, last]);

  if (!current) return <p className="muted">This program has no chip passes.</p>;
  const state = current.states[cycle];

  function choosePass(index: number) {
    setPass(index);
    setCycle(0);
    setPlaying(false);
  }

  return (
    <div>
      <div className="tabs">
        {run.passes.map((p, i) => (
          <button key={p.id} aria-selected={i === pass} onClick={() => choosePass(i)}>
            pass {i + 1}
          </button>
        ))}
      </div>
      <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
        <button onClick={() => setPlaying((p) => !p)}>{playing ? "Pause" : "Play"}</button>
        <input
          type="range"
          min={0}
          max={last}
          value={cycle}
          onChange={(e) => {
            setPlaying(false);
            setCycle(Number(e.target.value));
          }}
          aria-label="cycle"
        />
        <span className="muted">
          cycle {cycle} / {last} · {state?.fsmState}
        </span>
      </div>
      {state && <PEGrid state={state} />}
      <p className="muted" style={{ marginTop: 8 }}>
        Each pass runs on the real tiny-tpu RTL; this grid shows the RTL's own signals, cycle by cycle.
      </p>
    </div>
  );
}
