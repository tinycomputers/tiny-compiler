import { useEffect, useState } from "react";
import { PEGrid } from "./PEGrid";
import type { TpuRun } from "../lib/tpu";

// Plays the RTL's per-cycle trace of each chip pass on the systolic-array grid.
export default function TpuSchedule({ run }: { run: TpuRun }) {
  const [pass, setPass] = useState(0);
  const [cycle, setCycle] = useState(0);
  // Play on arrival, unless the reader asked for reduced motion.
  const [playing, setPlaying] = useState(() => !window.matchMedia("(prefers-reduced-motion: reduce)").matches);

  const current = run.passes[pass];
  const last = current ? current.states.length - 1 : 0;
  const lastPass = run.passes.length - 1;

  // One cycle per tick. At the end of a pass, the next pass starts; the replay stops after the last.
  useEffect(() => {
    if (!playing) return;
    const id = window.setInterval(() => {
      if (cycle < last) {
        setCycle(cycle + 1);
      } else if (pass < lastPass) {
        setPass(pass + 1);
        setCycle(0);
      } else {
        setPlaying(false);
      }
    }, 250);
    return () => window.clearInterval(id);
  }, [playing, cycle, last, pass, lastPass]);

  if (!current) return <p className="muted">This program has no chip passes.</p>;
  const state = current.states[cycle];

  function choosePass(index: number) {
    setPass(index);
    setCycle(0);
    setPlaying(false);
  }

  function replay() {
    setPass(0);
    setCycle(0);
    setPlaying(true);
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
        <button onClick={() => (playing ? setPlaying(false) : replay())}>
          {playing ? "Pause" : cycle === last && pass === lastPass ? "Replay" : "Play"}
        </button>
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
