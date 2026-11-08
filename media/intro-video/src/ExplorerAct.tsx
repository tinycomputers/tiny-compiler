import React from "react";
import { interpolate, spring, useCurrentFrame, useVideoConfig } from "remotion";
import { C, MONO, SANS, alpha, ease, mix, springs } from "./theme";
import { clamp, Check } from "./anim";
import { evG, scene, sec } from "./sync";
import { IR_BEFORE, PROGRAM, RESULT_PRETTY as RESULT, TGPU } from "./data";
import {
  BTN, Chrome, Code, Cursor, Header, LINE_H, PANEL, Panel, TAB, TabId, Tabs, WIN, Way, pressAt,
} from "./Explorer";

const A0 = scene(3).start;
export const ACT_FRAMES = scene(6).start + scene(6).frames - A0;
const L = (n: string) => evG(n) - A0;
const o = (i: number) => scene(i).start - A0;

/* ============================================================ small helpers */
const Fade: React.FC<{ f: number; at: number; children: React.ReactNode; y?: number; style?: React.CSSProperties }> = ({ f, at, y = 8, children, style }) => {
  const t = interpolate(f, [at, at + 10], [0, 1], { ...clamp, easing: ease.out });
  return <div style={{ opacity: t, transform: `translateY(${(1 - t) * y}px)`, ...style }}>{children}</div>;
};
const Label: React.FC<{ children: React.ReactNode; style?: React.CSSProperties }> = ({ children, style }) => (
  <div style={{ fontFamily: SANS, fontSize: 13, fontWeight: 600, color: C.muted, ...style }}>{children}</div>
);

/* ============================================================ placeholder */
const Placeholder: React.FC<{ f: number }> = ({ f }) => (
  <div style={{ height: 440, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 16,
    fontFamily: SANS, color: C.muted, fontSize: 14, opacity: 0.9 }}>
    <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 22px)", gap: 6 }}>
      {Array.from({ length: 16 }).map((_, i) => {
        const d = (i % 4) + Math.floor(i / 4);
        const k = (Math.sin(f / 14 - d * 0.9) + 1) / 2;
        return <div key={i} style={{ width: 22, height: 22, borderRadius: 5, background: mix(C.peIdle, C.cyan, k * 0.35), border: `1px solid ${C.line}` }} />;
      })}
    </div>
    <div>Press <b style={{ color: C.fg, fontWeight: 600 }}>Compile &amp; run</b> to see the passes, both chips and the head-to-head.</div>
  </div>
);

/* ============================================================ Passes */
const PILLS = ["parse", "shape-infer", "canonicalize", "const-fold", "cse", "dce", "fuse", "range-check", "legalize-gpu", "legalize-tpu"];
const irColor = (s: string): React.ReactNode =>
  s.split(/(%\w+|\bmatmul\b|\brelu\b|\bfunc\b|\breturn\b|\bi8\b|\bi32\b|\{[^}]*\})/g).map((t, i) => {
    if (!t) return null;
    let color: string = C.fg, weight = 400;
    if (/^%/.test(t)) color = C.cyan;
    else if (/^(func|return)$/.test(t)) color = C.accent;
    else if (/^(matmul|relu)$/.test(t)) weight = 700;
    else if (/^i(8|32)$/.test(t)) color = C.muted;
    else if (t.startsWith("{")) { color = C.lime; weight = 700; }
    return <span key={i} style={{ color, fontWeight: weight }}>{t}</span>;
  });

const PassesView: React.FC<{ f: number }> = ({ f }) => {
  const base = L("s3.click") + 10;
  const act = [L("s4.p0"), L("s4.p1"), L("s4.p2"), L("s4.p3"), L("s4.p4"), L("s4.p5"), L("s4.fuse"),
    L("s4.fuse") + sec(0.75), L("s4.legal"), L("s4.legal") + 4];
  const q = spring({ frame: f - L("s4.fuse"), fps: 30, config: springs.smooth });
  const lines = [0, 1, 2, 3, 4];
  const lineStyle = (i: number): React.CSSProperties => {
    if (i === 2) return { opacity: Math.max(0, 1 - q * 1.7), transform: `translateY(${-q * LINE_H}px) scale(${1 - q * 0.04})` };
    if (i >= 3) return { transform: `translateY(${-q * LINE_H}px)` };
    return {};
  };
  const levels = ["AST", "Graph IR", "Tile IR · Kernel IR", "tiny-gpu · tiny-tpu"];
  const lvActive = f < L("s4.legal") ? 1 : 2;
  return (
    <div>
      <Fade f={f} at={base}><Label>Pass pipeline</Label></Fade>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 8, marginTop: 10 }}>
        {PILLS.map((p, i) => {
          const a = act[i];
          const enter = interpolate(f, [base + i * 3, base + i * 3 + 12], [0, 1], { ...clamp, easing: ease.out });
          const st = f < a ? 0 : f < a + 9 ? 1 : 2;
          const hot = i === 6 && f >= a;
          return (
            <div key={p} style={{ height: 30, padding: "0 11px", borderRadius: 6, display: "flex", alignItems: "center", gap: 7,
              fontFamily: MONO, fontSize: 13, opacity: enter, transform: `translateY(${(1 - enter) * 8}px)`,
              border: `1px solid ${st === 1 ? C.accent : hot ? alpha(C.lime, 0.6) : C.line}`,
              background: st === 1 ? alpha(C.accent, 0.12) : hot ? alpha(C.lime, 0.1) : C.codeBg,
              color: st === 0 ? C.muted : C.fg }}>
              {st === 2 ? <Check size={14} color={hot ? C.lime : C.ok} p={interpolate(f, [a + 9, a + 17], [0, 1], clamp)} />
                : <span style={{ width: 7, height: 7, borderRadius: 7, background: st === 1 ? C.accent : C.line }} />}
              {p}
            </div>
          );
        })}
      </div>

      <Fade f={f} at={base + 22} style={{ marginTop: 20 }}>
        <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between" }}>
          <Label>Graph IR <span style={{ fontWeight: 400, marginLeft: 6 }}>{q > 0.4 ? "after fuse" : "after dce"}</span></Label>
          <div style={{ opacity: interpolate(q, [0.5, 1], [0, 1], clamp), transform: `scale(${0.9 + 0.1 * q})`, fontFamily: MONO, fontSize: 12.5,
            color: C.lime, background: alpha(C.lime, 0.12), border: `1px solid ${alpha(C.lime, 0.4)}`, borderRadius: 6, padding: "2px 9px" }}>
            2 ops → 1
          </div>
        </div>
        <div style={{ marginTop: 8, background: C.codeBg, borderRadius: 8, border: `1px solid ${C.line}`, padding: "14px 16px",
          fontFamily: MONO, fontSize: 14, lineHeight: `${LINE_H}px`, height: LINE_H * 5 + 28, position: "relative", overflow: "hidden" }}>
          {lines.map((i) => (
            <div key={i} style={{ position: "absolute", left: 16, right: 16, top: 14 + i * LINE_H, height: LINE_H, whiteSpace: "pre", ...lineStyle(i) }}>
              {i === 1 ? (
                <div style={{ position: "relative", borderRadius: 4, background: alpha(C.lime, 0.16 * Math.sin(Math.PI * Math.min(1, q))) }}>
                  <span style={{ opacity: 1 - q }}>{irColor(IR_BEFORE[1])}</span>
                  <span style={{ position: "absolute", left: 0, top: 0, opacity: q }}>
                    {irColor("  %1 = matmul{epilogue=relu} %A, %B : i32[4,4]")}
                  </span>
                </div>
              ) : i === 2 ? irColor(IR_BEFORE[2])
                : i === 3 ? <span style={{ opacity: 1 }}>{irColor(q > 0.5 ? "  return %1" : "  return %1")}</span>
                  : irColor(IR_BEFORE[i])}
            </div>
          ))}
        </div>
      </Fade>

      <Fade f={f} at={base + 34} style={{ marginTop: 20, display: "flex", alignItems: "center", gap: 8 }}>
        {levels.map((lv, i) => (
          <React.Fragment key={lv}>
            <div style={{ flex: i === 2 || i === 3 ? 1.4 : 1, height: 44, borderRadius: 8, display: "flex", alignItems: "center", justifyContent: "center",
              fontFamily: SANS, fontSize: 13, fontWeight: 600, textAlign: "center",
              border: `1px solid ${i === lvActive ? C.accent : C.line}`, background: i === lvActive ? alpha(C.accent, 0.12) : C.codeBg,
              color: i === lvActive ? C.fg : C.muted }}>{lv}</div>
            {i < 3 && <span style={{ color: C.muted, fontSize: 15 }}>→</span>}
          </React.Fragment>
        ))}
      </Fade>
    </div>
  );
};

/* ============================================================ tiny-gpu */
const asmColor = (s: string): React.ReactNode => {
  if (s.startsWith(";")) return <span style={{ color: C.muted, fontStyle: "italic" }}>{s}</span>;
  const [op, ...rest] = s.split(" ");
  const opColor = /^(LD|ST|SHLD|SHST)$/.test(op) ? C.cyan : op === "SYNCB" ? C.lime : C.fg;
  return (
    <span>
      <span style={{ color: opColor, fontWeight: 700 }}>{op}</span>{" "}
      {rest.join(" ").split(/(R\d)/g).map((t, i) => <span key={i} style={{ color: /^R\d$/.test(t) ? C.accent : C.fg }}>{t}</span>)}
    </span>
  );
};
const GpuView: React.FC<{ f: number; f0: number }> = ({ f, f0 }) => {
  const scroll = Math.max(0, f - f0 - 46) * 0.62;
  return (
    <div style={{ position: "relative", height: 452 }}>
      <Fade f={f} at={f0} style={{ display: "flex", gap: 8, alignItems: "center" }}>
        <Label style={{ whiteSpace: "nowrap" }}>kernel k0_matmul</Label>
        {["4 blocks × 4 threads", "5 registers"].map((t) => (
          <span key={t} style={{ fontFamily: MONO, fontSize: 12, color: C.muted, border: `1px solid ${C.line}`, borderRadius: 5, padding: "1px 8px", whiteSpace: "nowrap" }}>{t}</span>
        ))}
      </Fade>
      <div style={{ position: "absolute", top: 34, left: 0, right: 0, bottom: 42, overflow: "hidden", background: C.codeBg,
        border: `1px solid ${C.line}`, borderRadius: 8 }}>
        <div style={{ transform: `translateY(${-scroll}px)`, padding: "10px 0" }}>
          {TGPU.map((s, i) => {
            const at = f0 + 6 + i * 2.2;
            const t = interpolate(f, [at, at + 8], [0, 1], { ...clamp, easing: ease.out });
            return (
              <div key={i} style={{ display: "flex", height: 22, alignItems: "center", fontFamily: MONO, fontSize: 13.5, whiteSpace: "pre",
                opacity: t, transform: `translateX(${(1 - t) * -14}px)` }}>
                <span style={{ width: 40, textAlign: "right", paddingRight: 14, color: alpha(C.muted, 0.6) }}>{i + 1}</span>
                {asmColor(s)}
              </div>
            );
          })}
        </div>
        <div style={{ position: "absolute", left: 0, right: 0, bottom: 0, height: 54, background: `linear-gradient(transparent, ${C.codeBg})` }} />
      </div>
      <Fade f={f} at={f0 + 34} style={{ position: "absolute", left: 0, right: 0, bottom: 0, height: 32, display: "flex", gap: 22,
        alignItems: "center", fontFamily: MONO, fontSize: 13, color: C.muted }}>
        <span><b style={{ color: C.fg }}>885</b> instructions</span><span><b style={{ color: C.fg }}>3</b> cycles</span><span><b style={{ color: C.fg }}>16</b> threads</span>
        <span style={{ marginLeft: "auto", color: C.ok }}>R5–R7 untouched ✓</span>
      </Fade>
    </div>
  );
};

/* ============================================================ tiny-tpu */
const TpuView: React.FC<{ f: number; f0: number }> = ({ f, f0 }) => {
  const rel = f - (f0 + 10);
  const pass = rel < 0 ? 0 : Math.min(1, Math.floor(rel / 45));
  const cyc = rel < 0 ? -1 : Math.min(14, Math.floor((rel - pass * 45) / 3));
  const done = rel >= 90;
  const peState = (r: number, c: number) => {
    if (done) return "result";
    if (cyc < 0) return "idle";
    if (cyc < 4) return c <= cyc ? "weight" : "idle";
    const w = cyc - 4, d = r + c;
    if (d === w) return "active";
    if (d < w) return "result";
    return "weight";
  };
  const col = { idle: C.peIdle, weight: C.cyan, active: C.lime, result: C.green } as const;
  const total = Math.round(interpolate(rel, [0, 90], [0, 30], clamp));
  return (
    <div style={{ display: "flex", gap: 22, height: 452 }}>
      <div style={{ width: 262 }}>
        <Fade f={f} at={f0}><Label>4×4 PE array · real RTL</Label></Fade>
        <div style={{ position: "relative", marginTop: 14, width: 250, height: 250 }}>
          {Array.from({ length: 16 }).map((_, i) => {
            const r = Math.floor(i / 4), c = i % 4;
            const s = peState(r, c);
            const enter = interpolate(f, [f0 + i, f0 + i + 12], [0, 1], { ...clamp, easing: ease.out });
            return <div key={i} style={{ position: "absolute", left: c * 62, top: r * 62, width: 54, height: 54, borderRadius: 10,
              background: col[s], border: s === "idle" ? `1px solid ${C.line}` : "none", opacity: enter, transform: `scale(${0.6 + 0.4 * enter})`,
              boxShadow: s === "active" ? `0 0 0 3px ${alpha(C.lime, 0.18)}` : undefined }} />;
          })}
        </div>
        <Fade f={f} at={f0 + 14} style={{ marginTop: 14, fontFamily: MONO, fontSize: 13, color: C.muted, lineHeight: 1.7 }}>
          <div>pass <b style={{ color: C.fg }}>{pass}</b> of 2 · cycle <b style={{ color: C.fg }}>{String(Math.max(0, cyc)).padStart(2, "0")}</b>/15</div>
          <div style={{ display: "flex", gap: 10, marginTop: 6 }}>
            {([["weights", C.cyan], ["MAC", C.lime], ["result", C.green]] as const).map(([n, c]) => (
              <span key={n} style={{ display: "flex", alignItems: "center", gap: 5 }}><i style={{ width: 8, height: 8, borderRadius: 8, background: c, display: "inline-block" }} />{n}</span>
            ))}
          </div>
        </Fade>
      </div>
      <div style={{ flex: 1 }}>
        <Fade f={f} at={f0 + 4}><Label>Tile schedule · .ttpu.json</Label></Fade>
        {[0, 1].map((p) => {
          const live = rel >= p * 45 && !done && pass === p;
          const fin = rel >= (p + 1) * 45 || done;
          return (
            <Fade key={p} f={f} at={f0 + 10 + p * 6} style={{ marginTop: 12 }}>
              <div style={{ border: `1px solid ${live ? alpha(C.lime, 0.6) : C.line}`, background: live ? alpha(C.lime, 0.07) : C.codeBg,
                borderRadius: 8, padding: "11px 14px", fontFamily: MONO, fontSize: 13.5 }}>
                <div style={{ display: "flex", justifyContent: "space-between", fontFamily: SANS, fontWeight: 600, fontSize: 14 }}>
                  <span>pass {p}</span>
                  <span style={{ color: fin ? C.ok : live ? C.lime : C.muted, fontWeight: 500, fontSize: 13 }}>{fin ? "done ✓" : live ? "running" : "queued"}</span>
                </div>
                <div style={{ marginTop: 5, color: C.muted }}>
                  <span style={{ color: C.fg }}>A[0,{p * 4}]</span> × <span style={{ color: C.fg }}>B[{p * 4},0]</span> → psum
                </div>
              </div>
            </Fade>
          );
        })}
        <Fade f={f} at={f0 + 26} style={{ marginTop: 16 }}>
          <Label>Host work <span style={{ fontWeight: 400 }}>(not chip cycles)</span></Label>
          <div style={{ display: "flex", gap: 8, marginTop: 8 }}>
            {["alloc ×1", "acc ×2"].map((t) => (
              <span key={t} style={{ fontFamily: MONO, fontSize: 13, color: C.muted, border: `1px dashed ${C.line}`, borderRadius: 6, padding: "3px 10px" }}>{t}</span>
            ))}
          </div>
        </Fade>
        <Fade f={f} at={f0 + 36} style={{ marginTop: 22, borderTop: `1px solid ${C.line}`, paddingTop: 14, fontFamily: MONO, fontSize: 14, color: C.muted }}>
          2 passes × 15 =<span style={{ color: C.fg, fontWeight: 700, fontSize: 24, marginLeft: 10, fontFamily: SANS }}>{total}</span> RTL cycles
        </Fade>
      </div>
    </div>
  );
};

/* ============================================================ Run & compare */
const TH: React.CSSProperties = { textAlign: "left", padding: "6px 8px", color: C.muted, fontWeight: 600, fontSize: 12.5, borderBottom: `1px solid ${C.line}` };
const TD: React.CSSProperties = { padding: "7px 8px", borderBottom: `1px solid ${C.line}`, fontSize: 13, verticalAlign: "top" };
const RunView: React.FC<{ f: number }> = ({ f }) => {
  const E = { gpu: L("s6.gpu"), tpu: L("s6.tpu"), np: L("s6.numpy"), ok1: L("s6.ok1"), ok2: L("s6.ok2") };
  const cell = (at: number, i: number, text: React.ReactNode) => {
    const t = interpolate(f, [at + 16 + i * 4, at + 26 + i * 4], [0, 1], { ...clamp, easing: ease.out });
    const running = f >= at && f < at + 16 + i * 4;
    if (f < at) return <span style={{ color: alpha(C.muted, 0.5) }}>·</span>;
    if (running) return <span style={{ color: C.accent, opacity: 0.6 + 0.4 * Math.sin(f / 3) }}>running…</span>;
    return <span style={{ opacity: t, display: "inline-block", transform: `translateY(${(1 - t) * 5}px)` }}>{text}</span>;
  };
  const okCell = (at: number) => {
    const t = interpolate(f, [at, at + 12], [0, 1], { ...clamp, easing: ease.out });
    return (
      <span style={{ display: "inline-flex", alignItems: "center", gap: 6, color: C.ok, fontWeight: 700, opacity: t,
        transform: `scale(${0.85 + 0.15 * t})` }}>
        <Check size={16} p={t} /> bit-exact
      </span>
    );
  };
  const typed = Math.round(interpolate(f, [E.np, E.np + 26], [0, RESULT.length], { ...clamp, easing: ease.out }));
  return (
    <div style={{ fontFamily: SANS, color: C.fg }}>
      <table style={{ borderCollapse: "collapse", width: "100%", tableLayout: "fixed" }}>
        <colgroup><col style={{ width: 54 }} /><col style={{ width: 286 }} /><col style={{ width: 128 }} /><col style={{ width: 124 }} /></colgroup>
        <thead><tr><th style={TH}>result</th><th style={TH}>numpy (oracle)</th><th style={TH}>tiny-gpu</th><th style={TH}>tiny-tpu</th></tr></thead>
        <tbody><tr>
          <td style={{ ...TD, fontFamily: MONO }}>%1</td>
          <td style={{ ...TD, fontFamily: MONO, fontSize: 12.5, whiteSpace: "pre", lineHeight: 1.5 }}>
            {f < E.np ? <span style={{ color: alpha(C.muted, 0.5) }}>·</span> : RESULT.slice(0, typed)}
          </td>
          <td style={TD}>{f >= E.ok1 ? okCell(E.ok1) : <span style={{ color: alpha(C.muted, 0.5) }}>·</span>}</td>
          <td style={TD}>{f >= E.ok2 ? okCell(E.ok2) : <span style={{ color: alpha(C.muted, 0.5) }}>·</span>}</td>
        </tr></tbody>
      </table>
      <div style={{ fontWeight: 700, fontSize: 15, margin: "16px 0 4px" }}>Head to head</div>
      <table style={{ borderCollapse: "collapse", width: "100%", tableLayout: "fixed" }}>
        <colgroup><col style={{ width: 152 }} /><col style={{ width: 100 }} /><col style={{ width: 340 }} /></colgroup>
        <thead><tr><th style={TH}>measured</th><th style={TH}>tiny-gpu</th><th style={TH}>tiny-tpu</th></tr></thead>
        <tbody>
          {([
            ["cycles", "3", "30 (2 passes)"],
            ["instructions", "885", <span style={{ color: C.muted }}>not applicable (fixed-function datapath)</span>],
            ["global loads / stores", "160 / 16", <span style={{ color: C.muted }}>host-side (not counted on the chip)</span>],
            ["host operations", <span style={{ color: C.muted }}>none</span>, <span style={{ fontFamily: MONO, fontSize: 11.5 }}>{`{"alloc":1,"acc":2,"epilogue":0,…}`}</span>],
          ] as [string, React.ReactNode, React.ReactNode][]).map(([k, g, t], i) => (
            <tr key={k}>
              <td style={TD}>{k}</td>
              <td style={{ ...TD, fontFamily: MONO, fontWeight: 600 }}>{cell(E.gpu, i, g)}</td>
              <td style={{ ...TD, fontFamily: MONO, fontWeight: 600 }}>{cell(E.tpu, i, t)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <Fade f={f} at={E.ok2 + 8} style={{ marginTop: 12 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10, padding: "10px 14px", borderRadius: 8, background: alpha(C.ok, 0.1),
          border: `1px solid ${alpha(C.ok, 0.45)}`, fontWeight: 600, fontSize: 14 }}>
          <Check size={18} p={interpolate(f, [E.ok2 + 12, E.ok2 + 24], [0, 1], clamp)} /> Same program, same result on both chips: equal to numpy, bit for bit.
        </div>
      </Fade>
      <Fade f={f} at={E.tpu + 40} style={{ marginTop: 12, color: C.muted, fontSize: 12.5, lineHeight: 1.5 }}>
        tiny-gpu runs straight-line code in one cycle, so compare instructions as well as cycles. The two chips are different machines: this shows what each one does for the same program, not a performance verdict.
      </Fade>
    </div>
  );
};

/* ============================================================ the act */
const CAM: [number, number, number, number][] = [
  [0, 0.94, 0, 30],
  [o(3) + sec(0.9), 1.0, 0, 0],
  [o(3) + sec(1.5), 1.05, 60, 6],
  [o(3) + sec(4.0), 1.05, 60, 6],
  [o(3) + sec(4.9), 1.04, 10, 0],
  [o(4) + sec(0.7), 1.15, -150, 0],
  [o(5) + sec(3.6), 1.15, -150, 0],
  [o(6) + sec(0.1), 1.16, -160, -5],
  [o(6) + sec(7.4), 1.2, -175, -25],
  [ACT_FRAMES - 14, 1.2, -175, -25],
];

export const ExplorerAct: React.FC = () => {
  const f = useCurrentFrame();
  const { fps } = useVideoConfig();

  /* camera */
  let cs = CAM[0][1], cx = CAM[0][2], cy = CAM[0][3];
  for (let i = 0; i < CAM.length - 1; i++) {
    const [f0, s0, x0, y0] = CAM[i], [f1, s1, x1, y1] = CAM[i + 1];
    if (f >= f0 && f <= f1) {
      const t = ease.cam((f - f0) / Math.max(1, f1 - f0));
      cs = s0 + (s1 - s0) * t; cx = x0 + (x1 - x0) * t; cy = y0 + (y1 - y0) * t;
    } else if (f > f1) { cs = s1; cx = x1; cy = y1; }
  }
  const enter = spring({ frame: f - 2, fps, config: springs.smooth });
  const exit = interpolate(f, [ACT_FRAMES - 11, ACT_FRAMES - 1], [0, 1], { ...clamp, easing: ease.in });

  /* typing */
  const sched: [number, number][] = [[0.4, 0.85], [0.95, 1.35], [1.4, 1.8], [1.9, 2.6], [2.7, 3.0]];
  const shown = PROGRAM.map((ln, i) => {
    const p = interpolate(f, [o(3) + sec(sched[i][0]), o(3) + sec(sched[i][1])], [0, 1], clamp);
    return ln.slice(0, Math.round(ln.length * p));
  });
  const typing = PROGRAM.findIndex((ln, i) => shown[i].length < ln.length);
  const caretLine = typing === -1 ? PROGRAM.length - 1 : typing;
  const typingNow = f >= o(3) + sec(0.4) && f < o(3) + sec(3.4);
  const caretOn = typingNow || Math.floor(f / 15) % 2 === 0;

  /* callouts */
  const hl = spring({ frame: f - L("s3.hl"), fps, config: springs.snappy });
  const four = spring({ frame: f - L("s3.four"), fps, config: springs.snappy });
  const bracketOut = interpolate(f, [L("s3.click"), L("s3.click") + 14], [0, 1], clamp);
  const hlOut = interpolate(f, [L("s3.four"), L("s3.four") + 10], [1, 0], clamp);

  /* clicks, tabs */
  const cCompile = L("s3.click"), cGpu = L("s5.gpuTab"), cTpu = L("s5.tpuTab"), cRun = L("s6.tab");
  const compiledAt = cCompile + 6;
  let tab: TabId | null = null, tabAt = compiledAt;
  if (f >= compiledAt) { tab = "passes"; tabAt = compiledAt; }
  if (f >= cGpu + 3) { tab = "gpu"; tabAt = cGpu + 3; }
  if (f >= cTpu + 3) { tab = "tpu"; tabAt = cTpu + 3; }
  if (f >= cRun + 3) { tab = "run"; tabAt = cRun + 3; }
  let pressTab: TabId | null = null, pressAmt = 0;
  for (const [c, id] of [[cGpu, "gpu"], [cTpu, "tpu"], [cRun, "run"]] as [number, TabId][]) {
    const a = pressAt(f, c); if (a > 0) { pressTab = id; pressAmt = a; }
  }
  const btnPress = pressAt(f, cCompile);
  const busy = f >= cCompile && f < cCompile + 22;

  const cursorPath: Way[] = [
    [0, 640, 470], [o(3) + sec(3.05), 640, 470], [cCompile - 2, BTN.x - 8, BTN.y + 4], [cCompile + 14, BTN.x - 8, BTN.y + 4],
    [o(4) + sec(1.8), 1090, 520], [o(4) + sec(6.4), 1110, 540], [cGpu - 3, TAB.gpu - 4, TAB.y + 4], [cGpu + 14, TAB.gpu - 4, TAB.y + 4],
    [cGpu + 40, 1090, 500], [cTpu - 3, TAB.tpu - 4, TAB.y + 4], [cTpu + 14, TAB.tpu - 4, TAB.y + 4],
    [cTpu + 40, 1100, 520], [cRun - 3, TAB.run - 4, TAB.y + 4], [cRun + 14, TAB.run - 4, TAB.y + 4],
    [cRun + 50, 1010, 540],
  ];

  const tabFade = interpolate(f, [tabAt, tabAt + 9], [0, 1], { ...clamp, easing: ease.out });
  const lineTop = 10;

  return (
    <div style={{ position: "absolute", inset: 0, opacity: Math.min(1, enter * 1.5) * (1 - exit),
      transform: `scale(${1 - exit * 0.07})`, filter: `blur(${exit * 8}px)` }}>
      <div style={{ position: "absolute", left: 960, top: 500, width: 0, height: 0,
        transform: `translate(${cx}px, ${cy + (1 - enter) * 90}px) scale(${cs * (0.94 + 0.06 * enter)})` }}>
        <div style={{ position: "absolute", left: (-WIN.w * WIN.scale) / 2, top: (-WIN.h * WIN.scale) / 2, width: WIN.w, height: WIN.h,
          transformOrigin: "0 0", transform: `scale(${WIN.scale})` }}>
          <div style={{ position: "absolute", inset: 0, borderRadius: 14, overflow: "hidden", background: C.bg, border: `1px solid ${C.line}`,
            boxShadow: "0 1px 2px rgba(20,18,14,.08), 0 30px 60px -36px rgba(20,18,14,.35)" }}>
            <Chrome />
            <Header press={btnPress} busy={busy} />

            {/* left: editor */}
            <Panel x={PANEL.lx} w={PANEL.lw}>
              <div style={{ position: "relative", height: 430, border: `1px solid ${C.line}`, borderRadius: 6, background: C.panel, overflow: "hidden" }}>
                {PROGRAM.map((_, i) => {
                  const active = i === caretLine && typingNow;
                  const hot = i === 3 && hl > 0.05;
                  return (
                    <div key={i} style={{ position: "absolute", left: 0, right: 0, top: lineTop + i * LINE_H, height: LINE_H, display: "flex", alignItems: "center",
                      background: hot ? alpha(C.lime, 0.12 * hlOut) : active ? alpha(C.accent, 0.1) : "transparent",
                      fontFamily: MONO, fontSize: 15, whiteSpace: "pre" }}>
                      <span style={{ width: 38, textAlign: "right", paddingRight: 14, color: C.muted, fontSize: 13 }}>{i + 1}</span>
                      <span><Code text={shown[i]} hot={hot && hlOut > 0.5} /></span>
                      {i === caretLine && caretOn && f >= o(3) + sec(0.4) && f < L("s3.click") + 6 && <span style={{ width: 2, height: 17, background: C.fg, marginLeft: 1 }} />}
                    </div>
                  );
                })}
                {/* relu line callout */}
                <div style={{ position: "absolute", left: 38 + 15 * 9.1 + 16, top: lineTop + 3 * LINE_H + 1, opacity: Math.min(1, hl * 1.4) * hlOut,
                  transform: `translateX(${(1 - hl) * -10}px)`, fontFamily: MONO, fontSize: 12.5, color: C.lime, background: alpha(C.lime, 0.12),
                  border: `1px solid ${alpha(C.lime, 0.45)}`, borderRadius: 5, padding: "0 8px", height: 22, lineHeight: "20px" }}>
                  matmul → relu
                </div>
                {/* four lines bracket */}
                <div style={{ position: "absolute", left: 250, top: lineTop + 1 * LINE_H + 3, height: LINE_H * 4 - 6, width: 12 * four, opacity: four * (1 - bracketOut),
                  borderTop: `2px solid ${C.lime}`, borderBottom: `2px solid ${C.lime}`, borderRight: `2px solid ${C.lime}`, borderRadius: "0 8px 8px 0" }} />
                <div style={{ position: "absolute", left: 250 + 24, top: lineTop + 2.5 * LINE_H - 4, opacity: four * (1 - bracketOut), transform: `translateX(${(1 - four) * -8}px)`,
                  fontFamily: SANS, fontWeight: 700, fontSize: 15, color: C.lime }}>
                  4 lines
                </div>
              </div>
              <div style={{ marginTop: 12, fontFamily: SANS, fontSize: 12.5, color: C.muted, lineHeight: 1.5 }}>
                Language: <code style={{ fontFamily: MONO, color: C.fg }}>input</code>, <code style={{ fontFamily: MONO, color: C.fg }}>const</code>,{" "}
                <code style={{ fontFamily: MONO, color: C.fg }}>output</code>, <code style={{ fontFamily: MONO, color: C.fg }}>@</code> (matmul),{" "}
                <code style={{ fontFamily: MONO, color: C.fg }}>+</code>(row broadcast), <code style={{ fontFamily: MONO, color: C.fg }}>relu</code>. Integers only.
              </div>
            </Panel>

            {/* right: tabs + views */}
            <Panel x={PANEL.rx} w={PANEL.rw}>
              <Tabs active={tab} press={pressTab} pressAmt={pressAmt} />
              <div style={{ marginTop: 12, height: 456, opacity: tab ? tabFade : 1, transform: tab ? `translateY(${(1 - tabFade) * 8}px)` : undefined }}>
                {tab === null && <Placeholder f={f} />}
                {tab === "passes" && <PassesView f={f} />}
                {tab === "gpu" && <GpuView f={f} f0={tabAt} />}
                {tab === "tpu" && <TpuView f={f} f0={tabAt} />}
                {tab === "run" && <RunView f={f} />}
              </div>
            </Panel>

            <Cursor f={f} path={cursorPath} clicks={[cCompile, cGpu, cTpu, cRun]} show={[o(3) + sec(3.05), cRun + 70]} />
          </div>
        </div>
      </div>
    </div>
  );
};
