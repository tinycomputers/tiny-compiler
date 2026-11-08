import React from "react";
import { interpolate, spring, useCurrentFrame, useVideoConfig } from "remotion";
import { C, MONO, SANS, alpha, ease, mix, springs } from "./theme";
import { Check, Enter, Exit, Words, clamp } from "./anim";
import { PEMark, Wordmark } from "./Logo";
import { Cursor, Way } from "./Explorer";
import { ev, scene } from "./sync";

/* ------------------------------------------------------------------ S7 */
const ROWS = [
  { l: "-O0", n: 7320, c: "#cfcabd", w: 740, note: "naive" },
  { l: "-O1", n: 5208, c: C.cyan, w: 526, note: "licm + strength reduction" },
  { l: "-O2", n: 3288, c: C.lime, w: 332, note: "tiling, unrolling, folding" },
];
export const Scene7: React.FC = () => {
  const f = useCurrentFrame();
  const { fps } = useVideoConfig();
  const fr = scene(7).frames;
  const count = spring({ frame: f - ev("s7.count"), fps, config: { damping: 30, stiffness: 70 } });
  const pop = spring({ frame: f - ev("s7.count"), fps, config: springs.pop });
  const breathe = 1 + Math.sin(f / 18) * 0.012;
  return (
    <Exit frames={fr}>
      <div style={{ position: "absolute", left: 160, top: 96 }}>
        <Enter delay={4} y={30} blur={8}>
          <div style={{ fontFamily: MONO, fontSize: 84, fontWeight: 700, color: C.fg, letterSpacing: "-0.04em" }}>
            -O0 <span style={{ color: C.muted }}>→</span> <span style={{ color: C.lime }}>-O2</span>
          </div>
        </Enter>
        <Enter delay={12} y={16}>
          <div style={{ fontFamily: SANS, fontSize: 28, color: C.muted, marginTop: 6 }}>
            8×8 matmul + relu · tiny-gpu instructions executed
          </div>
        </Enter>
      </div>

      {ROWS.map((r, i) => {
        const at = ev("s7.bars") + i * 9;
        const g = spring({ frame: f - at, fps, config: springs.smooth });
        const w = r.w * g;
        const hero = i === 2;
        return (
          <div key={r.l} style={{ position: "absolute", left: 160, top: 360 + i * 138, height: 84, display: "flex", alignItems: "center" }}>
            <Enter delay={at - 4} y={14} x={-20} scale={1}>
              <div style={{ width: 150, fontFamily: MONO, fontSize: 44, fontWeight: 700, color: hero ? C.lime : C.fg }}>{r.l}</div>
            </Enter>
            <div style={{ width: w, height: 70, borderRadius: 14, background: r.c, opacity: hero ? 1 : 0.85,
              boxShadow: hero ? `0 12px 26px -16px ${alpha(C.accent, 0.5)}` : undefined }} />
            <div style={{ marginLeft: 22, opacity: Math.min(1, g * 1.4) }}>
              <div style={{ fontFamily: MONO, fontSize: 40, fontWeight: 700, color: hero ? C.lime : C.fg, fontVariantNumeric: "tabular-nums" }}>
                {Math.round(r.n * g).toLocaleString("en-US")}
              </div>
            </div>
          </div>
        );
      })}

      <div style={{ position: "absolute", left: 1290, top: 350, width: 540, textAlign: "center", opacity: Math.min(1, pop * 1.3),
        transform: `scale(${0.85 + 0.15 * pop})` }}>
        <div style={{ fontFamily: SANS, fontSize: 176, fontWeight: 800, letterSpacing: "-0.05em", color: C.lime, lineHeight: 1,
          fontVariantNumeric: "tabular-nums" }}>
          −{Math.round(55 * count)}%
        </div>
        <div style={{ fontFamily: SANS, fontSize: 38, fontWeight: 500, color: C.fg, marginTop: 6 }}>fewer instructions</div>
      </div>

      <div style={{ position: "absolute", left: 160, top: 800, display: "flex", gap: 22 }}>
        <Enter delay={ev("s7.loads")} y={20}>
          <div style={{ fontFamily: MONO, fontSize: 27, color: C.fg, background: C.codeBg, border: `1px solid ${C.line}`, borderRadius: 12, padding: "12px 22px" }}>
            global loads <span style={{ color: C.muted }}>1,088 →</span> <span style={{ color: C.cyan, fontWeight: 700 }}>576</span> <span style={{ color: C.muted }}>(−47%)</span>
          </div>
        </Enter>
        <Enter delay={ev("s7.measured")} y={20}>
          <div style={{ fontFamily: SANS, fontSize: 27, fontWeight: 600, color: C.fg, background: alpha(C.ok, 0.1), border: `1px solid ${alpha(C.ok, 0.45)}`,
            borderRadius: 12, padding: "12px 22px", display: "flex", alignItems: "center", gap: 10 }}>
            <Check size={26} p={interpolate(f, [ev("s7.measured") + 4, ev("s7.measured") + 16], [0, 1], clamp)} /> measured on the real simulator
          </div>
        </Enter>
      </div>
    </Exit>
  );
};

/* ------------------------------------------------------------------ S8 */
const hash = (n: number) => { const x = Math.sin(n * 127.1 + 311.7) * 43758.5453; return x - Math.floor(x); };
export const Scene8: React.FC = () => {
  const f = useCurrentFrame();
  const { fps } = useVideoConfig();
  const fr = scene(8).frames;
  const g0 = ev("s8.grid");
  const n = spring({ frame: f - g0, fps, config: { damping: 40, stiffness: 55 } });
  const stamp = ev("s8.stamp");
  return (
    <Exit frames={fr}>
      <div style={{ position: "absolute", left: 200, top: 238 }}>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(10, 40px)", gap: 9 }}>
          {Array.from({ length: 100 }).map((_, i) => {
            const r = Math.floor(i / 10), c = i % 10;
            const enter = interpolate(f, [i * 0.3, i * 0.3 + 14], [0, 1], { ...clamp, easing: ease.out });
            const at = g0 + (r + c) * 2.1 + hash(i) * 7;
            const t = interpolate(f, [at, at + 5, at + 16], [0, 1, 2], clamp);
            const col = t < 1 ? mix(C.peIdle, C.accent, t) : mix(C.accent, C.ok, t - 1);
            return <div key={i} style={{ width: 40, height: 40, borderRadius: 9, background: col, opacity: enter, transform: `scale(${0.5 + 0.5 * enter})`,
              border: t === 0 ? `1px solid ${C.line}` : "none", }} />;
          })}
        </div>
      </div>

      <div style={{ position: "absolute", left: 820, top: 190 }}>
        <Enter delay={6} y={30} blur={10}>
          <div style={{ fontFamily: SANS, fontSize: 200, fontWeight: 800, letterSpacing: "-0.05em", color: C.fg, lineHeight: 1, fontVariantNumeric: "tabular-nums" }}>
            {Math.round(5000 * n).toLocaleString("en-US")}
          </div>
        </Enter>
        <Enter delay={14} y={16}>
          <div style={{ fontFamily: SANS, fontSize: 46, fontWeight: 500, color: C.fg, marginTop: 6 }}>random programs, fuzzed</div>
        </Enter>
        <Enter delay={22} y={14}>
          <div style={{ fontFamily: MONO, fontSize: 25, color: C.muted, marginTop: 14, lineHeight: 1.55 }}>
            gpu -O0 · -O1 · -O2 &nbsp;/&nbsp; tpu -O0 · -O1<br />every result checked against the numpy oracle
          </div>
        </Enter>
      </div>

      <div style={{ position: "absolute", left: 820, top: 700 }}>
        <Enter delay={stamp} y={30} blur={10} cfg="snappy">
          <div style={{ fontFamily: SANS, fontSize: 92, fontWeight: 800, letterSpacing: "-0.04em", color: C.fg, lineHeight: 1 }}>
            Nothing <span style={{ color: C.lime }}>faked.</span>
          </div>
        </Enter>
        <div style={{ display: "flex", gap: 14, marginTop: 24 }}>
          {["real simulator", "real RTL", "numpy oracle"].map((t, i) => (
            <Enter key={t} delay={stamp + 8 + i * 5} y={16}>
              <div style={{ fontFamily: SANS, fontSize: 26, fontWeight: 600, color: C.fg, background: alpha(C.ok, 0.1), border: `1px solid ${alpha(C.ok, 0.45)}`,
                borderRadius: 12, padding: "9px 18px", display: "flex", alignItems: "center", gap: 9 }}>
                <Check size={24} p={interpolate(f, [stamp + 12 + i * 5, stamp + 24 + i * 5], [0, 1], clamp)} />{t}
              </div>
            </Enter>
          ))}
        </div>
      </div>
    </Exit>
  );
};

/* ------------------------------------------------------------------ S9 */
export const Scene9: React.FC = () => {
  const f = useCurrentFrame();
  const { fps } = useVideoConfig();
  const fr = scene(9).frames;
  const click = ev("s9.click");
  const brand = spring({ frame: f - ev("s9.brand"), fps, config: springs.smooth });
  const fin = ev("s9.final");
  const press = interpolate(f, [click - 1, click + 3, click + 10], [0, 1, 0], clamp);
  const glow = 1 + Math.sin(f / 10) * 0.18;
  const finalPulse = interpolate(f, [fin, fin + 8, fin + 30], [0, 1, 0], clamp);
  const path: Way[] = [[0, 1500, 1000], [ev("s9.head") + 20, 1500, 1000], [click - 2, 1150 - 40, 700 - 10], [click + 40, 1190, 740]];
  const rays = Array.from({ length: 16 });
  return (
    <Exit frames={fr}>
      {/* brand lockup */}
      <div style={{ position: "absolute", left: 0, right: 0, top: 70, height: 130 }}>
        <div style={{ position: "absolute", left: 960 - 60 - 251 * brand, top: 0 }}><PEMark size={120} delay={4} wave={0.18} /></div>
        <div style={{ position: "absolute", left: 960 - 311 + 148, top: 25, opacity: brand, transform: `translateX(${(1 - brand) * -30}px)`, filter: `blur(${(1 - brand) * 8}px)` }}>
          <Wordmark size={78} />
        </div>
      </div>

      <div style={{ position: "absolute", left: 210, right: 210, top: 285, display: "flex", justifyContent: "center" }}>
        <Words text="Compile your first tensor program." size={112} delay={ev("s9.head")} per={5} hl={["program"]} />
      </div>

      {/* CTA button */}
      <div style={{ position: "absolute", left: 0, right: 0, top: 640, display: "flex", justifyContent: "center" }}>
        <Enter delay={8} y={40} scale={0.92} cfg="pop">
          <div style={{ position: "relative" }}>
            {rays.map((_, i) => {
              const d = f - click;
              if (d < 0 || d > 34) return null;
              const t = ease.out(d / 34);
              const a = (i / rays.length) * Math.PI * 2 + hash(i) * 0.4, dist = 120 + hash(i + 9) * 150;
              return <div key={i} style={{ position: "absolute", left: 360 + Math.cos(a) * dist * t - 11, top: 55 + Math.sin(a) * dist * t - 11, width: 22, height: 22,
                borderRadius: 6, background: [C.lime, C.cyan, C.green][i % 3], opacity: 1 - t, transform: `scale(${1 - t * 0.5}) rotate(${t * 90}deg)` }} />;
            })}
            <div style={{ width: 720, height: 110, borderRadius: 20, background: C.accent, color: C.onAccent, fontFamily: SANS, fontSize: 44, fontWeight: 700,
              letterSpacing: "-0.02em", display: "flex", alignItems: "center", justifyContent: "center", gap: 18,
              transform: `scale(${1 - press * 0.04 + finalPulse * 0.03})`,
              boxShadow: `0 ${10 + finalPulse * 6}px 26px -14px ${alpha(C.accent, 0.55)}` }}>
              Open the Compiler Explorer
              <svg width="38" height="38" viewBox="0 0 24 24"><path d="M5 12h14M13 6l6 6-6 6" fill="none" stroke={C.onAccent} strokeWidth="2.6" strokeLinecap="round" strokeLinejoin="round" /></svg>
            </div>
          </div>
        </Enter>
      </div>

      <div style={{ position: "absolute", left: 0, right: 0, top: 800, display: "flex", justifyContent: "center" }}>
        <Enter delay={20} y={16}>
          <div style={{ fontFamily: MONO, fontSize: 32, color: finalPulse > 0.05 ? C.fg : C.muted, display: "flex", alignItems: "center", gap: 14 }}>
            tiny-compiler.vercel.app/app
          </div>
        </Enter>
      </div>

      <div style={{ position: "absolute", left: 0, top: 0, transform: "scale(1.7)", transformOrigin: "0 0" }}>
        <Cursor f={f} path={path.map(([a, x, y]) => [a, x / 1.7, y / 1.7] as Way)} clicks={[click]} show={[ev("s9.head") + 10, click + 36]} />
      </div>
    </Exit>
  );
};
