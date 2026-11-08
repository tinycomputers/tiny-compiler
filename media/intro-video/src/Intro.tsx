import React from "react";
import { interpolate, useCurrentFrame } from "remotion";
import { C, MONO, SANS, alpha, ease } from "./theme";
import { Enter, Exit, Words, clamp } from "./anim";
import { PEMark, Wordmark } from "./Logo";
import { Code } from "./Explorer";
import { ev, scene } from "./sync";

const Card: React.FC<{ title: string; sub: string; kind: "gpu" | "tpu"; delay: number; x: number }> = ({ title, sub, kind, delay, x }) => {
  const f = useCurrentFrame();
  return (
    <Enter delay={delay} y={50} x={x} scale={0.94} cfg="pop">
      <div style={{ width: 450, height: 190, borderRadius: 20, background: C.panel, border: `1px solid ${C.line}`, display: "flex", alignItems: "center",
        gap: 26, padding: "0 26px", boxShadow: "0 1px 2px rgba(20,18,14,.06), 0 16px 40px -24px rgba(20,18,14,.18)" }}>
        {kind === "tpu" ? <PEMark size={104} delay={delay + 4} wave={0.16} /> : (
          <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 22px)", gap: 6 }}>
            {Array.from({ length: 16 }).map((_, i) => {
              const k = (Math.sin(f / 7 + (i % 4) * 0.5 + Math.floor(i / 4) * 1.7) + 1) / 2;
              return <div key={i} style={{ width: 22, height: 22, borderRadius: 5, background: C.cyan, opacity: 0.25 + 0.75 * k }} />;
            })}
          </div>
        )}
        <div>
          <div style={{ fontFamily: SANS, fontSize: 42, fontWeight: 700, letterSpacing: "-0.03em", color: C.fg }}>{title}</div>
          <div style={{ fontFamily: SANS, fontSize: 20, color: C.muted, marginTop: 6, whiteSpace: "nowrap" }}>{sub}</div>
        </div>
      </div>
    </Enter>
  );
};

export const Scene1: React.FC = () => {
  const f = useCurrentFrame();
  const fr = scene(1).frames;
  const h2 = ev("s1.h2");
  const t = interpolate(f, [h2 - 6, h2 + 14], [0, 1], { ...clamp, easing: ease.inOut });
  const chipOut = interpolate(f, [h2 - 8, h2 + 4], [0, 1], { ...clamp, easing: ease.in });
  return (
    <Exit frames={fr}>
      <div style={{ position: "absolute", left: 0, right: 0, top: 0, display: "flex", justifyContent: "center",
        transform: `translateY(${380 - t * 215}px) scale(${1 - t * 0.48})`, transformOrigin: "50% 50%", opacity: 1 - t * 0.42 }}>
        <Words text="One tiny program." size={132} delay={ev("s1.h1")} per={5} />
      </div>
      <div style={{ position: "absolute", left: 0, right: 0, top: 530, display: "flex", justifyContent: "center",
        opacity: 1 - chipOut, transform: `translateY(${-chipOut * 28}px) scale(${1 - chipOut * 0.08})` }}>
        <Enter delay={ev("s1.code")} y={30} cfg="pop">
          <div style={{ fontFamily: MONO, fontSize: 52, padding: "20px 40px", borderRadius: 18, background: C.codeBg, border: `1px solid ${C.line}`,
            boxShadow: "0 1px 2px rgba(20,18,14,.06)", whiteSpace: "pre" }}>
            <Code text="C = relu(A @ B)" hot />
          </div>
        </Enter>
      </div>
      <div style={{ position: "absolute", left: 0, right: 0, top: 380, display: "flex", justifyContent: "center" }}>
        <Words text="Two different chips." size={132} delay={h2} per={5} hl={["chips"]} />
      </div>
      <div style={{ position: "absolute", left: 0, right: 0, top: 690, display: "flex", justifyContent: "center", gap: 64 }}>
        <Card title="tiny-gpu" sub="SIMT kernels · simulator" kind="gpu" delay={ev("s1.chips")} x={-90} />
        <Card title="tiny-tpu" sub="systolic tiles · real RTL" kind="tpu" delay={ev("s1.chips") + 6} x={90} />
      </div>
    </Exit>
  );
};

const LEVELS = [
  { l: "L1", n: "Source", s: ".tiny", ev: "s2.l1" },
  { l: "L2", n: "Graph IR", s: "SSA · typed · loc-tagged", ev: "s2.l2" },
  { l: "L3", n: "Tile IR · Kernel IR", s: "one lowering per chip", ev: "s2.l3" },
  { l: "L4", n: "The chips", s: "tiny-gpu · tiny-tpu", ev: "s2.l4" },
];

export const Scene2: React.FC = () => {
  const f = useCurrentFrame();
  const fr = scene(2).frames;
  const float = Math.sin(f / 34) * 4;
  return (
    <Exit frames={fr}>
      <div style={{ position: "absolute", left: 0, right: 0, top: 205, display: "flex", justifyContent: "center", alignItems: "center", gap: 56,
        transform: `translateY(${float}px)` }}>
        <PEMark size={236} delay={ev("s2.logo")} wave={0.17} />
        <div>
          <Enter delay={ev("s2.word")} y={34} blur={12} cfg="snappy"><Wordmark size={150} /></Enter>
          <Enter delay={ev("s2.tag")} y={20} blur={6}>
            <div style={{ fontFamily: SANS, fontSize: 46, fontWeight: 500, color: C.muted, letterSpacing: "-0.02em", marginTop: 20 }}>
              A small compiler for tensor programs
            </div>
          </Enter>
          <Enter delay={ev("s2.tag") + 26} y={16}>
            <div style={{ fontFamily: SANS, fontSize: 32, color: C.cyan, marginTop: 16, display: "flex", gap: 18 }}>
              <span>tiny-gpu</span><span style={{ opacity: 0.5 }}>·</span><span>tiny-tpu</span><span style={{ opacity: 0.5 }}>·</span><span>bit-exact vs numpy</span>
            </div>
          </Enter>
        </div>
      </div>
      <div style={{ position: "absolute", left: 0, right: 0, top: 690, display: "flex", justifyContent: "center", alignItems: "center", gap: 18 }}>
        {LEVELS.map((lv, i) => {
          const at = ev(lv.ev);
          const fresh = interpolate(f, [at, at + 6, at + 28], [0, 1, 0], clamp);
          return (
            <React.Fragment key={lv.l}>
              {i > 0 && <Enter delay={at - 3} x={-14} y={0} scale={1}><span style={{ color: C.muted, fontSize: 38, fontFamily: SANS }}>→</span></Enter>}
              <Enter delay={at} y={40} scale={0.94} cfg="pop">
                <div style={{ width: i === 2 ? 400 : 340, height: 142, borderRadius: 18, padding: "20px 26px", boxSizing: "border-box",
                  background: C.panel, border: `1px solid ${fresh > 0.01 ? alpha(C.lime, 0.2 + 0.6 * fresh) : C.line}`,
                  boxShadow: "0 1px 2px rgba(20,18,14,.06), 0 16px 40px -24px rgba(20,18,14,.18)" }}>
                  <div style={{ fontFamily: MONO, fontSize: 20, color: C.muted }}>{lv.l}</div>
                  <div style={{ fontFamily: SANS, fontSize: 33, fontWeight: 700, letterSpacing: "-0.02em", color: C.fg, marginTop: 4 }}>{lv.n}</div>
                  <div style={{ fontFamily: MONO, fontSize: 19, color: C.muted, marginTop: 6 }}>{lv.s}</div>
                </div>
              </Enter>
            </React.Fragment>
          );
        })}
      </div>
    </Exit>
  );
};
