import React from "react";
import { Easing, interpolate } from "remotion";
import { C, MONO, SANS, alpha, ease } from "./theme";
import { clamp } from "./anim";

/** Design-space geometry of the Compiler Explorer (px, before the 1.3 video scale). */
export const WIN = { w: 1200, h: 660, scale: 1.3 };
export const TAB = { y: 136, passes: 607, gpu: 691, tpu: 779, run: 885 };
export const BTN = { x: 1110, y: 72 };
export const PANEL = { lx: 24, lw: 524, rx: 560, rw: 616, top: 104, h: 536 };
export const LINE_H = 24;

/* ----------------------------------------------------------- browser chrome */
export const Chrome: React.FC = () => (
  <div style={{ position: "absolute", left: 0, top: 0, right: 0, height: 40, background: C.codeBg,
    borderBottom: `1px solid ${C.line}`, display: "flex", alignItems: "center", padding: "0 16px", gap: 8 }}>
    {[0, 1, 2].map((i) => <span key={i} style={{ width: 12, height: 12, borderRadius: 12, background: C.line }} />)}
    <div style={{ marginLeft: 24, height: 24, width: 380, borderRadius: 12, background: C.bg, border: `1px solid ${C.line}`,
      display: "flex", alignItems: "center", padding: "0 12px", fontFamily: SANS, fontSize: 12.5, color: C.muted, gap: 8 }}>
      <svg width="11" height="11" viewBox="0 0 12 12"><rect x="2" y="5" width="8" height="6" rx="1.5" fill="none" stroke={C.muted} strokeWidth="1.3" /><path d="M4 5V3.6a2 2 0 014 0V5" fill="none" stroke={C.muted} strokeWidth="1.3" /></svg>
      tiny-compiler.vercel.app/app
    </div>
  </div>
);

/* ----------------------------------------------------------- header controls */
const Select: React.FC<{ w: number; children: React.ReactNode }> = ({ w, children }) => (
  <div style={{ width: w, height: 34, border: `1px solid ${C.line}`, borderRadius: 6, background: C.panel, color: C.fg,
    display: "flex", alignItems: "center", justifyContent: "space-between", padding: "0 10px", fontFamily: SANS, fontSize: 14 }}>
    <span>{children}</span>
    <svg width="9" height="6" viewBox="0 0 10 6"><path d="M1 1l4 4 4-4" fill="none" stroke={C.muted} strokeWidth="1.6" strokeLinecap="round" /></svg>
  </div>
);

export const Header: React.FC<{ press: number; busy?: boolean }> = ({ press, busy }) => (
  <div style={{ position: "absolute", left: 24, right: 24, top: 40, height: 64, display: "flex", alignItems: "center",
    justifyContent: "space-between", fontFamily: SANS }}>
    <div style={{ display: "flex", alignItems: "baseline", gap: 12 }}>
      <span style={{ fontSize: 22, fontWeight: 700, color: C.fg, letterSpacing: "-0.01em" }}>Compiler Explorer</span>
      <span style={{ fontSize: 13, color: C.muted, textDecoration: "underline" }}>what is this?</span>
    </div>
    <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
      <Select w={110}>mm_relu</Select>
      <Select w={72}>-O2</Select>
      <span style={{ fontSize: 14, color: C.muted, marginLeft: 2 }}>seed</span>
      <div style={{ width: 60, height: 34, border: `1px solid ${C.line}`, borderRadius: 6, background: C.panel, color: C.fg,
        display: "flex", alignItems: "center", padding: "0 10px", fontSize: 14 }}>42</div>
      <div style={{ width: 132, height: 34, borderRadius: 6, background: C.accent, color: C.onAccent, fontWeight: 600, fontSize: 14,
        display: "flex", alignItems: "center", justifyContent: "center", border: `1px solid ${C.accent}`,
        transform: `scale(${1 - press * 0.05})`, filter: `brightness(${1 + press * 0.12})`,
        boxShadow: busy ? `0 0 0 3px ${alpha(C.accent, 0.25)}` : undefined }}>
        Compile &amp; run
      </div>
    </div>
  </div>
);

export const Panel: React.FC<{ x: number; w: number; children: React.ReactNode; style?: React.CSSProperties }> = ({ x, w, children, style }) => (
  <div style={{ position: "absolute", left: x, top: PANEL.top, width: w, height: PANEL.h, background: C.panel,
    border: `1px solid ${C.line}`, borderRadius: 8, padding: 12, boxSizing: "border-box", ...style }}>{children}</div>
);

/* ----------------------------------------------------------- tabs */
export type TabId = "passes" | "gpu" | "tpu" | "run";
const TABS: { id: TabId; label: string; w: number }[] = [
  { id: "passes", label: "Passes", w: 74 }, { id: "gpu", label: "tiny-gpu", w: 82 },
  { id: "tpu", label: "tiny-tpu", w: 82 }, { id: "run", label: "Run & compare", w: 118 },
];
export const Tabs: React.FC<{ active: TabId | null; press: TabId | null; pressAmt: number }> = ({ active, press, pressAmt }) => (
  <div style={{ display: "flex", gap: 6, fontFamily: SANS }}>
    {TABS.map((t) => {
      const on = active === t.id;
      const p = press === t.id ? pressAmt : 0;
      return (
        <div key={t.id} style={{ width: t.w, height: 34, borderRadius: 6, boxSizing: "border-box",
          border: `1px solid ${on ? C.accent : C.line}`, background: on ? C.accent : C.panel, color: on ? C.onAccent : C.fg,
          fontSize: 14, fontWeight: on ? 600 : 400, display: "flex", alignItems: "center", justifyContent: "center",
          transform: `scale(${1 - p * 0.06})` }}>{t.label}</div>
      );
    })}
  </div>
);

/* ----------------------------------------------------------- syntax */
const TOK = /(\b(?:input|output|const)\b|\bi8\b|\bi32\b|\brelu\b|\d+|[@=+()[\]:,])/g;
export const Code: React.FC<{ text: string; hot?: boolean }> = ({ text, hot }) => {
  if (text.startsWith("#")) return <span style={{ color: C.muted, fontStyle: "italic" }}>{text}</span>;
  return (
    <>
      {text.split(TOK).map((s, i) => {
        if (!s) return null;
        let color: string = C.fg, weight = 400;
        if (/^(input|output|const)$/.test(s)) color = C.accent;
        else if (/^i(8|32)$/.test(s)) color = C.cyan;
        else if (/^\d+$/.test(s)) color = C.cyan;
        else if (s === "relu") { weight = 700; color = hot ? C.lime : C.fg; }
        else if (/^[@=+]$/.test(s)) color = hot ? C.lime : C.muted;
        else if (/^[()[\]:,]$/.test(s)) color = C.muted;
        return <span key={i} style={{ color, fontWeight: weight }}>{s}</span>;
      })}
    </>
  );
};

/* ----------------------------------------------------------- cursor */
export type Way = [number, number, number]; // [frame, x, y] in design px
export const Cursor: React.FC<{ f: number; path: Way[]; clicks: number[]; show: [number, number] }> = ({ f, path, clicks, show }) => {
  let x = path[0][1], y = path[0][2];
  for (let i = 0; i < path.length - 1; i++) {
    const [f0, x0, y0] = path[i], [f1, x1, y1] = path[i + 1];
    if (f >= f0 && f <= f1) {
      const t = Easing.bezier(0.45, 0, 0.2, 1)((f - f0) / Math.max(1, f1 - f0));
      x = x0 + (x1 - x0) * t; y = y0 + (y1 - y0) * t;
    } else if (f > f1) { x = x1; y = y1; }
  }
  const vis = interpolate(f, [show[0], show[0] + 8, show[1] - 8, show[1]], [0, 1, 1, 0], clamp);
  const near = clicks.map((c) => f - c).find((d) => d >= 0 && d < 16);
  const press = near === undefined ? 0 : near < 5 ? 1 : 0;
  return (
    <div style={{ position: "absolute", left: 0, top: 0, opacity: vis, pointerEvents: "none", zIndex: 50 }}>
      {clicks.map((c, i) => {
        const d = f - c;
        if (d < 0 || d > 18) return null;
        const t = ease.out(d / 18);
        return <div key={i} style={{ position: "absolute", left: x - 22, top: y - 22, width: 44, height: 44, borderRadius: 44,
          border: `2px solid ${alpha(C.fg, 0.8)}`, opacity: 1 - t, transform: `scale(${0.3 + t * 1.2})` }} />;
      })}
      <svg width="26" height="30" viewBox="0 0 26 30" style={{ position: "absolute", left: x - 3, top: y - 2,
        transform: `scale(${1 - press * 0.12})`, filter: "drop-shadow(0 2px 3px rgba(20,18,14,.25))" }}>
        <path d="M3 2v22l5.6-5.2 3.6 8 4-1.8-3.6-7.8H20z" fill="#fff" stroke="#161513" strokeWidth="1.8" strokeLinejoin="round" />
      </svg>
    </div>
  );
};

/** pressed amount (0..1) for a click frame */
export const pressAt = (f: number, c: number) => interpolate(f, [c - 1, c + 2, c + 9], [0, 1, 0], clamp);
