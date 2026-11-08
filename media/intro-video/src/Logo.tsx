import React from "react";
import { useCurrentFrame, useVideoConfig, spring } from "remotion";
import { C, SANS, mix, springs } from "./theme";

const STOPS = [C.lime, C.cyan, C.green, C.peIdle, C.peIdle, C.peIdle, C.peIdle, C.lime];
const colorAt = (k: number) => {
  const i = Math.floor(k), t = k - i;
  return mix(STOPS[i % 8], STOPS[(i + 1) % 8], t);
};

/** The brand mark: a 4x4 PE grid with the systolic wavefront (og.png). `wave` > 0 sets it moving. */
export const PEMark: React.FC<{ size: number; delay?: number; wave?: number; cols?: number }> = ({ size, delay = 0, wave = 0, cols = 4 }) => {
  const f = useCurrentFrame();
  const { fps } = useVideoConfig();
  const gap = size * 0.06, cell = (size - gap * (cols - 1)) / cols;
  const phase = wave ? (f * wave) % 7 : 0;
  const cells: React.ReactNode[] = [];
  for (let r = 0; r < cols; r++)
    for (let c = 0; c < cols; c++) {
      const d = r + c;
      const p = spring({ frame: f - delay - d * 3 - r, fps, config: springs.pop });
      const k = (((d - phase) % 7) + 7) % 7;
      const col = colorAt(k);
      const lit = k < 3 || k > 6.2;
      cells.push(
        <div key={`${r}${c}`} style={{
          position: "absolute", left: c * (cell + gap), top: r * (cell + gap), width: cell, height: cell,
          borderRadius: cell * 0.24, background: col,
          border: lit ? "none" : `1px solid ${C.line}`, boxSizing: "border-box",
          opacity: Math.min(1, p * 1.6), transform: `scale(${0.4 + 0.6 * p})`,
          boxShadow: undefined,
        }} />,
      );
    }
  return <div style={{ position: "relative", width: size, height: size }}>{cells}</div>;
};

export const Wordmark: React.FC<{ size: number }> = ({ size }) => (
  <span style={{ fontFamily: SANS, fontWeight: 800, fontSize: size, letterSpacing: "-0.04em", color: C.fg, lineHeight: 1 }}>
    TinyCompiler
  </span>
);
