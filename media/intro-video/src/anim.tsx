import React from "react";
import { interpolate, spring, useCurrentFrame, useVideoConfig } from "remotion";
import { C, SANS, ease, springs } from "./theme";

export const clamp = { extrapolateLeft: "clamp", extrapolateRight: "clamp" } as const;

export const useSpringAt = (delay: number, cfg: keyof typeof springs = "smooth") => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  return spring({ frame: frame - delay, fps, config: springs[cfg] });
};

/** fade + rise + scale (+ optional blur): the workhorse entrance */
export const Enter: React.FC<{
  delay?: number; y?: number; scale?: number; blur?: number; x?: number;
  cfg?: keyof typeof springs; style?: React.CSSProperties; children: React.ReactNode;
}> = ({ delay = 0, y = 36, x = 0, scale = 0.96, blur = 0, cfg = "smooth", style, children }) => {
  const p = useSpringAt(delay, cfg);
  return (
    <div style={{
      opacity: Math.min(1, p * 1.4),
      transform: `translate(${(1 - p) * x}px, ${(1 - p) * y}px) scale(${scale + (1 - scale) * p})`,
      filter: blur ? `blur(${(1 - p) * blur}px)` : undefined,
      willChange: "transform, opacity", ...style,
    }}>{children}</div>
  );
};

export const Words: React.FC<{
  text: string; delay?: number; per?: number; size: number; weight?: number; color?: string;
  hl?: string[]; hlColor?: string; align?: "center" | "left"; style?: React.CSSProperties;
}> = ({ text, delay = 0, per = 4, size, weight = 700, color = C.fg, hl = [], hlColor = C.lime, align = "center", style }) => {
  const parts = text.split(" ");
  return (
    <div style={{
      display: "flex", flexWrap: "wrap", columnGap: size * 0.26, rowGap: 0,
      justifyContent: align === "center" ? "center" : "flex-start",
      fontFamily: SANS, fontSize: size, fontWeight: weight, letterSpacing: "-0.035em", lineHeight: 1.06, color, ...style,
    }}>
      {parts.map((w, i) => (
        <Enter key={i} delay={delay + i * per} y={size * 0.28} scale={0.98} blur={10} cfg="snappy">
          <span style={{ color: hl.includes(w.replace(/[.,]/g, "")) ? hlColor : undefined }}>{w}</span>
        </Enter>
      ))}
    </div>
  );
};

/** scene exit: faster than the entrance, applied to the whole scene */
export const Exit: React.FC<{ frames: number; children: React.ReactNode }> = ({ frames, children }) => {
  const frame = useCurrentFrame();
  const t = interpolate(frame, [frames - 9, frames - 1], [0, 1], { ...clamp, easing: ease.in });
  return (
    <div style={{ position: "absolute", inset: 0, opacity: 1 - t, transform: `scale(${1 + t * 0.045})`, filter: `blur(${t * 10}px)` }}>
      {children}
    </div>
  );
};

export const Dot: React.FC<{ c: string; s?: number }> = ({ c, s = 10 }) => (
  <span style={{ display: "inline-block", width: s, height: s, borderRadius: s, background: c }} />
);

export const Check: React.FC<{ size?: number; color?: string; p?: number }> = ({ size = 18, color = C.ok, p = 1 }) => (
  <svg width={size} height={size} viewBox="0 0 20 20" style={{ display: "block" }}>
    <path d="M4 10.5l4 4 8-9" fill="none" stroke={color} strokeWidth={2.6} strokeLinecap="round" strokeLinejoin="round"
      strokeDasharray={20} strokeDashoffset={20 * (1 - p)} />
  </svg>
);
