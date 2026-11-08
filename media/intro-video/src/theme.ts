import { Easing } from "remotion";
import { loadFont as loadInter } from "@remotion/google-fonts/Inter";
import { loadFont as loadMono } from "@remotion/google-fonts/JetBrainsMono";

// Tokens copied from web/src/styles/global.css (dark theme) and web/public/og.png.
export const C = {
  bg: "#fbfaf7",
  fg: "#1d1c1a",
  muted: "#6b6861",
  line: "#dedbd3",
  panel: "#ffffff",
  codeBg: "#f3f1ec",
  accent: "#2f6fd6", // the app's primary blue
  ok: "#1b7f4b",
  bad: "#b3261e",
  lime: "#2f6fd6", // hero highlight = accent (no neon)
  cyan: "#1f7a8c", // PE palette, darkened for paper
  green: "#1b7f4b",
  peIdle: "#ebe8e1",
  onAccent: "#ffffff",
} as const;

const inter = loadInter("normal", { weights: ["400", "500", "600", "700", "800"], subsets: ["latin"] });
const mono = loadMono("normal", { weights: ["400", "500", "700"], subsets: ["latin"] });
export const SANS = `${inter.fontFamily}, system-ui, sans-serif`;
export const MONO = `${mono.fontFamily}, ui-monospace, SFMono-Regular, Menlo, monospace`;

export const ease = {
  out: Easing.bezier(0.16, 1, 0.3, 1),
  inOut: Easing.bezier(0.65, 0, 0.35, 1),
  in: Easing.bezier(0.5, 0, 0.9, 0.4),
  cam: Easing.bezier(0.45, 0, 0.15, 1),
};
export const springs = {
  smooth: { damping: 24, stiffness: 120, mass: 0.9 },
  snappy: { damping: 20, stiffness: 220, mass: 0.7 },
  pop: { damping: 13, stiffness: 190, mass: 0.7 },
};

const hex = (h: string) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
export const mix = (a: string, b: string, t: number) => {
  const A = hex(a), B = hex(b);
  const k = Math.max(0, Math.min(1, t));
  return `rgb(${A.map((v, i) => Math.round(v + (B[i] - v) * k)).join(",")})`;
};
export const alpha = (h: string, a: number) => {
  const [r, g, b] = hex(h);
  return `rgba(${r},${g},${b},${a})`;
};
