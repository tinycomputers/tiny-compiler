import timing from "./timing.json";

export const FPS = timing.fps;
export const TOTAL = timing.total;
export const scene = (i: number) => timing.scenes[i - 1];
export const sec = (s: number) => Math.round(s * FPS);
const E = timing.events as Record<string, { scene: number; t: number }>;
/** scene-local frame of a named event */
export const ev = (name: string) => sec(E[name].t);
/** global frame of a named event */
export const evG = (name: string) => scene(E[name].scene).start + sec(E[name].t);
export const words = timing.words as { t: string; s: number; d: number }[][];
