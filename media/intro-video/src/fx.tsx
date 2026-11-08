import React from "react";
import { AbsoluteFill, interpolate, useCurrentFrame, useVideoConfig } from "remotion";
import { C, alpha } from "./theme";

/** Paper background with a faint, fixed hairline grid: the product's own light surface, no decoration. */
export const Background: React.FC = () => (
  <AbsoluteFill style={{ background: C.bg }}>
    <AbsoluteFill style={{
      backgroundImage: `linear-gradient(${alpha(C.line, 0.55)} 1px, transparent 1px), linear-gradient(90deg, ${alpha(C.line, 0.55)} 1px, transparent 1px)`,
      backgroundSize: "96px 96px", backgroundPosition: "-1px -1px",
      maskImage: "radial-gradient(ellipse at center, black 35%, transparent 80%)",
      WebkitMaskImage: "radial-gradient(ellipse at center, black 35%, transparent 80%)",
      opacity: 0.6,
    }} />
  </AbsoluteFill>
);

/** Thin progress rule along the bottom edge. */
export const Progress: React.FC = () => {
  const f = useCurrentFrame();
  const { durationInFrames } = useVideoConfig();
  const w = interpolate(f, [0, durationInFrames], [0, 100]);
  return <div style={{ position: "absolute", left: 0, bottom: 0, height: 3, width: `${w}%`, background: C.accent }} />;
};
