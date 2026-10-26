/**
 * PEGrid - pure presentational SVG rendering of the 4×4 systolic array.
 *
 * Accepts a single CycleState snapshot and renders:
 *   - 4×4 PE cells colored by hardware state (idle / weight-loaded / active / result)
 *   - Weight and psum values inside each cell
 *   - Activation-in indicator (bottom-left of cell, appears when actIn ≠ 0)
 *   - West input tokens on the left edge (animated, appear during STREAM)
 *   - South output tokens below the grid (highlighted when valid)
 *   - Static flow arrows indicating activation (→) and psum (↓) direction
 *
 * Color transitions use CSS `transition` so CSS variables resolve natively.
 * Token appearance/disappearance uses Framer Motion for spring-based animation.
 * `prefers-reduced-motion` disables all transitions (snaps to new state).
 *
 * RULE: no WASM calls here. No side effects. Keyed elements for stable identity.
 */

import { motion, AnimatePresence, useReducedMotion } from "framer-motion";
import type { ReactNode } from "react";
import type { CycleState, PEState, FsmState } from "../lib/state-schema";

// ─── Layout constants (SVG user-units, 1:1 with px at natural size) ──────────

const CELL = 88;
const GAP = 10;
const N = 4;
const WEST_W = 70;
const TOP_H = 40;
const SOUTH_H = 64;
const GRID_SIDE = N * CELL + (N - 1) * GAP; // 382
const SVG_W = WEST_W + GRID_SIDE + 12; // 464
const SVG_H = TOP_H + GRID_SIDE + SOUTH_H; // 486

/** SVG x-origin of a PE column (left edge of cell). */
function cellX(col: number): number {
  return WEST_W + col * (CELL + GAP);
}
/** SVG y-origin of a PE row (top edge of cell). */
function cellY(row: number): number {
  return TOP_H + row * (CELL + GAP);
}

// ─── PE state color logic ─────────────────────────────────────────────────────

function peFill(pe: PEState, fsm: FsmState): string {
  if (pe.active) return "var(--pe-active)";
  if ((fsm === "DRAIN" || fsm === "DONE") && pe.psum !== 0) return "var(--pe-result)";
  if (pe.weight !== 0) return "var(--pe-weight-loaded)";
  return "var(--pe-idle)";
}

function pePsumColor(pe: PEState, fsm: FsmState): string {
  if (pe.active) return "var(--primary-foreground)";
  if ((fsm === "DRAIN" || fsm === "DONE") && pe.psum !== 0) return "var(--primary-foreground)";
  return "var(--foreground)";
}

/** Clamp a number to fit in the cell (≤ 6 chars). Full value in SVG <title>. */
function fmt(n: number): string {
  const s = n.toString();
  return s.length > 6 ? s.slice(0, 5) + "…" : s;
}

// ─── Static flow arrows ───────────────────────────────────────────────────────

function FlowArrows() {
  const arrows: ReactNode[] = [];

  // Horizontal: activations flow right (→) between PE columns
  for (let row = 0; row < N; row++) {
    for (let col = 0; col < N - 1; col++) {
      const x1 = cellX(col) + CELL + 1;
      const x2 = cellX(col + 1) - 1;
      const ay = cellY(row) + CELL / 2;
      arrows.push(
        <g key={`h-${row}-${col}`}>
          <line x1={x1} y1={ay} x2={x2 - 4} y2={ay} strokeWidth={1} />
          <polygon
            points={`${x2 - 6},${ay - 3} ${x2},${ay} ${x2 - 6},${ay + 3}`}
          />
        </g>,
      );
    }
  }

  // Vertical: partial sums flow down (↓) between PE rows
  for (let row = 0; row < N - 1; row++) {
    for (let col = 0; col < N; col++) {
      const ax = cellX(col) + CELL / 2;
      const y1 = cellY(row) + CELL + 1;
      const y2 = cellY(row + 1) - 1;
      arrows.push(
        <g key={`v-${row}-${col}`}>
          <line x1={ax} y1={y1} x2={ax} y2={y2 - 4} strokeWidth={1} />
          <polygon
            points={`${ax - 3},${y2 - 6} ${ax},${y2} ${ax + 3},${y2 - 6}`}
          />
        </g>,
      );
    }
  }

  return (
    <g
      fill="var(--muted-foreground)"
      stroke="var(--muted-foreground)"
      opacity={0.3}
    >
      {arrows}
    </g>
  );
}

// ─── PE cell ──────────────────────────────────────────────────────────────────

interface PECellProps {
  pe: PEState;
  fsm: FsmState;
  /** When true, all CSS transitions are disabled (prefers-reduced-motion). */
  reduced: boolean;
}

function PECell({ pe, fsm, reduced }: PECellProps) {
  const x = cellX(pe.col);
  const y = cellY(pe.row);
  const fill = peFill(pe, fsm);
  const psumColor = pePsumColor(pe, fsm);
  const colorTr = reduced ? "none" : "fill 120ms ease-out";
  const motionTr = reduced ? { duration: 0 } : { duration: 0.18, ease: "easeOut" as const };

  return (
    <g
      role="img"
      aria-label={`PE[${pe.row}][${pe.col}] weight=${pe.weight} actIn=${pe.actIn} psum=${pe.psum}${pe.active ? " active" : ""}`}
    >
      {/* Background - CSS transition for CSS-var color interpolation */}
      <rect
        x={x}
        y={y}
        width={CELL}
        height={CELL}
        rx={5}
        style={{ fill, transition: colorTr }}
      />
      {/* Border */}
      <rect
        x={x}
        y={y}
        width={CELL}
        height={CELL}
        rx={5}
        fill="none"
        stroke="var(--border)"
        strokeWidth={1}
      />

      {/* Weight - top-left corner */}
      <text
        x={x + 7}
        y={y + 13}
        fontSize={9}
        fontFamily="'Geist Mono', ui-monospace, monospace"
        fill="var(--muted-foreground)"
      >
        w={pe.weight}
      </text>

      {/* psum - center, large */}
      <text
        x={x + CELL / 2}
        y={y + CELL / 2 + 9}
        textAnchor="middle"
        fontSize={20}
        fontWeight="bold"
        fontFamily="'Geist Mono', ui-monospace, monospace"
        style={{ fill: psumColor, transition: colorTr }}
      >
        <title>{pe.psum}</title>
        {fmt(pe.psum)}
      </text>

      {/* actIn indicator - bottom-left, shown when activation is non-zero */}
      <AnimatePresence>
        {pe.actIn !== 0 && (
          <motion.g
            key={`actin-${pe.row}-${pe.col}`}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={motionTr}
          >
            <circle
              cx={x + 11}
              cy={y + CELL - 12}
              r={9}
              fill="var(--pe-active)"
              opacity={0.2}
            />
            <text
              x={x + 11}
              y={y + CELL - 9}
              textAnchor="middle"
              fontSize={9}
              fontFamily="'Geist Mono', ui-monospace, monospace"
              fill="var(--pe-active)"
            >
              {pe.actIn}
            </text>
          </motion.g>
        )}
      </AnimatePresence>
    </g>
  );
}

// ─── South output token ───────────────────────────────────────────────────────

interface SouthTokenProps {
  col: number;
  value: number;
  valid: boolean;
  reduced: boolean;
}

function SouthToken({ col, value, valid, reduced }: SouthTokenProps) {
  const x = cellX(col) + CELL / 2;
  const lineY1 = TOP_H + GRID_SIDE + 1;
  const circY = lineY1 + 28;
  const colorTr = reduced ? "none" : "fill 120ms ease-out, stroke 120ms ease-out";

  return (
    <g>
      <line
        x1={x}
        y1={lineY1}
        x2={x}
        y2={lineY1 + 16}
        style={{
          stroke: valid ? "var(--pe-result)" : "var(--border)",
          transition: colorTr,
        }}
        strokeWidth={valid ? 1.5 : 1}
      />
      <circle
        cx={x}
        cy={circY}
        r={20}
        style={{
          fill: valid ? "var(--pe-result)" : "var(--pe-idle)",
          opacity: valid ? 1 : 0.45,
          transition: colorTr,
        }}
      />
      <text
        x={x}
        y={circY + 5}
        textAnchor="middle"
        fontSize={12}
        fontWeight="bold"
        fontFamily="'Geist Mono', ui-monospace, monospace"
        style={{
          fill: valid ? "var(--primary-foreground)" : "var(--muted-foreground)",
          transition: colorTr,
        }}
      >
        <title>{value}</title>
        {fmt(value)}
      </text>
    </g>
  );
}

// ─── PEGrid (main export) ─────────────────────────────────────────────────────

export interface PEGridProps {
  /** Hardware state snapshot for a single clock cycle. Pure read - no side effects. */
  state: CycleState;
}

export function PEGrid({ state }: PEGridProps) {
  const reduced = useReducedMotion() ?? false;
  const tokenTr = reduced
    ? { duration: 0 }
    : { duration: 0.12, ease: "easeOut" as const };

  return (
    <div
      className="w-full overflow-x-auto"
      role="region"
      aria-label="4×4 systolic array PE grid"
    >
      <svg
        viewBox={`0 0 ${SVG_W} ${SVG_H}`}
        style={{ width: "100%", maxWidth: SVG_W, display: "block", margin: "0 auto" }}
      >
        {/* Flow direction arrows (static) */}
        <FlowArrows />

        {/* Column headers C0–C3 */}
        {Array.from({ length: N }, (_, col) => (
          <text
            key={col}
            x={cellX(col) + CELL / 2}
            y={TOP_H - 10}
            textAnchor="middle"
            fontSize={11}
            fontFamily="'Geist Mono', ui-monospace, monospace"
            fill="var(--muted-foreground)"
          >
            C{col}
          </text>
        ))}

        {/* Row labels R0–R3 */}
        {Array.from({ length: N }, (_, row) => (
          <text
            key={row}
            x={12}
            y={cellY(row) + CELL / 2 + 4}
            textAnchor="end"
            fontSize={11}
            fontFamily="'Geist Mono', ui-monospace, monospace"
            fill="var(--muted-foreground)"
          >
            R{row}
          </text>
        ))}

        {/* West input tokens - animate in/out as activations arrive */}
        <AnimatePresence>
          {state.westInputs
            .map((val, row) => ({ val, row }))
            .filter(({ val }) => val !== 0)
            .map(({ val, row }) => {
              const tcx = WEST_W / 2 - 2;
              const tcy = cellY(row) + CELL / 2;
              return (
                <motion.g
                  key={`west-${row}`}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -10 }}
                  transition={tokenTr}
                >
                  <circle
                    cx={tcx}
                    cy={tcy}
                    r={16}
                    fill="var(--pe-active)"
                    opacity={0.12}
                  />
                  <circle
                    cx={tcx}
                    cy={tcy}
                    r={16}
                    fill="none"
                    stroke="var(--pe-active)"
                    strokeWidth={1.5}
                  />
                  <text
                    x={tcx}
                    y={tcy + 4}
                    textAnchor="middle"
                    fontSize={12}
                    fontFamily="'Geist Mono', ui-monospace, monospace"
                    fontWeight="bold"
                    fill="var(--pe-active)"
                  >
                    {val}
                  </text>
                  {/* Arrow pointing into grid */}
                  <line
                    x1={tcx + 17}
                    y1={tcy}
                    x2={WEST_W - 4}
                    y2={tcy}
                    stroke="var(--pe-active)"
                    strokeWidth={1.5}
                  />
                  <polygon
                    points={`${WEST_W - 8},${tcy - 3} ${WEST_W - 2},${tcy} ${WEST_W - 8},${tcy + 3}`}
                    fill="var(--pe-active)"
                  />
                </motion.g>
              );
            })}
        </AnimatePresence>

        {/* PE cells */}
        {state.pes.map((pe) => (
          <PECell
            key={`pe-${pe.row}-${pe.col}`}
            pe={pe}
            fsm={state.fsmState}
            reduced={reduced}
          />
        ))}

        {/* South output tokens */}
        {state.southOutputs.map((so) => (
          <SouthToken
            key={`south-${so.col}`}
            col={so.col}
            value={so.value}
            valid={so.valid}
            reduced={reduced}
          />
        ))}
      </svg>
    </div>
  );
}
