/**
 * state-schema.ts - TinyTPU per-cycle state types (RTL↔Viz contract)
 *
 * Sync requirement: this file and docs/STATE_SCHEMA.md define the same contract.
 * Any field added, removed, or renamed here must be reflected there, and vice versa.
 *
 * All integer values use `number` (JS number is 64-bit float, sufficient for int32).
 * `weight` and `actIn` are signed int8 (−128..127).
 * `psum` and south `value` are signed int32.
 */

// ---------------------------------------------------------------------------
// FSM state
// ---------------------------------------------------------------------------

/**
 * Controller FSM state for a given cycle.
 *
 * "DONE" is a harness-derived convenience value - the hardware FSM has no DONE
 * state; after DRAIN it returns to IDLE. The harness reports "DONE" on the single
 * cycle where the hardware `done` signal is asserted.
 */
export type FsmState = "IDLE" | "LOAD_WEIGHTS" | "STREAM" | "DRAIN" | "DONE";

// ---------------------------------------------------------------------------
// Processing element state
// ---------------------------------------------------------------------------

/**
 * State of one PE (processing element) for a single clock cycle.
 * 16 entries per CycleState, ordered row-major: index i*4 + j = PE[row=i][col=j].
 */
export interface PEState {
  /** Row index (0–3). */
  readonly row: number;
  /** Column index (0–3). */
  readonly col: number;
  /**
   * Stationary weight loaded into this PE.
   * Hardware source: dbg_weight[i][j] (signed int8).
   */
  readonly weight: number;
  /**
   * Activation input to this PE on this cycle.
   *
   * Derived in the harness - not a direct register read:
   *   actIn[i][j] = (j === 0) ? westInputs[i] : dbg_act[i][j-1]
   *
   * dbg_act[i][j] is pe[i][j].act_out (registered passthrough of the previous
   * cycle's act_in); actIn reconstructs the current-cycle activation input.
   * Signed int8 range (−128..127).
   */
  readonly actIn: number;
  /**
   * Registered partial sum output of this PE on this cycle.
   * Hardware source: dbg_psum[i][j] (signed int32).
   */
  readonly psum: number;
  /**
   * True when this PE is performing a MAC this cycle.
   * Derived: fsmState === "STREAM" && actIn !== 0.
   */
  readonly active: boolean;
}

// ---------------------------------------------------------------------------
// South (bottom-edge) output
// ---------------------------------------------------------------------------

/** State of one column's bottom-edge output for a single clock cycle. */
export interface SouthOutput {
  /** Column index (0–3). */
  readonly col: number;
  /**
   * Accumulated partial sum leaving the bottom of this column.
   * Hardware source: dbg_south[j] = psum_south[j] (signed int32).
   */
  readonly value: number;
  /**
   * True when this column's output carries a final result element on this cycle.
   * Computed in the harness from FSM state and internal cycle counters.
   */
  readonly valid: boolean;
}

// ---------------------------------------------------------------------------
// Top-level per-cycle state
// ---------------------------------------------------------------------------

/** Complete hardware state snapshot for one clock cycle. */
export interface CycleState {
  /**
   * Cycle index, starting at 0 on the cycle start fires.
   * Increments by 1 each tick until done is true.
   */
  readonly cycle: number;
  /** Controller FSM state for this cycle. */
  readonly fsmState: FsmState;
  /**
   * State of all 16 PEs, row-major order.
   * Index i*4 + j corresponds to PE[row=i][col=j].
   * Length is always N*N = 16.
   */
  readonly pes: readonly PEState[];
  /**
   * Activations entering the left (west) edge of each row this cycle.
   * westInputs[i] = act_west[i]; hardware source: dbg_west[i].
   * Length is always N = 4. Signed int8 range (−128..127).
   * Zero when FSM is not in STREAM or row i is outside its skew window.
   */
  readonly westInputs: readonly number[];
  /**
   * Bottom-edge outputs, one per column. Length is always N = 4.
   */
  readonly southOutputs: readonly SouthOutput[];
  /**
   * True on the single cycle where the hardware done signal is asserted.
   * At this point all 16 result elements are captured and fsmState === "DONE".
   * For N=4 this fires 14 cycles after start (matches golden.py expected_cycles()).
   */
  readonly done: boolean;
}
