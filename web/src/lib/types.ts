// The shape of the JSON that runtime/web_api.py returns.

export interface Pass {
  name: string;
  ir: string;
}

export interface Bounds {
  min: number;
  max: number;
}

export interface GpuEstimate {
  launches: number;
  threads: number;
  instructions: Bounds;
  cycles: Bounds;
  loads: number;
  stores: number;
  shared_loads: number;
  shared_stores: number;
  registers: number;
}

export interface GpuRun {
  outputs: Record<string, number[]>;
  exact: boolean;
  reservedIntact: boolean;
  cycles: number;
  instructions: number;
  loads: number;
  stores: number;
  shared: number;
  barriers: number;
}

export interface GpuResult {
  ok: boolean;
  error?: string;
  passes: Pass[];
  asm?: string;
  kernels?: string[];
  estimate?: GpuEstimate;
  run?: GpuRun | null;
}

export interface TtpuProgram {
  format: string;
  version: number;
  name: string;
  inputs: string[];
  outputs: string[];
  buffers: Record<string, { dtype: string; shape: number[]; kind: string; data?: number[] }>;
  commands: Array<Record<string, unknown>>;
  estimate: { passes: number; rtl_cycles: number; host_ops: Record<string, number>; utilization: number | null };
}

export interface TpuResult {
  ok: boolean;
  error?: string;
  passes: Pass[];
  tile?: string;
  program?: TtpuProgram;
  estimate?: TtpuProgram["estimate"];
}

export interface Analysis {
  ok: boolean;
  error?: string;
  opt?: number;
  seed?: number;
  inputs?: Record<string, number[]>;
  outputNames?: string[];
  oracle?: number[][] | null;
  oracleError?: string | null;
  gpu?: GpuResult;
  tpu?: TpuResult;
}
