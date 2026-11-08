// Real compiler output for examples/mm_relu.tiny (captured from `tinyc`, -O2).
export const PROGRAM = [
  "# The headline program: C = relu(A @ B).",
  "input A : i8[4, 8]",
  "input B : i8[8, 4]",
  "C = relu(A @ B)",
  "output C",
];

export const IR_BEFORE = [
  "func @main(%A: i8[4,8], %B: i8[8,4]) -> (i32[4,4]) {",
  "  %0 = matmul %A, %B : i32[4,4]",
  "  %1 = relu %0 : i32[4,4]",
  "  return %1",
  "}",
];

export const TGPU = [
  "; kernel k0_matmul: %1 = matmul  (mm_relu.tiny:4:5)",
  "; grid: 4 blocks x 4 threads; 8 shared words per block; 5 registers",
  "SET R0, 0",
  "MUL R1, R5, 8",
  "ADD R2, R6, 32",
  "ADD R3, R1, R6",
  "LD R3, R3",
  "SHST R6, R3",
  "ADD R3, R6, 4",
  "ADD R1, R1, R3",
  "LD R1, R1",
  "SHST R3, R1",
  "SYNCB",
  "SHLD R1, 0",
  "LD R3, R2",
  "MUL R4, R1, R3",
  "ADD R0, R0, R4",
  "ADD R2, R2, 4",
  "SHLD R1, 1",
  "LD R3, R2",
  "MUL R4, R1, R3",
  "ADD R0, R0, R4",
  "ADD R2, R2, 4",
  "SHLD R1, 2",
  "LD R3, R2",
  "MUL R4, R1, R3",
  "ADD R0, R0, R4",
  "ADD R2, R2, 4",
  "SHLD R1, 3",
  "LD R3, R2",
  "MUL R4, R1, R3",
  "ADD R0, R0, R4",
];

export const RESULT_PRETTY = "[[16161,15523,0,2949],\n [0,8580,1770,15819],\n [0,10549,1568,5105],\n [0,9847,5002,0]]";
export const RESULT = "[[16161,15523,0,2949],[0,8580,1770,15819],[0,10549,1568,5105],[0,9847,5002,0]]";
