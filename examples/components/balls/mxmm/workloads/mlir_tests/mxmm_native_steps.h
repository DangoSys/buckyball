#pragma once

extern "C" void _mlir_ciface_step_m1_n16_first();
extern "C" void _mlir_ciface_step_m1_n16_last();
extern "C" void _mlir_ciface_step_m1_n16_single();
extern "C" void _mlir_ciface_step_m1_n48_first();
extern "C" void _mlir_ciface_step_m1_n48_last();
extern "C" void _mlir_ciface_step_m1_n48_single();
extern "C" void _mlir_ciface_step_m16_n16_first();
extern "C" void _mlir_ciface_step_m16_n16_last();
extern "C" void _mlir_ciface_step_m16_n16_single();
extern "C" void _mlir_ciface_step_m16_n48_first();
extern "C" void _mlir_ciface_step_m16_n48_last();
extern "C" void _mlir_ciface_step_m16_n48_single();
extern "C" void _mlir_ciface_step_m32_n16_first();
extern "C" void _mlir_ciface_step_m32_n16_last();
extern "C" void _mlir_ciface_step_m32_n16_single();
extern "C" void _mlir_ciface_step_m32_n48_first();
extern "C" void _mlir_ciface_step_m32_n48_last();
extern "C" void _mlir_ciface_step_m32_n48_single();
static void (*steps[3][2][3])() = {
    {{_mlir_ciface_step_m1_n16_first, _mlir_ciface_step_m1_n16_last,
      _mlir_ciface_step_m1_n16_single},
     {_mlir_ciface_step_m1_n48_first, _mlir_ciface_step_m1_n48_last,
      _mlir_ciface_step_m1_n48_single}},
    {{_mlir_ciface_step_m16_n16_first, _mlir_ciface_step_m16_n16_last,
      _mlir_ciface_step_m16_n16_single},
     {_mlir_ciface_step_m16_n48_first, _mlir_ciface_step_m16_n48_last,
      _mlir_ciface_step_m16_n48_single}},
    {{_mlir_ciface_step_m32_n16_first, _mlir_ciface_step_m32_n16_last,
      _mlir_ciface_step_m32_n16_single},
     {_mlir_ciface_step_m32_n48_first, _mlir_ciface_step_m32_n48_last,
      _mlir_ciface_step_m32_n48_single}}};
