/*
 * Copyright (C) 2023-2024 ETH Zurich and University of Bologna
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Authors: Luca Balboni <luca.balboni10@studio.unibo.it>
 *         Based on redmule_test.c by Victor Isachi
 *
 */

#include "magia_tile_utils.h"
#include "magia_utils.h"
#include "redmule_mm_utils.h"
#include "w_input.h"
#include "x_input.h"
#include "y_input.h"
#include "z_output.h"
#include <stdint.h>

#define M_SIZE 4
#define N_SIZE 64
#define K_SIZE 4
#define REFERENCE_K 64
#define DIFF_TH 0x0011

int main(void) {
  uint32_t tile_base = L1_BASE + get_hartid() * L1_TILE_OFFSET;
  uint32_t x_base = tile_base + 0x00012048;
  uint32_t w_base = tile_base + 0x00016048;
  uint32_t y_base = tile_base + 0x0001A048;
  for (unsigned i = 0; i < M_SIZE * N_SIZE; ++i)
    mmio16(x_base + 2 * i) = x_inp[i];
  for (unsigned i = 0; i < N_SIZE * K_SIZE; ++i)
    mmio16(w_base + 2 * i) = w_inp[(i / K_SIZE) * REFERENCE_K + i % K_SIZE];
  for (unsigned i = 0; i < M_SIZE * K_SIZE; ++i)
    mmio16(y_base + 2 * i) = y_inp[(i / K_SIZE) * REFERENCE_K + i % K_SIZE];
  asm volatile("fence rw, rw" : : : "memory");

  hwpe_cg_enable();
  hwpe_soft_clear();
  while (hwpe_acquire_job() < 0) {
  }
  redmule_cfg(x_base, w_base, y_base, M_SIZE, N_SIZE, K_SIZE, (uint8_t)gemm_ops,
              (uint8_t)Float16, (uint8_t)Float16);
  if (get_hartid() == 0)
    printf("RedMulE tile 0: starting 4x64x4 GEMM\n");
  hwpe_trigger_job();
  if (hwpe_wait_for_completion()) {
    printf("RedMulE tile %u: completion timeout\n", get_hartid());
    return 1;
  }
  asm volatile("fence rw, rw" : : : "memory");
  hwpe_cg_disable();

  unsigned errors = 0;
  for (unsigned i = 0; i < M_SIZE * K_SIZE; ++i) {
    uint16_t computed = mmio16(y_base + 2 * i);
    uint16_t expected = z_oup[(i / K_SIZE) * REFERENCE_K + i % K_SIZE];
    unsigned diff =
        computed > expected ? computed - expected : expected - computed;
    if (diff > DIFF_TH) {
      if (errors == 0)
        printf("RedMulE tile %u: element %u got %04x expected %04x\n",
               get_hartid(), i, computed, expected);
      ++errors;
    }
  }
  if (errors || get_hartid() == 0)
    printf("RedMulE tile %u: 4x64x4, %u errors\n", get_hartid(), errors);
  return errors;
}
