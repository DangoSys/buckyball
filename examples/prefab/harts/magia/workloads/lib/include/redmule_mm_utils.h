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
 *
 * RedMulE MM Utilities
 *
 * This header contains MM-related definitions and functions for RedMulE control
 * using MMIO-based register access.
 */

#ifndef REDMULE_MM_UTILS_H
#define REDMULE_MM_UTILS_H

#include "magia_tile_utils.h"
#include <stdint.h>

/* OBI2HWPE Protocol Stamps */
#define REDMULE_ADDR_BASE REDMULE_BASE

#define HWPE_WRITE(value, offset)                                              \
  *(volatile int *)(REDMULE_ADDR_BASE + offset) = value
#define HWPE_READ(offset) *(volatile int *)(REDMULE_ADDR_BASE + offset)

/* Register offsets (RedMulE hwpe-ctrl) */
#define REDMULE_REG_OFFS 0x00
#define REDMULE_TRIGGER 0x00
#define REDMULE_ACQUIRE 0x04
#define REDMULE_EVT_ENABLE 0x08
#define REDMULE_STATUS 0x0C
#define REDMULE_RUNNING_JOB 0x10
#define REDMULE_SOFT_CLEAR 0x14

/* RedMulE configuration registers */
#define REDMULE_REG_MCNFIG0 0x20
#define REDMULE_REG_MCNFIG1 0x24
#define REDMULE_REG_MCNFIG2 0x28
#define REDMULE_REG_MARITH0 0x2c
#define REDMULE_REG_MARITH1 0x30
#define REDMULE_REG_MARITH2 0x34
#define REDMULE_REG_MOPCNT 0x38

/* Operations and formats */
#define Float16 0x1
#define Float16Alt 0x2
#define Float8 0x3
#define Float8Alt 0x4

#define gemm_ops 0x1
#define matmul_ops 0x0
#define addmax 0x2
#define addmin 0x3
#define mulmax 0x4
#define mulmin 0x5
#define maxmin 0x6
#define minmax 0x7

/* HWPE Register Access Functions */
static inline void redmule_x_add_set(unsigned int value) {
  HWPE_WRITE(value, REDMULE_REG_OFFS + REDMULE_REG_MARITH0);
}

static inline void redmule_w_add_set(unsigned int value) {
  HWPE_WRITE(value, REDMULE_REG_OFFS + REDMULE_REG_MARITH1);
}

static inline void redmule_z_add_set(unsigned int value) {
  HWPE_WRITE(value, REDMULE_REG_OFFS + REDMULE_REG_MARITH2);
}

static inline void redmule_mcfg_set(uint32_t mcfg0, uint32_t mcfg1,
                                    uint32_t mcfg2) {
  HWPE_WRITE(mcfg0, REDMULE_REG_OFFS + REDMULE_REG_MCNFIG0);
  HWPE_WRITE(mcfg1, REDMULE_REG_OFFS + REDMULE_REG_MCNFIG1);
  HWPE_WRITE(mcfg2, REDMULE_REG_OFFS + REDMULE_REG_MCNFIG2);
}

static inline void hwpe_trigger_job() { HWPE_WRITE(0, REDMULE_TRIGGER); }

static inline int hwpe_acquire_job() {
  int result = HWPE_READ(REDMULE_ACQUIRE);
  return result;
}

static inline unsigned int hwpe_get_status() {
  unsigned int result = HWPE_READ(REDMULE_STATUS);
  return result;
}

static inline void hwpe_soft_clear() { HWPE_WRITE(0, REDMULE_SOFT_CLEAR); }

static inline void hwpe_cg_enable() { return; }

static inline void hwpe_cg_disable() { return; }

static inline int hwpe_wait_for_completion(void) {
  for (unsigned polls = 0; polls < 100000; ++polls) {
    if (hwpe_get_status() == 0)
      return 0;
  }
  return 1;
}

/* RedMulE Configuration Function */
static inline void redmule_cfg(unsigned int x, unsigned int w, unsigned int z,
                               uint16_t m_size, uint16_t n_size,
                               uint16_t k_size, uint8_t gemm_op,
                               uint8_t gemm_fmt_in, uint8_t gemm_fmt_out) {

  uint32_t mcfg0 = (k_size << 16) | (m_size << 0);
  uint32_t mcfg1 = (gemm_fmt_out << 25) | (gemm_fmt_in << 23) |
                   (gemm_op << 20) | (n_size << 0);
  uint32_t mcfg2 = 0;

  redmule_x_add_set((unsigned int)x);
  redmule_w_add_set((unsigned int)w);
  redmule_z_add_set((unsigned int)z);
  redmule_mcfg_set(mcfg0, mcfg1, mcfg2);
}

#endif /* REDMULE_MM_UTILS_H */
