/*
 * Copyright (C) 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT
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
 * Authors: Carlotta Chiarini
 *
 *  Collective Utility Functions
 */
#ifndef MAGIA_COLL_UTILS_H
#define MAGIA_COLL_UTILS_H

#include "magia_utils.h"
#include <math.h>
#include <stdint.h>

#define COLLECTIVE_MASK_OFFSET (COLL_CTRL_BASE + 0x00)
#define COLLECTIVE_OP_OFFSET (COLL_CTRL_BASE + 0x04)

#define COLLECTIVE_ADDR_OFFSET (0xB0000000)

#define MASK_OFFSET 20

#define MULTICAST 1
#define LSBAND 2
#define FP_ADD 3
#define FP_MUL 4
#define FP_MIN 5
#define FP_MAX 6
#define INT_ADD 7
#define INT_MUL 8
#define INT_MINS 9
#define INT_MINU 10
#define INT_MAXS 11
#define INT_MAXU 12

#define ALL 1
#define COLUMN 2
#define ROW 3

#define mmio32(x) (*(volatile uint32_t *)(x))

static inline void set_collective_mask(uint32_t collective_mask) {
  mmio32(COLLECTIVE_MASK_OFFSET) = collective_mask;
}

static inline void set_collective_op(uint32_t collective_op) {
  mmio32(COLLECTIVE_OP_OFFSET) = collective_op;
}

static inline uint32_t gen_collective_mask(uint32_t geometry) {
  uint8_t shift = log2(NUM_HARTS) / 2;
  uint8_t mask = sqrt(NUM_HARTS) - 1;
  if (geometry == ALL)
    return (mask << (MASK_OFFSET)) | (mask << (MASK_OFFSET + shift));
  else if (geometry == COLUMN)
    return (mask << (MASK_OFFSET + shift));
  else if (geometry == ROW)
    return (mask << (MASK_OFFSET));
}

static inline void compiler_barrier() { asm volatile("" ::: "memory"); }

static inline void memory_fence() { asm volatile("fence" ::: "memory"); }

#endif
