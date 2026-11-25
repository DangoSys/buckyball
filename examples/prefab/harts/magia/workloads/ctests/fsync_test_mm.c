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
 *          Based on fsync_test.c by Victor Isachi
 *
 */

#include "fsync_mm_utils.h"
#include "magia_tile_utils.h"
#include "magia_utils.h"

static volatile uint32_t arrived[NUM_HARTS];

int main(void) {
  uint32_t hart = get_hartid();
  unsigned errors = 0;
  for (unsigned round = 1; round <= 3; ++round) {
    for (unsigned level = 1; level <= 31 - __builtin_clz(NUM_HARTS); ++level)
      fsync_mm(0, (1u << level) - 1);

    arrived[hart] = round;
    asm volatile("fence rw, rw" : : : "memory");
    fsync_mm(0, NUM_HARTS - 1);
    for (unsigned tile = 0; tile < NUM_HARTS; ++tile)
      errors += arrived[tile] != round;
    fsync_mm(0, NUM_HARTS - 1);
  }
  if (hart == 0)
    printf("FractalSync: %u tiles, 3 rounds, %u errors\n", NUM_HARTS, errors);
  return errors;
}
