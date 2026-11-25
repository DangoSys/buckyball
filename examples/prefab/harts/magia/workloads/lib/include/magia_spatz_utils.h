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
 * Authors: Luca Balboni <luca.balboni10@studio.unibo.it>
 *
 *  Spatz Utility Functions
 */
#ifndef MAGIA_SPATZ_UTILS_H
#define MAGIA_SPATZ_UTILS_H

#include "magia_tile_utils.h"
#include <stdint.h>

#define SPATZ_CLK_EN (SPATZ_CTRL_BASE + 0x00)
#define SPATZ_READY (SPATZ_CTRL_BASE + 0x04)
#define SPATZ_START (SPATZ_CTRL_BASE + 0x08)
#define SPATZ_TASKBIN (SPATZ_CTRL_BASE + 0x0C)
#define SPATZ_DATA (SPATZ_CTRL_BASE + 0x10)
#define SPATZ_RETURN (SPATZ_CTRL_BASE + 0x14)
#define SPATZ_DONE (SPATZ_CTRL_BASE + 0x18)

#define mmio32(x) (*(volatile uint32_t *)(x))

static inline void spatz_clk_en(void) { mmio32(SPATZ_CLK_EN) = 1; }

static inline void spatz_clk_dis(void) { mmio32(SPATZ_CLK_EN) = 0; }

static inline void spatz_set_func(uint32_t addr) {
  mmio32(SPATZ_TASKBIN) = addr;
}

static inline void spatz_trigger_en_irq(void) { mmio32(SPATZ_START) = 1; }

static inline void spatz_trigger_dis_irq(void) { mmio32(SPATZ_START) = 0; }

static inline void spatz_done(void) { mmio32(SPATZ_DONE) = 1; }

static inline uint32_t spatz_get_exit_code(void) {
  return mmio32(SPATZ_RETURN);
}

static inline void spatz_run_task(uint32_t spatz_task_addr) {
  spatz_set_func(spatz_task_addr);
  spatz_trigger_en_irq();
  while (mmio32(SPATZ_START) != 0)
    ;
}

static inline void spatz_pass_params(uint32_t params_ptr) {
  mmio32(SPATZ_DATA) = params_ptr;
}

static inline void spatz_run_task_with_params(uint32_t spatz_task_addr,
                                              uint32_t params_ptr) {
  spatz_pass_params(params_ptr);
  spatz_run_task(spatz_task_addr);
}

static inline void spatz_init(uint32_t spatz_start_addr) {
  spatz_set_func(spatz_start_addr);
  spatz_clk_en();
  while (mmio32(SPATZ_READY) == 0) {
    printf("Waiting for Spatz to be ready...\n");
  };
}

#endif
