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
 * Bare-metal PULP Cluster Utility Functions (CV32 control side).
 *
 * Dynamic dispatch model (CV32 -> PULP cluster):
 *   1. pulp_init(binary)        boot all cores into the PULP dispatcher loop;
 *                               polls PULP_READY until every core is armed.
 *   2. pulp_run_task(task,mask) dispatch the task function to a subset of
 *                               cores: writes NB_CORES_TO_WAIT =
 * popcount(mask), TASKBIN = task, then PULP_START = mask which fires per-core
 * MEI pulses. CV32 polls PULP_START until it self-clears (all ACKs).
 *   3. cluster_wait_done_eu()/_polling() wait for DONE quorum (PULP_DONE).
 *
 *   pulp_run_task_with_params() also writes PULP_DATA so the task receives a
 *   context pointer as its first argument (a0 in the trap handler).
 *
 * Register map: see magia_tile_utils.h (PULP_CTRL_BASE).
 */
#ifndef MAGIA_PULP_UTILS_H
#define MAGIA_PULP_UTILS_H

#include "magia_tile_utils.h"
#include <stdint.h>

/* ---- Low-level register helpers ---------------------------------------- */

static inline void pulp_clk_en(void) { mmio32(PULP_CLK_EN) = 1; }
static inline void pulp_clk_dis(void) { mmio32(PULP_CLK_EN) = 0; }

static inline void pulp_set_binary(uint32_t addr) {
  mmio32(PULP_BINARY) = addr;
}

static inline void pulp_set_nb_cores_to_wait(uint32_t nb_cores) {
  mmio32(PULP_NB_CORES_TO_WAIT) = nb_cores;
}

static inline void pulp_set_func(uint32_t task_addr) {
  mmio32(PULP_TASKBIN) = task_addr;
}

static inline void pulp_pass_params(uint32_t params_ptr) {
  mmio32(PULP_DATA) = params_ptr;
}

static inline uint32_t _pulp_popcount(uint32_t x) {
  x = x - ((x >> 1) & 0x55555555u);
  x = (x & 0x33333333u) + ((x >> 2) & 0x33333333u);
  x = (x + (x >> 4)) & 0x0f0f0f0fu;
  return (x * 0x01010101u) >> 24;
}

/* ---- High-level dispatch API ------------------------------------------- */

/**
 * @brief Boot the PULP cluster: write the binary entry point, enable all
 *        cores (CLK_EN broadcast), then wait until every core has armed its
 *        dispatcher (PULP_READY == 1).
 */
static inline void pulp_init(uint32_t binary_start) {
  pulp_set_binary(binary_start);
  pulp_clk_en();
  while ((mmio32(PULP_READY) & 1u) == 0u) {
  }
}

/**
 * @brief Dispatch @p task_addr to the cores selected by @p core_mask.
 *        Returns once every selected core has ACK'd the start (write-0 to
 *        PULP_START), i.e. once the cores have entered the task function.
 *        Use cluster_wait_done_eu()/cluster_wait_done_polling() for completion.
 */
static inline void pulp_run_task(uint32_t task_addr, uint32_t core_mask) {
  pulp_set_nb_cores_to_wait(_pulp_popcount(core_mask));
  pulp_set_func(task_addr);
  mmio32(PULP_START) = core_mask;
  while (mmio32(PULP_START) != 0u) {
  }
}

/**
 * @brief Dispatch a task with a context pointer passed as first argument.
 */
static inline void pulp_run_task_with_params(uint32_t task_addr,
                                             uint32_t params_ptr,
                                             uint32_t core_mask) {
  pulp_pass_params(params_ptr);
  pulp_run_task(task_addr, core_mask);
}

#endif /* MAGIA_PULP_UTILS_H */
