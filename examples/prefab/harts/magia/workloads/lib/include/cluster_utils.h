/*
 * Copyright (C) 2023-2026 ETH Zurich, University of Bologna and Fondazione
 * Chips-IT
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
 * Bare-metal PULP Cluster Utilities for MAGIA.
 *
 * Two usage perspectives:
 *
 *   CV32 (main core) — orchestrator:
 *     cluster_boot(binary)         boot all PULP cores into the dispatcher
 *                                  loop (= pulp_init); polls PULP_READY.
 *     cluster_arm_done_event()     clear/enable the CV32 EU done event
 *     cluster_dispatch_task()      write NB_CORES_TO_WAIT, TASKBIN, START;
 *                                  returns once selected cores have ACK'd
 *     cluster_wait_done_polling()  spin on the CV32 EU done event
 *     cluster_done_pending()       non-blocking EU done-event check
 *     cluster_wait_done_eu()       WFE on PULP_DONE (EU bit 12)
 *     cluster_stop()               de-assert PULP CLK_EN
 *
 *   PULP cluster core — worker:
 *     cluster_core_id()            local index within the cluster (0..N-1)
 *     cluster_tile_id()            tile index of this core
 *     cluster_chunk_offset/size()  data partition helpers
 *
 * Hardware (obi_slave_ctrl_cluster.sv) memory map @ PULP_CTRL_BASE = 0x1740:
 *   see magia_tile_utils.h. EU bit 12 = PULP_DONE quorum.
 */

#ifndef CLUSTER_UTILS_H
#define CLUSTER_UTILS_H

#include "event_unit_utils.h"
#include "magia_pulp_utils.h"
#include "magia_tile_utils.h"
#include <stdint.h>

// =============================================================================
// CV32 (main core) — orchestrator API
// =============================================================================

static inline void cluster_boot(uint32_t binary_start) {
  pulp_init(binary_start);
}

static inline void cluster_dispatch_task(uint32_t task_addr,
                                         uint32_t core_mask) {
  pulp_run_task(task_addr, core_mask);
}

static inline void cluster_dispatch_task_with_params(uint32_t task_addr,
                                                     uint32_t params_ptr,
                                                     uint32_t core_mask) {
  pulp_run_task_with_params(task_addr, params_ptr, core_mask);
}

static inline void cluster_stop(void) { pulp_clk_dis(); }

static inline void cluster_wait_done_polling(void) {
  (void)eu_cluster_done_wait(EU_WAIT_MODE_POLLING);
}

static inline uint32_t cluster_done_pending(void) {
  return eu_check_events(EU_CLUSTER_DONE_MASK) != 0;
}

static inline void cluster_arm_done_event(void) { eu_cluster_done_init(); }

static inline void cluster_wait_done_eu(void) {
  (void)eu_cluster_done_wait(EU_WAIT_MODE_WFE);
}

// =============================================================================
// Cluster core (worker) — identity helpers
// =============================================================================

static inline uint32_t cluster_core_id(void) {
  uint32_t hartid;
  asm volatile("csrr %0, mhartid" : "=r"(hartid));
  return (hartid - PULP_HARTID_BASE) % PULP_CORE_COUNT;
}

static inline uint32_t cluster_tile_id(void) {
  uint32_t hartid;
  asm volatile("csrr %0, mhartid" : "=r"(hartid));
  return (hartid - PULP_HARTID_BASE) / PULP_CORE_COUNT;
}

// =============================================================================
// Cluster core (worker) — data partitioning helpers
// =============================================================================

static inline uint32_t cluster_chunk_offset(uint32_t total, uint32_t n_cores,
                                            uint32_t core_id) {
  return (total / n_cores) * core_id;
}

static inline uint32_t cluster_chunk_size(uint32_t total, uint32_t n_cores,
                                          uint32_t core_id) {
  uint32_t base = total / n_cores;
  return (core_id == n_cores - 1) ? (total - base * core_id) : base;
}

#endif /* CLUSTER_UTILS_H */
