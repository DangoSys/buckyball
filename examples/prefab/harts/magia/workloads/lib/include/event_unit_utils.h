/*
 * Copyright (C) 2024 ETH Zurich and University of Bologna
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
 * MAGIA Event Unit Utilities
 * Two modes: POLLING (non-blocking) and WFE (blocking with cv.elw sleep or
 * standard lw)
 */

#ifndef EVENT_UNIT_UTILS_H
#define EVENT_UNIT_UTILS_H

#include "magia_tile_utils.h"
#include <stdint.h>

//=============================================================================
// REGISTER DEFINITIONS AND CONSTANTS
//=============================================================================

#define EU_BASE EVENT_UNIT_BASE

// Control and status registers
#define EU_CORE_MASK (EU_BASE + 0x00)
#define EU_CORE_MASK_AND (EU_BASE + 0x04)
#define EU_CORE_MASK_OR (EU_BASE + 0x08)
#define EU_CORE_IRQ_MASK (EU_BASE + 0x0C)
#define EU_CORE_IRQ_MASK_AND (EU_BASE + 0x10)
#define EU_CORE_IRQ_MASK_OR (EU_BASE + 0x14)
#define EU_CORE_STATUS (EU_BASE + 0x18)
#define EU_CORE_BUFFER (EU_BASE + 0x1C)
#define EU_CORE_BUFFER_MASKED (EU_BASE + 0x20)
#define EU_CORE_BUFFER_IRQ_MASKED (EU_BASE + 0x24)
#define EU_CORE_BUFFER_CLEAR (EU_BASE + 0x28)

// Wait registers (blocking event load - cv.elw if pulp extensions are
// activated)
#define EU_CORE_EVENT_WAIT (EU_BASE + 0x38)
#define EU_CORE_EVENT_WAIT_CLEAR (EU_BASE + 0x3C)

// Hardware mutex registers (0x04 * mutex_id offset)
#define EU_CORE_HW_MUTEX (EU_BASE + 0x0C0) // R/W: HW mutex management

// Hardware barrier registers (0x20 * barr_id offset)
#define HW_BARR_TRIGGER_MASK (EU_BASE + 0x400) // R/W: Barrier trigger mask
#define HW_BARR_STATUS (EU_BASE + 0x404)       // R: Barrier status
#define HW_BARR_TARGET_MASK (EU_BASE + 0x40C)  // R/W: Barrier target mask
#define HW_BARR_TRIGGER (EU_BASE + 0x410)      // W: Manual barrier trigger
#define HW_BARR_TRIGGER_SELF (EU_BASE + 0x414) // R: Automatic trigger
#define HW_BARR_TRIGGER_WAIT (EU_BASE + 0x418) // R: Trigger + sleep
#define HW_BARR_TRIGGER_WAIT_CLEAR                                             \
  (EU_BASE + 0x41C) // R: Trigger + sleep + clear

// Software event trigger registers (0x04 * sw_event_id offset)
#define EU_CORE_TRIGG_SW_EVENT (EU_BASE + 0x600) // W: Generate SW event
#define EU_CORE_TRIGG_SW_EVENT_WAIT                                            \
  (EU_BASE + 0x640) // R: Generate event + sleep
#define EU_CORE_TRIGG_SW_EVENT_WAIT_CLEAR                                      \
  (EU_BASE + 0x680) // R: Generate event + sleep + clear

// SoC event FIFO register
#define EU_CORE_CURRENT_EVENT (EU_BASE + 0x700) // R: SoC event FIFO

// Event bit mapping
#define EU_DMA_EVT_0_BIT 2
#define EU_DMA_EVT_1_BIT 3
#define EU_TIMER_EVT_0_BIT 4
#define EU_TIMER_EVT_1_BIT 5

#define EU_REDMULE_UNUSED_BIT 8
#define EU_REDMULE_BUSY_BIT 9
#define EU_REDMULE_DONE_BIT 10
#define EU_REDMULE_EVT1_BIT 11

// RedMulE event masks
#define EU_REDMULE_DONE_MASK (1 << EU_REDMULE_DONE_BIT)
#define EU_REDMULE_BUSY_MASK (1 << EU_REDMULE_BUSY_BIT)
#define EU_REDMULE_ALL_MASK 0x0F00

// iDMA events (DMA events [3:2] + extended [31:26])
#define EU_IDMA_A2O_DONE_BIT 2
#define EU_IDMA_O2A_DONE_BIT 3
#define EU_IDMA_A2O_DONE_MASK (1 << EU_IDMA_A2O_DONE_BIT)
#define EU_IDMA_O2A_DONE_MASK (1 << EU_IDMA_O2A_DONE_BIT)
#define EU_IDMA_ALL_DONE_MASK (EU_IDMA_A2O_DONE_MASK | EU_IDMA_O2A_DONE_MASK)
#define EU_IDMA_A2O_ERROR_BIT 26
#define EU_IDMA_O2A_ERROR_BIT 27
#define EU_IDMA_A2O_START_BIT 28
#define EU_IDMA_O2A_START_BIT 29
#define EU_IDMA_A2O_BUSY_BIT 30
#define EU_IDMA_O2A_BUSY_BIT 31
#define EU_IDMA_A2O_ERROR_MASK (1 << EU_IDMA_A2O_ERROR_BIT)
#define EU_IDMA_O2A_ERROR_MASK (1 << EU_IDMA_O2A_ERROR_BIT)
#define EU_IDMA_A2O_START_MASK (1 << EU_IDMA_A2O_START_BIT)
#define EU_IDMA_O2A_START_MASK (1 << EU_IDMA_O2A_START_BIT)
#define EU_IDMA_A2O_BUSY_MASK (1 << EU_IDMA_A2O_BUSY_BIT)
#define EU_IDMA_O2A_BUSY_MASK (1 << EU_IDMA_O2A_BUSY_BIT)

// FSync events (cluster events [25:24])
#define EU_FSYNC_DONE_BIT 24
#define EU_FSYNC_ERROR_BIT 25
#define EU_FSYNC_DONE_MASK (1 << EU_FSYNC_DONE_BIT)
#define EU_FSYNC_ERROR_MASK (1 << EU_FSYNC_ERROR_BIT)
#define EU_FSYNC_ALL_MASK (EU_FSYNC_DONE_MASK | EU_FSYNC_ERROR_MASK)

// Spatz events (accelerator events [8] + cluster events [23])
#define EU_SPATZ_DONE_BIT 8
#define EU_SPATZ_START_BIT 23
#define EU_SPATZ_DONE_MASK (1 << EU_SPATZ_DONE_BIT)
#define EU_SPATZ_START_MASK (1 << EU_SPATZ_START_BIT)
#define EU_SPATZ_ALL_MASK (EU_SPATZ_DONE_MASK | EU_SPATZ_START_MASK)

// Bare-metal PULP cluster completion event generated by the tile CSR.
#define EU_CLUSTER_DONE_BIT 12
#define EU_CLUSTER_DONE_MASK (1 << EU_CLUSTER_DONE_BIT)
#define EU_CLUSTER_EVT_MASK EU_CLUSTER_DONE_MASK

// Wait modes
typedef enum { EU_WAIT_MODE_POLLING = 0, EU_WAIT_MODE_WFE = 1 } eu_wait_mode_t;

//=============================================================================
// LOW-LEVEL HAL (PULP-compatible evt_read32)
//=============================================================================

/* cv.elw on CV32E40P, p.elw on older PULP toolchains */
static inline unsigned int evt_read32(unsigned int base, unsigned int offset) {
  unsigned int value;
  unsigned int addr = base + offset;
#if defined(CV32E40P)
  __asm__ __volatile__("cv.elw %0, 0(%1)" : "=r"(value) : "r"(addr) : "memory");
#else
  // Standard lw for cores without PULP extensions (CV32E40X, etc.)
  __asm__ __volatile__("lw %0, 0(%1)" : "=r"(value) : "r"(addr) : "memory");
#endif
  return value;
}

//=============================================================================
// BASIC CONTROL FUNCTIONS
//=============================================================================

static inline void eu_init(void) {
  mmio32(EU_CORE_BUFFER_CLEAR) = 0xFFFFFFFF;
  mmio32(EU_CORE_MASK) = 0x00000000;
  mmio32(EU_CORE_IRQ_MASK) = 0x00000000;
}

static inline void eu_enable_events(uint32_t event_mask) {
  mmio32(EU_CORE_MASK_OR) = event_mask;
}

static inline void eu_disable_events(uint32_t event_mask) {
  mmio32(EU_CORE_MASK_AND) = event_mask;
}

static inline void eu_enable_irq(uint32_t irq_mask) {
  mmio32(EU_CORE_IRQ_MASK_OR) = irq_mask;
}

static inline void eu_disable_irq(uint32_t irq_mask) {
  mmio32(EU_CORE_IRQ_MASK_AND) = irq_mask;
}

static inline void eu_clear_events(uint32_t event_mask) {
  mmio32(EU_CORE_BUFFER_CLEAR) = event_mask;
}

//=============================================================================
// STATUS READ FUNCTIONS (non-blocking)
//=============================================================================

static inline uint32_t eu_get_events(void) { return mmio32(EU_CORE_BUFFER); }

static inline uint32_t eu_get_events_masked(void) {
  return mmio32(EU_CORE_BUFFER_MASKED);
}

static inline uint32_t eu_check_events(uint32_t event_mask) {
  return mmio32(EU_CORE_BUFFER_MASKED) & event_mask;
}

//=============================================================================
// WAIT FUNCTIONS (polling and blocking)
//=============================================================================

// POLLING mode: non-blocking busy-wait
static inline uint32_t eu_wait_events_polling(uint32_t event_mask,
                                              uint32_t timeout_cycles) {
  uint32_t cycles = 0;
  uint32_t detected_events;
  do {
    detected_events = eu_check_events(event_mask);
    if (detected_events) {
      eu_clear_events(detected_events);
      return detected_events;
    }
    wait_nop(10);
    cycles += 10;
  } while (timeout_cycles == 0 || cycles < timeout_cycles);
  return 0;
}

// WFE mode: blocking event load (cv.elw or lw depending on if pulp extensions
// are activated)
static inline uint32_t eu_wait_events_wfe(uint32_t event_mask) {
  eu_enable_events(event_mask);
  return evt_read32(EU_BASE, EU_CORE_EVENT_WAIT_CLEAR - EU_BASE);
}

// Generic wait with mode selection
static inline uint32_t eu_wait_events(uint32_t event_mask, eu_wait_mode_t mode,
                                      uint32_t timeout_cycles) {
  if (mode == EU_WAIT_MODE_WFE)
    return eu_wait_events_wfe(event_mask);
  else
    return eu_wait_events_polling(event_mask, timeout_cycles);
}

// PULP HAL compatible functions
static inline unsigned int eu_evt_wait(void) {
  return evt_read32(EU_BASE, EU_CORE_EVENT_WAIT - EU_BASE);
}

static inline unsigned int eu_evt_waitAndClr(void) {
  return evt_read32(EU_BASE, EU_CORE_EVENT_WAIT_CLEAR - EU_BASE);
}

static inline unsigned int eu_evt_maskWaitAndClr(unsigned int evtMask) {
  eu_enable_events(evtMask);
  unsigned int result = eu_evt_waitAndClr();
  eu_disable_events(evtMask);
  return result;
}

//=============================================================================
// PULP CLUSTER FUNCTIONS
//=============================================================================

static inline void eu_cluster_done_init(void) {
  /* Absolute write (not OR) — clears stale RedMulE/iDMA events that
     would otherwise wake the WFE before the real PULP_DONE. */
  mmio32(EU_CORE_MASK) = 0x00000000;      /* disable ALL events (absolute) */
  eu_clear_events(0xFFFFFFFF);            /* drop any stale latched events  */
  eu_enable_events(EU_CLUSTER_DONE_MASK); /* enable ONLY cluster-done (b12)  */
}

static inline uint32_t eu_cluster_done_wait(eu_wait_mode_t mode) {
  eu_enable_events(EU_CLUSTER_DONE_MASK);
  return eu_wait_events(EU_CLUSTER_DONE_MASK, mode, 1000000);
}

static inline void eu_pulp_init(void) { eu_cluster_done_init(); }

static inline uint32_t eu_pulp_wait(eu_wait_mode_t mode) {
  return eu_cluster_done_wait(mode);
}

//=============================================================================
// REDMULE FUNCTIONS
//=============================================================================

static inline void eu_redmule_init(void) {
  eu_clear_events(0xFFFFFFFF);
  eu_enable_events(EU_REDMULE_DONE_MASK);
}

static inline uint32_t eu_redmule_wait_completion(eu_wait_mode_t mode) {
  return eu_wait_events(EU_REDMULE_DONE_MASK, mode, 1000000);
}

static inline uint32_t eu_redmule_is_busy(void) {
  return eu_check_events(EU_REDMULE_BUSY_MASK);
}

static inline uint32_t eu_redmule_is_done(void) {
  return eu_check_events(EU_REDMULE_DONE_MASK);
}

//=============================================================================
// IDMA FUNCTIONS
//=============================================================================

static inline void eu_idma_init(void) {
  eu_clear_events(0xFFFFFFFF);
  eu_enable_events(EU_IDMA_ALL_DONE_MASK);
}

static inline uint32_t eu_idma_wait_completion(eu_wait_mode_t mode) {
  return eu_wait_events(EU_IDMA_ALL_DONE_MASK, mode, 1000000);
}

static inline uint32_t eu_idma_wait_direction_completion(uint32_t direction,
                                                         eu_wait_mode_t mode) {
  uint32_t wait_mask =
      direction ? EU_IDMA_O2A_DONE_MASK : EU_IDMA_A2O_DONE_MASK;
  return eu_wait_events(wait_mask, mode, 1000000);
}

static inline uint32_t eu_idma_wait_a2o_completion(eu_wait_mode_t mode) {
  return eu_wait_events(EU_IDMA_A2O_DONE_MASK, mode, 1000000);
}

static inline uint32_t eu_idma_wait_o2a_completion(eu_wait_mode_t mode) {
  return eu_wait_events(EU_IDMA_O2A_DONE_MASK, mode, 1000000);
}

static inline uint32_t eu_idma_is_done(void) {
  return eu_check_events(EU_IDMA_ALL_DONE_MASK);
}

static inline uint32_t eu_idma_a2o_is_done(void) {
  return eu_check_events(EU_IDMA_A2O_DONE_MASK);
}

static inline uint32_t eu_idma_o2a_is_done(void) {
  return eu_check_events(EU_IDMA_O2A_DONE_MASK);
}

static inline uint32_t eu_idma_is_busy(void) {
  uint32_t events = eu_get_events();
  return events & (EU_IDMA_A2O_BUSY_MASK | EU_IDMA_O2A_BUSY_MASK);
}

static inline uint32_t eu_idma_has_error(void) {
  uint32_t events = eu_get_events();
  return events & (EU_IDMA_A2O_ERROR_MASK | EU_IDMA_O2A_ERROR_MASK);
}

//=============================================================================
// FSYNC FUNCTIONS
//=============================================================================

static inline void eu_fsync_init(void) {
  eu_clear_events(0xFFFFFFFF);
  eu_enable_events(EU_FSYNC_ALL_MASK);
}

static inline uint32_t eu_fsync_wait_completion(eu_wait_mode_t mode) {
  return eu_wait_events(EU_FSYNC_DONE_MASK, mode, 1000000);
}

static inline uint32_t eu_fsync_is_done(void) {
  return eu_check_events(EU_FSYNC_DONE_MASK);
}

static inline uint32_t eu_fsync_has_error(void) {
  return eu_check_events(EU_FSYNC_ERROR_MASK);
}

//=============================================================================
// SPATZ FUNCTIONS
//=============================================================================

static inline void eu_spatz_init(void) {
  eu_clear_events(0xFFFFFFFF);
  eu_enable_events(EU_SPATZ_DONE_MASK);
}

static inline uint32_t eu_spatz_is_done(void) {
  return eu_check_events(EU_SPATZ_DONE_MASK);
}

static inline void eu_wait_spatz_wfe(uint32_t event_mask) {
  while (!eu_check_events(event_mask)) {
    eu_evt_wait();
  }
  eu_clear_events(event_mask);
}

static inline void eu_wait_spatz_polling(uint32_t event_mask) {
  while (!eu_check_events(event_mask)) {
    wait_nop(10);
  }
  eu_clear_events(event_mask);
}

//=============================================================================
// MULTI-ACCELERATOR FUNCTIONS
//=============================================================================

static inline void eu_multi_init(uint32_t redmule_en, uint32_t idma_a2o_en,
                                 uint32_t idma_o2a_en, uint32_t fsync_en) {
  eu_clear_events(0xFFFFFFFF);
  uint32_t event_mask = 0;

  if (redmule_en) {
    event_mask |= EU_REDMULE_ALL_MASK;
  }
  if (idma_a2o_en) {
    event_mask |= EU_IDMA_A2O_DONE_MASK;
  }
  if (idma_o2a_en) {
    event_mask |= EU_IDMA_O2A_DONE_MASK;
  }
  if (fsync_en) {
    event_mask |= EU_FSYNC_ALL_MASK;
  }

  if (event_mask)
    eu_enable_events(event_mask);
}

static inline uint32_t eu_multi_wait_all(uint32_t wait_redmule,
                                         uint32_t wait_idma_a2o,
                                         uint32_t wait_idma_o2a,
                                         uint32_t wait_fsync,
                                         eu_wait_mode_t mode) {
  uint32_t required_mask = 0;
  if (wait_redmule)
    required_mask |= EU_REDMULE_DONE_MASK;
  if (wait_idma_a2o)
    required_mask |= EU_IDMA_A2O_DONE_MASK;
  if (wait_idma_o2a)
    required_mask |= EU_IDMA_O2A_DONE_MASK;
  if (wait_fsync)
    required_mask |= EU_FSYNC_DONE_MASK;

  eu_enable_events(required_mask);

  if (mode == EU_WAIT_MODE_WFE) {
    uint32_t accumulated = 0;
    while ((accumulated & required_mask) != required_mask) {
      uint32_t new_events =
          evt_read32(EU_BASE, EU_CORE_EVENT_WAIT_CLEAR - EU_BASE);
      accumulated |= new_events;
    }
    return accumulated;
  } else {
    uint32_t timeout = 1000000;
    uint32_t cycles = 0;
    uint32_t accumulated = 0;
    while (cycles < timeout && (accumulated & required_mask) != required_mask) {
      accumulated |= eu_check_events(required_mask);
      wait_nop(10);
      cycles += 10;
    }
    return accumulated;
  }
}

#endif /* EVENT_UNIT_UTILS_H */
