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
 *         Based on idma_utils.h by Victor Isachi
 *
 * MAGIA iDMA Memory-Mapped I/O Utils
 */

#ifndef IDMA_MM_UTILS_H
#define IDMA_MM_UTILS_H

#include "magia_tile_utils.h"
#include <stdint.h>

//=============================================================================
// Register Definitions and Constants
//=============================================================================

// iDMA Memory-Mapped Register Base Addresses
#define IDMA_MM_DIRECTION_OFFSET (0x200)
#define IDMA_MM_BASE_AXI2OBI (IDMA_BASE) // direction=0, L2 to L1
#define IDMA_MM_BASE_OBI2AXI                                                   \
  (IDMA_BASE + IDMA_MM_DIRECTION_OFFSET) // direction=1, L1 to L2

#define IDMA_CONF_OFFSET (0x00)
#define IDMA_STATUS_OFFSET (0x04)
#define IDMA_NEXT_ID_OFFSET (0x44)
#define IDMA_DONE_ID_OFFSET (0x84)
#define IDMA_MULTICAST_MASK_OFFSET (0xc4)
#define IDMA_COLLECTIVE_SEL_OFFSET (0xc8)
#define IDMA_DST_ADDR_LOW_OFFSET (0xD0)
#define IDMA_SRC_ADDR_LOW_OFFSET (0xD8)
#define IDMA_LENGTH_LOW_OFFSET (0xE0)
#define IDMA_DST_STRIDE_2_LOW_OFFSET (0xE8)
#define IDMA_SRC_STRIDE_2_LOW_OFFSET (0xF0)
#define IDMA_REPS_2_LOW_OFFSET (0xF8)
#define IDMA_DST_STRIDE_3_LOW_OFFSET (0x100)
#define IDMA_SRC_STRIDE_3_LOW_OFFSET (0x108)
#define IDMA_REPS_3_LOW_OFFSET (0x110)

// Register Addresses - now direction-aware
#define IDMA_CONF_ADDR(is_l1_to_l2)                                            \
  ((is_l1_to_l2) ? (IDMA_MM_BASE_OBI2AXI + IDMA_CONF_OFFSET)                   \
                 : (IDMA_MM_BASE_AXI2OBI + IDMA_CONF_OFFSET))
#define IDMA_STATUS_ADDR(is_l1_to_l2, id)                                      \
  ((is_l1_to_l2) ? (IDMA_MM_BASE_OBI2AXI + IDMA_STATUS_OFFSET + ((id) * 4))    \
                 : (IDMA_MM_BASE_AXI2OBI + IDMA_STATUS_OFFSET + ((id) * 4)))
#define IDMA_NEXT_ID_ADDR(is_l1_to_l2, id)                                     \
  ((is_l1_to_l2) ? (IDMA_MM_BASE_OBI2AXI + IDMA_NEXT_ID_OFFSET + ((id) * 4))   \
                 : (IDMA_MM_BASE_AXI2OBI + IDMA_NEXT_ID_OFFSET + ((id) * 4)))
#define IDMA_DONE_ID_ADDR(is_l1_to_l2, id)                                     \
  ((is_l1_to_l2) ? (IDMA_MM_BASE_OBI2AXI + IDMA_DONE_ID_OFFSET + ((id) * 4))   \
                 : (IDMA_MM_BASE_AXI2OBI + IDMA_DONE_ID_OFFSET + ((id) * 4)))
#define IDMA_DST_ADDR_LOW_ADDR(is_l1_to_l2)                                    \
  ((is_l1_to_l2) ? (IDMA_MM_BASE_OBI2AXI + IDMA_DST_ADDR_LOW_OFFSET)           \
                 : (IDMA_MM_BASE_AXI2OBI + IDMA_DST_ADDR_LOW_OFFSET))
#define IDMA_SRC_ADDR_LOW_ADDR(is_l1_to_l2)                                    \
  ((is_l1_to_l2) ? (IDMA_MM_BASE_OBI2AXI + IDMA_SRC_ADDR_LOW_OFFSET)           \
                 : (IDMA_MM_BASE_AXI2OBI + IDMA_SRC_ADDR_LOW_OFFSET))
#define IDMA_LENGTH_LOW_ADDR(is_l1_to_l2)                                      \
  ((is_l1_to_l2) ? (IDMA_MM_BASE_OBI2AXI + IDMA_LENGTH_LOW_OFFSET)             \
                 : (IDMA_MM_BASE_AXI2OBI + IDMA_LENGTH_LOW_OFFSET))
#define IDMA_DST_STRIDE_2_LOW_ADDR(is_l1_to_l2)                                \
  ((is_l1_to_l2) ? (IDMA_MM_BASE_OBI2AXI + IDMA_DST_STRIDE_2_LOW_OFFSET)       \
                 : (IDMA_MM_BASE_AXI2OBI + IDMA_DST_STRIDE_2_LOW_OFFSET))
#define IDMA_SRC_STRIDE_2_LOW_ADDR(is_l1_to_l2)                                \
  ((is_l1_to_l2) ? (IDMA_MM_BASE_OBI2AXI + IDMA_SRC_STRIDE_2_LOW_OFFSET)       \
                 : (IDMA_MM_BASE_AXI2OBI + IDMA_SRC_STRIDE_2_LOW_OFFSET))
#define IDMA_REPS_2_LOW_ADDR(is_l1_to_l2)                                      \
  ((is_l1_to_l2) ? (IDMA_MM_BASE_OBI2AXI + IDMA_REPS_2_LOW_OFFSET)             \
                 : (IDMA_MM_BASE_AXI2OBI + IDMA_REPS_2_LOW_OFFSET))
#define IDMA_DST_STRIDE_3_LOW_ADDR(is_l1_to_l2)                                \
  ((is_l1_to_l2) ? (IDMA_MM_BASE_OBI2AXI + IDMA_DST_STRIDE_3_LOW_OFFSET)       \
                 : (IDMA_MM_BASE_AXI2OBI + IDMA_DST_STRIDE_3_LOW_OFFSET))
#define IDMA_SRC_STRIDE_3_LOW_ADDR(is_l1_to_l2)                                \
  ((is_l1_to_l2) ? (IDMA_MM_BASE_OBI2AXI + IDMA_SRC_STRIDE_3_LOW_OFFSET)       \
                 : (IDMA_MM_BASE_AXI2OBI + IDMA_SRC_STRIDE_3_LOW_OFFSET))
#define IDMA_REPS_3_LOW_ADDR(is_l1_to_l2)                                      \
  ((is_l1_to_l2) ? (IDMA_MM_BASE_OBI2AXI + IDMA_REPS_3_LOW_OFFSET)             \
                 : (IDMA_MM_BASE_AXI2OBI + IDMA_REPS_3_LOW_OFFSET))
#define IDMA_MULTICAST_MASK_ADDR(is_l1_to_l2)                                  \
  ((is_l1_to_l2) ? (IDMA_MM_BASE_OBI2AXI + IDMA_MULTICAST_MASK_OFFSET)         \
                 : (IDMA_MM_BASE_AXI2OBI + IDMA_MULTICAST_MASK_OFFSET))
#define IDMA_COLLECTIVE_SEL_ADDR(is_l1_to_l2)                                  \
  ((is_l1_to_l2) ? (IDMA_MM_BASE_OBI2AXI + IDMA_COLLECTIVE_SEL_OFFSET)         \
                 : (IDMA_MM_BASE_AXI2OBI + IDMA_COLLECTIVE_SEL_OFFSET))

// Configuration Register Bit Fields
#define IDMA_CONF_DECOUPLE_AW_BIT (0)
#define IDMA_CONF_DECOUPLE_RW_BIT (1)
#define IDMA_CONF_SRC_REDUCE_LEN_BIT (2)
#define IDMA_CONF_DST_REDUCE_LEN_BIT (3)
#define IDMA_CONF_SRC_MAX_LLEN_MASK (0x70) // bits 6:4
#define IDMA_CONF_SRC_MAX_LLEN_SHIFT (4)
#define IDMA_CONF_DST_MAX_LLEN_MASK (0x380) // bits 9:7
#define IDMA_CONF_DST_MAX_LLEN_SHIFT (7)
#define IDMA_CONF_ENABLE_ND_MASK (0xC00) // bits 11:10
#define IDMA_CONF_ENABLE_ND_SHIFT (10)
#define IDMA_CONF_SRC_PROTOCOL (12)
#define IDMA_CONF_DST_PROTOCOL (15)

// Status Register Bit Fields
#define IDMA_STATUS_BUSY_MASK (0x3FF) // bits 9:0

// Transfer Direction Constants
#define IDMA_DIR_L2_TO_L1 (0) // AXI2OBI direction
#define IDMA_DIR_L1_TO_L2 (1) // OBI2AXI direction

// Direction aliases
#define IDMA_EXT2LOC 0 // L2 to L1 (AXI2OBI)
#define IDMA_LOC2EXT 1 // L1 to L2 (OBI2AXI)

// Transfer dimensions
#define IDMA_1D 0
#define IDMA_2D 1
#define IDMA_3D 2

// Protocol definitions
typedef enum {
  IDMA_PROT_AXI = 0, // AXI protocol: L2 memory
  IDMA_PROT_OBI = 1  // OBI protocol: L1 memory
} idma_prot_t;

typedef unsigned int dma_ext_t;

// Configuration macros
#define IDMA_DEFAULT_CONFIG 0x0

//=============================================================================
// Low-Level Register Access Functions
//=============================================================================

static inline void idma_mm_conf_dir(uint32_t is_l1_to_l2, uint32_t decouple_aw,
                                    uint32_t decouple_rw,
                                    uint32_t src_reduce_len,
                                    uint32_t dst_reduce_len,
                                    uint32_t src_max_llen,
                                    uint32_t dst_max_llen, uint32_t enable_nd) {
  uint32_t conf_val = 0;

  if (decouple_aw)
    conf_val |= (1 << IDMA_CONF_DECOUPLE_AW_BIT);
  if (decouple_rw)
    conf_val |= (1 << IDMA_CONF_DECOUPLE_RW_BIT);
  if (src_reduce_len)
    conf_val |= (1 << IDMA_CONF_SRC_REDUCE_LEN_BIT);
  if (dst_reduce_len)
    conf_val |= (1 << IDMA_CONF_DST_REDUCE_LEN_BIT);

  conf_val |= ((src_max_llen & 0x7) << IDMA_CONF_SRC_MAX_LLEN_SHIFT);
  conf_val |= ((dst_max_llen & 0x7) << IDMA_CONF_DST_MAX_LLEN_SHIFT);
  conf_val |= ((enable_nd & 0x3) << IDMA_CONF_ENABLE_ND_SHIFT);
  conf_val |= ((enable_nd & 0x3) << IDMA_CONF_ENABLE_ND_SHIFT);

  if (is_l1_to_l2)
    conf_val |= (1 << IDMA_CONF_SRC_PROTOCOL);
  else
    conf_val |= (1 << IDMA_CONF_DST_PROTOCOL);

  mmio32(IDMA_CONF_ADDR(is_l1_to_l2)) = conf_val;
}

static inline void idma_mm_conf_default_dir(uint32_t is_l1_to_l2) {
  idma_mm_conf_dir(is_l1_to_l2, 0, 0, 0, 0, 0, 0, 3);
}

static inline uint32_t idma_mm_is_busy_dir(uint32_t is_l1_to_l2,
                                           uint32_t stream_id) {
  if (stream_id >= 16)
    return 0;
  uint32_t status = mmio32(IDMA_STATUS_ADDR(is_l1_to_l2, stream_id));
  return (status & IDMA_STATUS_BUSY_MASK) ? 1 : 0;
}

static inline uint32_t idma_mm_start_transfer_dir(uint32_t is_l1_to_l2,
                                                  uint32_t stream_id) {
  if (stream_id >= 16)
    return 0;
  uint32_t transfer_id = mmio32(IDMA_NEXT_ID_ADDR(is_l1_to_l2, stream_id));
  return transfer_id;
}

static inline void idma_mm_set_multicast_mask(uint32_t is_l1_to_l2,
                                              uint32_t mask) {
  mmio32(IDMA_MULTICAST_MASK_ADDR(is_l1_to_l2)) = mask;
}

static inline void idma_mm_set_collective_sel(uint32_t is_l1_to_l2,
                                              uint32_t collective_op) {
  mmio32(IDMA_COLLECTIVE_SEL_ADDR(is_l1_to_l2)) = collective_op;
}

static inline uint32_t idma_mm_get_done_id_dir(uint32_t is_l1_to_l2,
                                               uint32_t stream_id) {
  if (stream_id >= 16)
    return 0;
  return mmio32(IDMA_DONE_ID_ADDR(is_l1_to_l2, stream_id));
}

static inline void idma_mm_set_addr_len_dir(uint32_t is_l1_to_l2,
                                            uint32_t dst_addr,
                                            uint32_t src_addr,
                                            uint32_t length) {
  mmio32(IDMA_DST_ADDR_LOW_ADDR(is_l1_to_l2)) = dst_addr;
  mmio32(IDMA_SRC_ADDR_LOW_ADDR(is_l1_to_l2)) = src_addr;
  mmio32(IDMA_LENGTH_LOW_ADDR(is_l1_to_l2)) = length;
}

static inline void idma_mm_set_2d_params_dir(uint32_t is_l1_to_l2,
                                             uint32_t dst_stride_2,
                                             uint32_t src_stride_2,
                                             uint32_t reps_2) {
  mmio32(IDMA_DST_STRIDE_2_LOW_ADDR(is_l1_to_l2)) = dst_stride_2;
  mmio32(IDMA_SRC_STRIDE_2_LOW_ADDR(is_l1_to_l2)) = src_stride_2;
  mmio32(IDMA_REPS_2_LOW_ADDR(is_l1_to_l2)) = reps_2;
}

static inline void idma_mm_set_3d_params_dir(uint32_t is_l1_to_l2,
                                             uint32_t dst_stride_3,
                                             uint32_t src_stride_3,
                                             uint32_t reps_3) {
  mmio32(IDMA_DST_STRIDE_3_LOW_ADDR(is_l1_to_l2)) = dst_stride_3;
  mmio32(IDMA_SRC_STRIDE_3_LOW_ADDR(is_l1_to_l2)) = src_stride_3;
  mmio32(IDMA_REPS_3_LOW_ADDR(is_l1_to_l2)) = reps_3;
}

static inline uint32_t idma_mm_wait_for_completion(uint32_t direction,
                                                   uint32_t transfer_id) {
  if (transfer_id == 0)
    return 0;

  uint32_t is_l1_to_l2 = (direction == IDMA_DIR_L1_TO_L2) ? 1 : 0;
  uint32_t stream_id = 0;
  uint32_t timeout = 1000000;

  while (timeout-- > 0) {
    uint32_t is_busy = idma_mm_is_busy_dir(is_l1_to_l2, stream_id);

    if (!is_busy) {
      uint32_t done_id = idma_mm_get_done_id_dir(is_l1_to_l2, stream_id);
      if (done_id == transfer_id) {
        return 1;
      }
    }

    wait_nop(10);
  }

  return 0;
}

//=============================================================================
// High-Level DMA API - 1D Transfers
//=============================================================================

// Forward declarations
static inline int idma_L1ToL2(unsigned int src, unsigned int dst,
                              unsigned short size);
static inline int idma_L2ToL1(unsigned int src, unsigned int dst,
                              unsigned short size);
static inline int idma_L1ToL1(unsigned int src, unsigned int dst,
                              unsigned short size);
static inline int idma_L1ToL2_2d(unsigned int src, unsigned int dst,
                                 unsigned short size, unsigned int src_stride,
                                 unsigned int dst_stride,
                                 unsigned int num_reps);
static inline int idma_L2ToL1_2d(unsigned int src, unsigned int dst,
                                 unsigned short size, unsigned int src_stride,
                                 unsigned int dst_stride,
                                 unsigned int num_reps);
static inline int idma_L1ToL1_2d(unsigned int src, unsigned int dst,
                                 unsigned short size, unsigned int src_stride,
                                 unsigned int dst_stride,
                                 unsigned int num_reps);

static inline int dma_memcpy(dma_ext_t ext, unsigned int loc,
                             unsigned short size, int ext2loc) {
  if (ext2loc)
    return idma_L2ToL1(ext, loc, size);
  else
    return idma_L1ToL2(loc, ext, size);
}

static inline int dma_l1ToExt(dma_ext_t ext, unsigned int loc,
                              unsigned short size) {
  return idma_L1ToL2(loc, ext, size);
}

static inline int dma_extToL1(unsigned int loc, dma_ext_t ext,
                              unsigned short size) {
  return idma_L2ToL1(ext, loc, size);
}

static inline int idma_memcpy(unsigned int src, unsigned int dst,
                              unsigned int size, idma_prot_t src_prot,
                              idma_prot_t dst_prot) {
  if (src_prot == IDMA_PROT_OBI && dst_prot == IDMA_PROT_AXI) {
    return idma_L1ToL2(src, dst, size);
  } else if (src_prot == IDMA_PROT_AXI && dst_prot == IDMA_PROT_OBI) {
    return idma_L2ToL1(src, dst, size);
  } else if (src_prot == IDMA_PROT_OBI && dst_prot == IDMA_PROT_OBI) {
    return idma_L1ToL1(src, dst, size);
  }
  return 0;
}

static inline int idma_L1ToL2(unsigned int src, unsigned int dst,
                              unsigned short size) {
  idma_mm_conf_default_dir(1);
  idma_mm_set_addr_len_dir(1, dst, src, size);
  idma_mm_set_2d_params_dir(1, 0, 0, 1);
  idma_mm_set_3d_params_dir(1, 0, 0, 1);
  return idma_mm_start_transfer_dir(1, 0);
}

static inline int collective(unsigned int src, unsigned int dst,
                             unsigned short size, uint32_t mask,
                             uint32_t collective_op) {
  idma_mm_conf_default_dir(1);
  idma_mm_set_addr_len_dir(1, dst, src, size);
  idma_mm_set_2d_params_dir(1, 0, 0, 1);
  idma_mm_set_3d_params_dir(1, 0, 0, 1);
  idma_mm_set_multicast_mask(1, mask);
  idma_mm_set_collective_sel(1, collective_op);
  return idma_mm_start_transfer_dir(1, 0);
}

static inline int idma_L2ToL1(unsigned int src, unsigned int dst,
                              unsigned short size) {
  idma_mm_conf_default_dir(0);
  idma_mm_set_addr_len_dir(0, dst, src, size);
  idma_mm_set_2d_params_dir(0, 0, 0, 1);
  idma_mm_set_3d_params_dir(0, 0, 0, 1);
  return idma_mm_start_transfer_dir(0, 0);
}

// L1-to-L1 transfers: Remote to Local (PULL)
static inline int idma_L1ToL1_pull(unsigned int remote_src,
                                   unsigned int local_dst,
                                   unsigned short size) {
  idma_mm_conf_default_dir(
      0); // AXI2OBI: read from remote (AXI), write to local (OBI)
  idma_mm_set_addr_len_dir(0, local_dst, remote_src, size);
  idma_mm_set_2d_params_dir(0, 0, 0, 1);
  idma_mm_set_3d_params_dir(0, 0, 0, 1);
  return idma_mm_start_transfer_dir(0, 0);
}

// L1-to-L1 transfers: Local to Remote (PUSH)
static inline int idma_L1ToL1_push(unsigned int local_src,
                                   unsigned int remote_dst,
                                   unsigned short size) {
  idma_mm_conf_default_dir(
      1); // OBI2AXI: read from local (OBI), write to remote (AXI)
  idma_mm_set_addr_len_dir(1, remote_dst, local_src, size);
  idma_mm_set_2d_params_dir(1, 0, 0, 1);
  idma_mm_set_3d_params_dir(1, 0, 0, 1);
  return idma_mm_start_transfer_dir(1, 0);
}

//=============================================================================
// High-Level DMA API - 2D Transfers
//=============================================================================

static inline int idma_memcpy_2d(unsigned int src, unsigned int dst,
                                 unsigned int size, unsigned int src_stride,
                                 unsigned int dst_stride, unsigned int num_reps,
                                 idma_prot_t src_prot, idma_prot_t dst_prot) {
  if (src_prot == IDMA_PROT_OBI && dst_prot == IDMA_PROT_AXI) {
    return idma_L1ToL2_2d(src, dst, size, src_stride, dst_stride, num_reps);
  } else if (src_prot == IDMA_PROT_AXI && dst_prot == IDMA_PROT_OBI) {
    return idma_L2ToL1_2d(src, dst, size, src_stride, dst_stride, num_reps);
  } else if (src_prot == IDMA_PROT_OBI && dst_prot == IDMA_PROT_OBI) {
    return idma_L1ToL1_2d(src, dst, size, src_stride, dst_stride, num_reps);
  }
  return 0;
}

static inline int idma_L1ToL2_2d(unsigned int src, unsigned int dst,
                                 unsigned short size, unsigned int src_stride,
                                 unsigned int dst_stride,
                                 unsigned int num_reps) {
  idma_mm_conf_default_dir(1);
  idma_mm_set_addr_len_dir(1, dst, src, size);
  idma_mm_set_2d_params_dir(1, dst_stride, src_stride, num_reps);
  idma_mm_set_3d_params_dir(1, 0, 0, 1);
  return idma_mm_start_transfer_dir(1, 0);
}

static inline int idma_L2ToL1_2d(unsigned int src, unsigned int dst,
                                 unsigned short size, unsigned int src_stride,
                                 unsigned int dst_stride,
                                 unsigned int num_reps) {
  idma_mm_conf_default_dir(0);
  idma_mm_set_addr_len_dir(0, dst, src, size);
  idma_mm_set_2d_params_dir(0, dst_stride, src_stride, num_reps);
  idma_mm_set_3d_params_dir(0, 0, 0, 1);
  return idma_mm_start_transfer_dir(0, 0);
}

// L1-to-L1 2D transfers: Remote to Local (PULL)
static inline int
idma_L1ToL1_pull_2d(unsigned int remote_src, unsigned int local_dst,
                    unsigned short size, unsigned int src_stride,
                    unsigned int dst_stride, unsigned int num_reps) {
  idma_mm_conf_default_dir(0); // AXI2OBI
  idma_mm_set_addr_len_dir(0, local_dst, remote_src, size);
  idma_mm_set_2d_params_dir(0, dst_stride, src_stride, num_reps);
  idma_mm_set_3d_params_dir(0, 0, 0, 1);
  return idma_mm_start_transfer_dir(0, 0);
}

// L1-to-L1 2D transfers: Local to Remote (PUSH)
static inline int
idma_L1ToL1_push_2d(unsigned int local_src, unsigned int remote_dst,
                    unsigned short size, unsigned int src_stride,
                    unsigned int dst_stride, unsigned int num_reps) {
  idma_mm_conf_default_dir(1); // OBI2AXI
  idma_mm_set_addr_len_dir(1, remote_dst, local_src, size);
  idma_mm_set_2d_params_dir(1, dst_stride, src_stride, num_reps);
  idma_mm_set_3d_params_dir(1, 0, 0, 1);
  return idma_mm_start_transfer_dir(1, 0);
}

// Legacy L1-to-L1 2D function (assumes remote->local for backward
// compatibility)
static inline int idma_L1ToL1_2d(unsigned int src, unsigned int dst,
                                 unsigned short size, unsigned int src_stride,
                                 unsigned int dst_stride,
                                 unsigned int num_reps) {
  return idma_L1ToL1_pull_2d(src, dst, size, src_stride, dst_stride, num_reps);
}

//=============================================================================
// Status and Wait Functions
//=============================================================================

static inline unsigned int idma_tx_cplt(unsigned int dma_tx_id) {
  uint32_t done_id_axi2obi = idma_mm_get_done_id_dir(0, 0);
  uint32_t done_id_obi2axi = idma_mm_get_done_id_dir(1, 0);

  return (done_id_axi2obi == dma_tx_id) || (done_id_obi2axi == dma_tx_id);
}

static inline unsigned int dma_status() {
  return idma_mm_is_busy_dir(0, 0) || idma_mm_is_busy_dir(1, 0);
}

static inline void dma_wait(unsigned int dma_tx_id) {
  while (!idma_tx_cplt(dma_tx_id)) {
    wait_nop(1);
  }
}

static inline void dma_barrier() {
  while (dma_status()) {
    wait_nop(1);
  }
}

#endif /*IDMA_MM_UTILS_H*/
