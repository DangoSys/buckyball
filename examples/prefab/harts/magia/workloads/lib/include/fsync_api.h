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
 * Authors: Victor Isachi <victor.isachi@unibo.it>
 *
 * MAGIA FractalSync common synchronization patterns API
 */

#ifndef FSYNC_API_H
#define FSYNC_API_H

#include "fsync_isa_utils.h"
#include "magia_tile_utils.h"
#include "magia_utils.h"

#define _FS_GLOBAL_AGGR (0xFFFFFFFF >> (1 + __builtin_clz(NUM_HARTS)))
#define _FS_GLOBAL_ID (-1)
#define _FS_HNBR_AGGR (0x1)
#define _FS_HNBR_ID (0)
#define _FS_VNBR_AGGR (0x1)
#define _FS_VNBR_ID (1)
#define _FS_HRING_AGGR (0x1)
#define _FS_HRING_ID (2)
#define _FS_VRING_AGGR (0x1)
#define _FS_VRING_ID (3)
#define _FS_RC_LVL (0x1 << (29 - __builtin_clz(NUM_HARTS)))
#define _FS_RC_AGGR (0x155 >> (__builtin_clz(NUM_HARTS) - 21))

// Lookup table indicating the id of row synchronization
uint32_t row_id_lookup(volatile uint32_t hartid_y) {
  if (hartid_y < MESH_Y_TILES / 2)
    return 2 * hartid_y;
  else
    return 2 * (hartid_y - MESH_Y_TILES / 2);
}

// Lookup table indicating the id of column synchronization
uint32_t col_id_lookup(volatile uint32_t hartid_x) {
  if (hartid_x < MESH_X_TILES / 2)
    return 2 * hartid_x + 1;
  else
    return 2 * (hartid_x - MESH_X_TILES / 2) + 1;
}

static inline void fsync_hnbr() { fsync(_FS_HNBR_ID, _FS_HNBR_AGGR); }

static inline void fsync_vnbr() { fsync(_FS_VNBR_ID, _FS_VNBR_AGGR); }

void fsync_hring() {
  uint32_t hartid = get_hartid();
  uint32_t hartid_x = GET_X_ID(hartid);
  uint32_t hartid_y = GET_Y_ID(hartid);
  if ((hartid_x == 0) || (hartid_x == MESH_X_TILES - 1)) {
    uint32_t id = row_id_lookup(hartid_y);
    fsync(id, _FS_RC_LVL);
  } else {
    fsync(_FS_HRING_ID, _FS_HRING_AGGR);
  }
}

void fsync_vring() {
  uint32_t hartid = get_hartid();
  uint32_t hartid_x = GET_X_ID(hartid);
  uint32_t hartid_y = GET_Y_ID(hartid);
  if ((hartid_y == 0) || (hartid_y == MESH_Y_TILES - 1)) {
    uint32_t id = col_id_lookup(hartid_x);
    fsync(id, _FS_RC_LVL);
  } else {
    fsync(_FS_VRING_ID, _FS_VRING_AGGR);
  }
}

void fsync_rows() {
  uint32_t hartid = get_hartid();
  uint32_t hartid_y = GET_Y_ID(hartid);
  uint32_t id = row_id_lookup(hartid_y);
  fsync(id, _FS_RC_AGGR);
}

void fsync_cols() {
  uint32_t hartid = get_hartid();
  uint32_t hartid_x = GET_X_ID(hartid);
  uint32_t id = col_id_lookup(hartid_x);
  fsync(id, _FS_RC_AGGR);
}

static inline void fsync_global() { fsync(_FS_GLOBAL_ID, _FS_GLOBAL_AGGR); }

#endif /*FSYNC_API_H*/
