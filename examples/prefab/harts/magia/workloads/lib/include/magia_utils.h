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
 * MAGIA Utils
 */

#ifndef MAGIA_UTILS_H
#define MAGIA_UTILS_H

#include "magia_tile_utils.h"

#define STR_OFFSET (0x00000000)
#define STR_BASE (RESERVED_START + STR_OFFSET)
#define SYNC_OFFSET (0x0000D800)
#define SYNC_BASE (RESERVED_START + SYNC_OFFSET)
#define SYNC_EN (SYNC_BASE + 0x4)

#define MESH_Y_TILES (4)
#define MESH_X_TILES (4)
#define NUM_HARTS (MESH_Y_TILES * MESH_X_TILES)

#define GET_Y_ID(mhartid) ((mhartid) / MESH_Y_TILES)
#define GET_X_ID(mhartid) ((mhartid) % MESH_X_TILES)
#define GET_ID(y_id, x_id) (((y_id) * MESH_X_TILES) + (x_id))

#define h_pprintf(x) (h_psprint(get_hartid(), x))
#define n_pprintf(x) (n_psprint(get_hartid(), x))
#define pprintf(x) (psprint(get_hartid(), x))
#define pprintln (pprintf("\n"))

static inline uint32_t get_hartid(void) {
  uint32_t hartid;
  asm volatile("csrr %0, mhartid" : "=r"(hartid) :);
  return hartid;
}

static inline void amo_increment(volatile uint32_t addr,
                                 volatile uint32_t amnt) {
  asm volatile("amoadd.w x0, %1, (%0)" ::"r"(addr), "r"(amnt) : "memory");
}

char *utoa(unsigned int value, unsigned int base, char *result) {
  if (base < 2 || base > 16) {
    *result = '\0';
    return result;
  }

  char *ptr0 = result;
  char *ptr1 = result;
  char tmp_char;
  unsigned int tmp_value;

  do {
    tmp_value = value;
    value /= base;
    *ptr0++ = "0123456789ABCDEF"[tmp_value - value * base];
  } while (value);

  *ptr0-- = '\0';

  // Reverse the string
  while (ptr1 < ptr0) {
    tmp_char = *ptr0;
    *ptr0-- = *ptr1;
    *ptr1++ = tmp_char;
  }
  return result;
}

char *bs(uint32_t x) {
  uint32_t hartid = get_hartid();
  char *address = (char *)(uintptr_t)(STR_BASE + L1_TILE_OFFSET * hartid);

  return utoa(x, 2, address);
}

char *ds(uint32_t x) {
  uint32_t hartid = get_hartid();
  char *address = (char *)(uintptr_t)(STR_BASE + L1_TILE_OFFSET * hartid);

  return utoa(x, 10, address);
}

char *hs(uint32_t x) {
  uint32_t hartid = get_hartid();
  char *address = (char *)(uintptr_t)(STR_BASE + L1_TILE_OFFSET * hartid);

  return utoa(x, 16, address);
}

void h_psprint(uint32_t hartid, const char *string) {
  mmio8(MAGIA_MAILBOX_BASE + 0x20000) = '[';
  mmio8(MAGIA_MAILBOX_BASE + 0x20000) = 'm';
  mmio8(MAGIA_MAILBOX_BASE + 0x20000) = 'h';
  mmio8(MAGIA_MAILBOX_BASE + 0x20000) = 'a';
  mmio8(MAGIA_MAILBOX_BASE + 0x20000) = 'r';
  mmio8(MAGIA_MAILBOX_BASE + 0x20000) = 't';
  mmio8(MAGIA_MAILBOX_BASE + 0x20000) = 'i';
  mmio8(MAGIA_MAILBOX_BASE + 0x20000) = 'd';
  mmio8(MAGIA_MAILBOX_BASE + 0x20000) = ' ';
  char *mhardid_str = ds(hartid);
  uint8_t mhartid_idx = 0;
  while (mhardid_str[mhartid_idx] != '\0')
    mmio8(MAGIA_MAILBOX_BASE + 0x20000) = mhardid_str[mhartid_idx++];
  mmio8(MAGIA_MAILBOX_BASE + 0x20000) = ']';
  mmio8(MAGIA_MAILBOX_BASE + 0x20000) = ' ';

  uint8_t index = 0;
  while (string[index] != '\0')
    mmio8(MAGIA_MAILBOX_BASE + 0x20000) = string[index++];
}

void n_psprint(uint32_t hartid, const char *string) {
  uint8_t index = 0;
  while (string[index] != '\0')
    mmio8(MAGIA_MAILBOX_BASE + 0x20000) = string[index++];
  mmio8(MAGIA_MAILBOX_BASE + 0x20000) = '\n';
}

void psprint(uint32_t hartid, const char *string) {
  uint8_t index = 0;
  while (string[index] != '\0')
    mmio8(MAGIA_MAILBOX_BASE + 0x20000) = string[index++];
}

#endif /*MAGIA_UTILS_H*/
