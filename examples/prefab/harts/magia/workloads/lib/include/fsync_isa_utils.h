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
 * MAGIA FractalSync ISA Utils
 */

#ifndef FSYNC_ISA_UTILS_H
#define FSYNC_ISA_UTILS_H

/* synch instruction */
// asm volatile(
//      ".word (0x0       << 25) | \     /* Reserved - 0x0 */
//             (0b00110   << 20) | \     /* R2 - t1 */
//             (0b00101   << 15) | \     /* R1 - t0 */
//             (0b010     << 12) | \     /* FUNC3 */
//             (0x0       <<  7) | \     /* Reserved - 0x0 */
//             (0b1011011 <<  0)   \n"); /* OPCODE */
static inline void fsync(volatile uint32_t id, volatile uint32_t aggregate) {
  asm volatile("addi t1, %0, 0" ::"r"(id));
  asm volatile("addi t0, %0, 0" ::"r"(aggregate));
  asm volatile(".word (0x0       << 25) | \
              (0b00110   << 20) | \
              (0b00101   << 15) | \
              (0b010     << 12) | \
              (0x0       <<  7) | \
              (0b1011011 <<  0)   \n");
}

#endif /*FSYNC_ISA_UTILS_H*/
