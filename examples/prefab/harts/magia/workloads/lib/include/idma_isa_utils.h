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
 * MAGIA iDMA ISA Utils
 */

#ifndef IDMA_ISA_UTILS_H
#define IDMA_ISA_UTILS_H

/* conf instruction */
// asm volatile(
//      ".word (0x0       << 27) | \     /* Reserved - 0x0 */
//             (0b11      << 26) | \     /* Enable ND extension - see iDMA
//             documentation */ (0b0       << 25) | \     /* Direction - 0 for
//             AXI2OBI (L2 to L1), 1 for OBI2AXI (L1 to L2) */ (0b000     << 22)
//             | \     /* Destination maximum logarithmic length - see iDMA
//             documentation */ (0b000     << 19) | \     /* Source maximum
//             logarithmic length - see iDMA documentation */ (0b0       << 18)
//             | \     /* Destination reduce length - see iDMA documentation */
//             (0b0       << 17) | \     /* Source reduce length - see iDMA
//             documentation */ (0b0       << 16) | \     /* Decouple R/W - see
//             iDMA documentation */ (0b0       << 15) | \     /* Decouple R/AW
//             - see iDMA documentation */ (0b000     << 12) | \     /* FUNC3 */
//             (0x0       <<  7) | \     /* Reserved - 0x0 */
//             (0b1011011 <<  0)   \n"); /* OPCODE */
static inline void idma_conf_in() {
  asm volatile(".word (0x0       << 27) | \
              (0b11      << 26) | \
              (0b0       << 25) | \
              (0b000     << 22) | \
              (0b000     << 19) | \
              (0b0       << 18) | \
              (0b0       << 17) | \
              (0b0       << 16) | \
              (0b0       << 15) | \
              (0b000     << 12) | \
              (0x0       <<  7) | \
              (0b1011011 <<  0)   \n");
}

/* conf instruction */
// asm volatile(
//      ".word (0x0       << 27) | \     /* Reserved - 0x0 */
//             (0b11      << 26) | \     /* Enable ND extension - see iDMA
//             documentation */ (0b1       << 25) | \     /* Direction - 0 for
//             AXI2OBI (L2 to L1), 1 for OBI2AXI (L1 to L2) */ (0b000     << 22)
//             | \     /* Destination maximum logarithmic length - see iDMA
//             documentation */ (0b000     << 19) | \     /* Source maximum
//             logarithmic length - see iDMA documentation */ (0b0       << 18)
//             | \     /* Destination reduce length - see iDMA documentation */
//             (0b0       << 17) | \     /* Source reduce length - see iDMA
//             documentation */ (0b0       << 16) | \     /* Decouple R/W - see
//             iDMA documentation */ (0b0       << 15) | \     /* Decouple R/AW
//             - see iDMA documentation */ (0b000     << 12) | \     /* FUNC3 */
//             (0x0       <<  7) | \     /* Reserved - 0x0 */
//             (0b1011011 <<  0)   \n"); /* OPCODE */
static inline void idma_conf_out() {
  asm volatile(".word (0x0       << 27) | \
              (0b11      << 26) | \
              (0b1       << 25) | \
              (0b000     << 22) | \
              (0b000     << 19) | \
              (0b0       << 18) | \
              (0b0       << 17) | \
              (0b0       << 16) | \
              (0b0       << 15) | \
              (0b000     << 12) | \
              (0x0       <<  7) | \
              (0b1011011 <<  0)   \n");
}

/* set instruction */
// asm volatile(
//      ".word (0b00111   << 27) | \     /* R3 - t2 */
//             (0x0       << 26) | \     /* Reserved - 0x0 */
//             (0b0       << 25) | \     /* Direction - 0 for AXI2OBI (L2 to
//             L1), 1 for OBI2AXI (L1 to L2) */ (0b00110   << 20) | \     /* R2
//             - t1 */ (0b00101   << 15) | \     /* R1 - t0 */ (0b000     << 12)
//             | \     /* FUNC3 - ADDR/LEN */ (0x0       <<  7) | \     /*
//             Reserved - 0x0 */ (0b1111011 <<  0)   \n"); /* OPCODE */
static inline void idma_set_addr_len_in(volatile uint32_t dst_addr,
                                        volatile uint32_t src_addr,
                                        volatile uint32_t len) {
  asm volatile("addi t2, %0, 0" ::"r"(dst_addr));
  asm volatile("addi t1, %0, 0" ::"r"(src_addr));
  asm volatile("addi t0, %0, 0" ::"r"(len));
  asm volatile(".word (0b00111   << 27) | \
              (0x0       << 26) | \
              (0b0       << 25) | \
              (0b00110   << 20) | \
              (0b00101   << 15) | \
              (0b000     << 12) | \
              (0x0       <<  7) | \
              (0b1111011 <<  0)   \n");
}

/* set instruction */
// asm volatile(
//      ".word (0b00111   << 27) | \     /* R3 - t2 */
//             (0x0       << 26) | \     /* Reserved - 0x0 */
//             (0b1       << 25) | \     /* Direction - 0 for AXI2OBI (L2 to
//             L1), 1 for OBI2AXI (L1 to L2) */ (0b00110   << 20) | \     /* R2
//             - t1 */ (0b00101   << 15) | \     /* R1 - t0 */ (0b000     << 12)
//             | \     /* FUNC3 - ADDR/LEN */ (0x0       <<  7) | \     /*
//             Reserved - 0x0 */ (0b1111011 <<  0)   \n"); /* OPCODE */
static inline void idma_set_addr_len_out(volatile uint32_t dst_addr,
                                         volatile uint32_t src_addr,
                                         volatile uint32_t len) {
  asm volatile("addi t2, %0, 0" ::"r"(dst_addr));
  asm volatile("addi t1, %0, 0" ::"r"(src_addr));
  asm volatile("addi t0, %0, 0" ::"r"(len));
  asm volatile(".word (0b00111   << 27) | \
              (0x0       << 26) | \
              (0b1       << 25) | \
              (0b00110   << 20) | \
              (0b00101   << 15) | \
              (0b000     << 12) | \
              (0x0       <<  7) | \
              (0b1111011 <<  0)   \n");
}

/* set instruction */
// asm volatile(
//      ".word (0b00111   << 27) | \     /* R3 - t2 */
//             (0x0       << 26) | \     /* Reserved - 0x0 */
//             (0b0       << 25) | \     /* Direction - 0 for AXI2OBI (L2 to
//             L1), 1 for OBI2AXI (L1 to L2) */ (0b00110   << 20) | \     /* R2
//             - t1 */ (0b00101   << 15) | \     /* R1 - t0 */ (0b001     << 12)
//             | \     /* FUNC3 - STD_2/REP_2 */ (0x0       <<  7) | \     /*
//             Reserved - 0x0 */ (0b1111011 <<  0)   \n"); /* OPCODE */
static inline void idma_set_std2_rep2_in(volatile uint32_t dst_std_2,
                                         volatile uint32_t src_std_2,
                                         volatile uint32_t reps_2) {
  asm volatile("addi t2, %0, 0" ::"r"(dst_std_2));
  asm volatile("addi t1, %0, 0" ::"r"(src_std_2));
  asm volatile("addi t0, %0, 0" ::"r"(reps_2));
  asm volatile(".word (0b00111   << 27) | \
              (0x0       << 26) | \
              (0b0       << 25) | \
              (0b00110   << 20) | \
              (0b00101   << 15) | \
              (0b001     << 12) | \
              (0x0       <<  7) | \
              (0b1111011 <<  0)   \n");
}

/* set instruction */
// asm volatile(
//      ".word (0b00111   << 27) | \     /* R3 - t2 */
//             (0x0       << 26) | \     /* Reserved - 0x0 */
//             (0b1       << 25) | \     /* Direction - 0 for AXI2OBI (L2 to
//             L1), 1 for OBI2AXI (L1 to L2) */ (0b00110   << 20) | \     /* R2
//             - t1 */ (0b00101   << 15) | \     /* R1 - t0 */ (0b001     << 12)
//             | \     /* FUNC3 - STD_2/REP_2 */ (0x0       <<  7) | \     /*
//             Reserved - 0x0 */ (0b1111011 <<  0)   \n"); /* OPCODE */
static inline void idma_set_std2_rep2_out(volatile uint32_t dst_std_2,
                                          volatile uint32_t src_std_2,
                                          volatile uint32_t reps_2) {
  asm volatile("addi t2, %0, 0" ::"r"(dst_std_2));
  asm volatile("addi t1, %0, 0" ::"r"(src_std_2));
  asm volatile("addi t0, %0, 0" ::"r"(reps_2));
  asm volatile(".word (0b00111   << 27) | \
              (0x0       << 26) | \
              (0b1       << 25) | \
              (0b00110   << 20) | \
              (0b00101   << 15) | \
              (0b001     << 12) | \
              (0x0       <<  7) | \
              (0b1111011 <<  0)   \n");
}

/* set instruction */
// asm volatile(
//      ".word (0b00111   << 27) | \     /* R3 - t2 */
//             (0x0       << 26) | \     /* Reserved - 0x0 */
//             (0b0       << 25) | \     /* Direction - 0 for AXI2OBI (L2 to
//             L1), 1 for OBI2AXI (L1 to L2) */ (0b00110   << 20) | \     /* R2
//             - t1 */ (0b00101   << 15) | \     /* R1 - t0 */ (0b010     << 12)
//             | \     /* FUNC3 - STD_3/REP_3 */ (0x0       <<  7) | \     /*
//             Reserved - 0x0 */ (0b1111011 <<  0)   \n"); /* OPCODE */
static inline void idma_set_std3_rep3_in(volatile uint32_t dst_std_3,
                                         volatile uint32_t src_std_3,
                                         volatile uint32_t reps_3) {
  asm volatile("addi t2, %0, 0" ::"r"(dst_std_3));
  asm volatile("addi t1, %0, 0" ::"r"(src_std_3));
  asm volatile("addi t0, %0, 0" ::"r"(reps_3));
  asm volatile(".word (0b00111   << 27) | \
              (0x0       << 26) | \
              (0b0       << 25) | \
              (0b00110   << 20) | \
              (0b00101   << 15) | \
              (0b010     << 12) | \
              (0x0       <<  7) | \
              (0b1111011 <<  0)   \n");
}

/* set instruction */
// asm volatile(
//      ".word (0b00111   << 27) | \     /* R3 - t2 */
//             (0x0       << 26) | \     /* Reserved - 0x0 */
//             (0b1       << 25) | \     /* Direction - 0 for AXI2OBI (L2 to
//             L1), 1 for OBI2AXI (L1 to L2) */ (0b00110   << 20) | \     /* R2
//             - t1 */ (0b00101   << 15) | \     /* R1 - t0 */ (0b010     << 12)
//             | \     /* FUNC3 - STD_3/REP_3 */ (0x0       <<  7) | \     /*
//             Reserved - 0x0 */ (0b1111011 <<  0)   \n"); /* OPCODE */
static inline void idma_set_std3_rep3_out(volatile uint32_t dst_std_3,
                                          volatile uint32_t src_std_3,
                                          volatile uint32_t reps_3) {
  asm volatile("addi t2, %0, 0" ::"r"(dst_std_3));
  asm volatile("addi t1, %0, 0" ::"r"(src_std_3));
  asm volatile("addi t0, %0, 0" ::"r"(reps_3));
  asm volatile(".word (0b00111   << 27) | \
              (0x0       << 26) | \
              (0b1       << 25) | \
              (0b00110   << 20) | \
              (0b00101   << 15) | \
              (0b010     << 12) | \
              (0x0       <<  7) | \
              (0b1111011 <<  0)   \n");
}

/* start instruction */
// asm volatile(
//      ".word (0x0       << 26) | \     /* Reserved - 0x0 */
//             (0b0       << 25) | \     /* Direction - 0 for AXI2OBI (L2 to
//             L1), 1 for OBI2AXI (L1 to L2) */ (0x0       << 15) | \     /*
//             Reserved - 0x0 */ (0b111     << 12) | \     /* FUNC3 - START */
//             (0x0       <<  7) | \     /* Reserved - 0x0 */
//             (0b1111011 <<  0)   \n"); /* OPCODE */
static inline void idma_start_in() {
  asm volatile(".word (0x0       << 26) | \
              (0b0       << 25) | \
              (0x0       << 15) | \
              (0b111     << 12) | \
              (0x0       <<  7) | \
              (0b1111011 <<  0)   \n");
}

/* start instruction */
// asm volatile(
//      ".word (0x0       << 26) | \     /* Reserved - 0x0 */
//             (0b1       << 25) | \     /* Direction - 0 for AXI2OBI (L2 to
//             L1), 1 for OBI2AXI (L1 to L2) */ (0x0       << 15) | \     /*
//             Reserved - 0x0 */ (0b111     << 12) | \     /* FUNC3 - START */
//             (0x0       <<  7) | \     /* Reserved - 0x0 */
//             (0b1111011 <<  0)   \n"); /* OPCODE */
static inline void idma_start_out() {
  asm volatile(".word (0x0       << 26) | \
              (0b1       << 25) | \
              (0x0       << 15) | \
              (0b111     << 12) | \
              (0x0       <<  7) | \
              (0b1111011 <<  0)   \n");
}

#endif /*IDMA_ISA_UTILS_H*/
