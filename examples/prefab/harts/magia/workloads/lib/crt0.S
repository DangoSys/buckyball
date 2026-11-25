/*
 * Copyright (C) 2018-2019 ETH Zurich and University of Bologna
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
 */

/*
 * Authors: Germain Haugou, ETH (germain.haugou@iis.ee.ethz.ch)
 *          Francesco Conti, ETHZ & UNIBO
 */

  .section .text
  .global _start
_start:

  # Manually reset (latch-based) registers of the RF
  addi x1, x0, 0
  addi x2, x0, 0
  addi x3, x0, 0
  addi x4, x0, 0
  addi x5, x0, 0
  addi x6, x0, 0
  addi x7, x0, 0
  addi x8, x0, 0
  addi x9, x0, 0
  addi x10, x0, 0
  addi x11, x0, 0
  addi x12, x0, 0
  addi x13, x0, 0
  addi x14, x0, 0
  addi x15, x0, 0
  addi x16, x0, 0
  addi x17, x0, 0
  addi x18, x0, 0
  addi x19, x0, 0
  addi x20, x0, 0
  addi x21, x0, 0
  addi x22, x0, 0
  addi x23, x0, 0
  addi x24, x0, 0
  addi x25, x0, 0
  addi x26, x0, 0
  addi x27, x0, 0
  addi x28, x0, 0
  addi x29, x0, 0
  addi x30, x0, 0
  addi x31, x0, 0

  # Cluster PEs will also starts here to avoid aligning another entry point
  # Just re-route them to the right entry
  csrr    a0, mhartid
  andi    a1, a0, 0x1f
  srli    a0, a0, 5
  # Control-core IRQ inputs are disconnected; use EU polling / event loads.
  csrci   mstatus, 8
  csrw    mie, zero

#if defined(__riscv_f) && defined(__cv32e40p__)
  li      t0, 0x2000
  csrrs   zero, mstatus, t0
#endif

  # clear the bss segment
  csrr    t2, mhartid
  bnez    t2, skip_bss_clear
  la      t0, _bss_start
  la      t1, _bss_end
1:
  sw      zero, 0(t0)
  addi    t0, t0, 4
  bltu    t0, t1, 1b

  la t0, bb_started
  li t1, 1
  fence rw, rw
  sw t1, 0(t0)
skip_bss_clear:
  la t0, bb_started
2:
  lw t1, 0(t0)
  beqz t1, 2b
  fence rw, rw

  /* Stack initialization: top of the CV32 stack band (0x14000), grows down.
   * Must use _stack_start (region top), NOT `stack` (region bottom + _min_stack),
   * otherwise sp starts near the floor and underflows past 0x10000. */
  la   x2, _stack_start

.section .text

  // On all other chips we simply pass 0.
  addi  a0, x0, 0
  addi  a1, x0, 0

  // Jump to main program entry point (argc = a0, argv = a1).
  la    t2, main
  jalr  x1, t2
  mv    s0, a0

  /* If program returns from main, call exit routine */
  mv   a0, s0
  la   t2, exit
  jalr  x1, t2

  .global _init
  .global _fini
_init:
_fini:
  # These don't have to do anything since we use init_array/fini_array.
  ret

exit:
  mv a0, s0
  tail bb_exit

.section .vectors, "ax"
.option norvc;

  .org 0x80
  jal x0, _start
