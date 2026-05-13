/* Buckyball board support header for Embench

   Adapted for Buckyball baremetal environment.

   SPDX-License-Identifier: GPL-3.0-or-later */

#ifndef BOARDSUPPORT_H
#define BOARDSUPPORT_H

/* CPU frequency in MHz (for benchmark scaling)
   1 = full benchmark (may be slow on Verilator) */
#define CPU_MHZ 1

/* Warmup heat level (reduced for faster testing) */
#define WARMUP_HEAT 0

/* Board initialization function */
void initialise_board(void);

/* Trigger functions for performance measurement */
void start_trigger(void);
void stop_trigger(void);

#endif /* BOARDSUPPORT_H */
