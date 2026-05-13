/* Buckyball board support implementation for Embench

   Adapted for Buckyball baremetal environment.

   SPDX-License-Identifier: GPL-3.0-or-later */

#include "boardsupport.h"

/* Initialize board

   For Buckyball, crt0.S/start.S already handles all necessary initialization
   (stack setup, BSS clearing), so this is a no-op. */

void initialise_board(void) {
  /* Empty - initialization done by crt0.S/start.S */
}

/* Start performance measurement trigger

   This is a compiler barrier to prevent code motion across the trigger point.
   In a real system, this might write to a performance counter or GPIO. */

void __attribute__((noinline)) start_trigger(void) {
  /* Memory barrier to prevent compiler reordering */
  asm volatile("" ::: "memory");
}

/* Stop performance measurement trigger

   This is a compiler barrier to prevent code motion across the trigger point.
   In a real system, this might write to a performance counter or GPIO. */

void __attribute__((noinline)) stop_trigger(void) {
  /* Memory barrier to prevent compiler reordering */
  asm volatile("" ::: "memory");
}
