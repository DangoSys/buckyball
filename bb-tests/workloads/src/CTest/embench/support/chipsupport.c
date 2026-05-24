/* Buckyball chip support implementation for Embench

   Adapted for Buckyball baremetal environment.

   SPDX-License-Identifier: GPL-3.0-or-later */

#include "chipsupport.h"

/* Initialize chip

   For Buckyball, no chip-specific initialization is needed.
   All necessary setup is done by crt0.S/start.S. */

void initialise_chip(void) {
  /* Empty - no chip-specific initialization needed */
}
