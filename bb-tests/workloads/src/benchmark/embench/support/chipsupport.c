/* Buckyball design support implementation for Embench

   Adapted for Buckyball baremetal environment.

   SPDX-License-Identifier: GPL-3.0-or-later */

#include "chipsupport.h"

/* Initialize design

   For Buckyball, no design-specific initialization is needed.
   All necessary setup is done by crt0.S/start.S. */

void initialise_design(void) {
  /* Empty - no design-specific initialization needed */
}
