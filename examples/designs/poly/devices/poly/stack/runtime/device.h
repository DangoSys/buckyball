/* The poly device runtime: the executor of a static device program on the
 * poly harts, values moving between them through the device SHM over the
 * hart links (see device.c). It implements the program contract of the
 * framework, program.h. */
#pragma once

#include <program.h>
