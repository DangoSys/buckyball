/* The piplus device runtime: values move between the harts as copies over
 * the coherent DDR (see device.c). The program and its entries run on the
 * device program executor of the framework, program.h. */
#pragma once

#include <program.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Copies `bytes` from `source` on hart `from` to `target` on hart `to` on the
 * program's hart, now. (lower-device) */
void device_copy(uint64_t from, uint64_t to, uint64_t target, uint64_t source,
                 uint64_t bytes);

#ifdef __cplusplus
}
#endif
