/* The piplus device runtime: values move between the cores as copies over
 * the coherent DDR (see device.c). The program and its jobs run on the
 * device program runtime of the framework, program.h. */
#pragma once

#include <program.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Copies `bytes` from `source` on core `from`, written by `after`, to
 * `target` on core `to`, which copies once `after` is done; returns the job
 * of the copy. (lower-device) */
uint64_t device_transfer(uint64_t from, uint64_t to, uint64_t target,
                         uint64_t source, uint64_t bytes, uint64_t after);
/* Copies `bytes` from `source` on core `from` to `target` on core `to` on the
 * program's core, now. (lower-device) */
void device_copy(uint64_t from, uint64_t to, uint64_t target, uint64_t source,
                 uint64_t bytes);

#ifdef __cplusplus
}
#endif
