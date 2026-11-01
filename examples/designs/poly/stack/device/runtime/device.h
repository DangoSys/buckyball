/* The poly device runtime: values move between cores in coherent DDR.
 * The program and its jobs run on the device program runtime
 * of the framework, program.h. */
#pragma once

#include <program.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Moves `bytes` from `source`, written by `after` on core `from`, to `target`
 * on core `to`; returns the job after which
 * `target` holds them. (lower-device) */
uint64_t device_transfer(uint64_t from, uint64_t to, uint64_t target,
                         uint64_t source, uint64_t bytes, uint64_t after);

#ifdef __cplusplus
}
#endif
