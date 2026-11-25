#ifndef BBSW_THREAD_RUN_H
#define BBSW_THREAD_RUN_H

#ifdef BUCKYBALL_STREAM
#include <kernel.h>
#include <stream.h>

/* Runs `body` on the hart with its Ball instructions (BUCKYBALL_STREAM) on
 * the hart's hardware thread of THREAD_SIGNATURE: they go out in batches, a
 * batch at each DMA instruction and at the end, each run to completion. */
static inline void bb_thread_run(void (*body)(void *), void *argument) {
  struct kernel kernel;
  struct kernel_slot slots[thread_query(0, THREAD_FIELD_COUNT)];
  kernel_init(&kernel, slots, sizeof(slots) / sizeof(slots[0]));
  stream_run(THREAD_SIGNATURE, body, argument);
}
#else
/* A hart's own accelerator executes the Ball instructions of `body` directly.
 */
static inline void bb_thread_run(void (*body)(void *), void *argument) {
  body(argument);
}
#endif

#endif
