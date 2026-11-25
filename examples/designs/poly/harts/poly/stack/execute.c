#include "kernel.h"
#include "runtime.h"

/* Execution for a hart with hardware threads: the job's entry receives the
 * hart's kernel. */
static int execute(void *context, int (*entry)(void *, void *),
                   void *argument) {
  (void)context;
  size_t count = thread_query(0, THREAD_FIELD_COUNT);
  if (!count)
    __builtin_trap();
  struct kernel_slot slots[count];
  struct kernel kernel;
  kernel_init(&kernel, slots, count);
  int status = entry(&kernel, argument);
  if (kernel.active)
    __builtin_trap();
  return status;
}

static const struct runtime_binding binding
    __attribute__((section("runtime_bindings"), used)) = {HART_SIGNATURE,
                                                          execute};
