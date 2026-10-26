#include "engine.h"
#include "runtime.h"

/* Execution for a core with hardware threads: the job's entry receives the
 * core's engine. */
static int execute(void *context, int (*entry)(void *, void *),
                   void *argument) {
  (void)context;
  size_t count = thread_query(0, THREAD_FIELD_COUNT);
  if (!count)
    __builtin_trap();
  struct engine_slot slots[count];
  struct engine engine;
  engine_init(&engine, slots, count);
  int status = entry(&engine, argument);
  if (engine.active)
    __builtin_trap();
  return status;
}

static const struct runtime_binding binding
    __attribute__((section("runtime_bindings"), used)) = {CORE_SIGNATURE,
                                                          execute};
