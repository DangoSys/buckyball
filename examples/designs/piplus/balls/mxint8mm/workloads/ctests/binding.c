#include "task.h"
#include <engine.h>

static const unsigned char image[] __attribute__((aligned(8))) = {
#embed "image.bin"
};

/* Loads the test's thread program into a thread, once. */
static void load(uint32_t thread) {
  if (sizeof(image) > thread_query(thread, THREAD_FIELD_CODE_BYTES))
    __builtin_trap();
  thread_tss_write(0, image, sizeof(image));
  thread_load(thread, THREAD_CODE, 0, 0, (sizeof(image) + 7) & ~(size_t)7);
}

/* Hands a thread the argument of one job. */
static void prepare(uint32_t thread, void *opaque, struct thread_task *task) {
  uint64_t base = thread_query(thread, THREAD_FIELD_TLS_BASE);
  thread_tss_write(0, opaque, sizeof(struct ball_task));
  thread_load(thread, THREAD_TLS, 0, 0, sizeof(struct ball_task));
  task->entry = 0;
  task->code_end = sizeof(image);
  task->argument = base;
  task->stack = base + thread_query(thread, THREAD_FIELD_TLS_BYTES);
}

static const struct engine_binding binding
    __attribute__((section("engine_bindings"), used)) = {THREAD_SIGNATURE, load,
                                                         prepare};
