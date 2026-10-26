#include "matmul.h"
#include "params.h"
#include <engine.h>

static const unsigned char image[] __attribute__((aligned(8))) = {
#embed "program.bin"
};

/* Loads the matmul program and its task descriptor into a thread. */
static void prepare(uint32_t thread, void *opaque, struct thread_task *task) {
  size_t code_bytes = sizeof(image);
  size_t data_bytes = sizeof(struct matmul_task);
  uint64_t tls_base = thread_query(thread, THREAD_FIELD_TLS_BASE);
  uint64_t tls_bytes = thread_query(thread, THREAD_FIELD_TLS_BYTES);
  if (code_bytes > thread_query(thread, THREAD_FIELD_CODE_BYTES) ||
      ((data_bytes + 7) & ~(size_t)7) + 256 > tls_bytes ||
      code_bytes > thread_query(thread, THREAD_FIELD_TSS_BYTES) ||
      data_bytes > thread_query(thread, THREAD_FIELD_TSS_BYTES))
    __builtin_trap();
  thread_tss_write(0, image, code_bytes);
  thread_load(thread, THREAD_CODE, 0, 0, (code_bytes + 7) & ~(size_t)7);
  thread_tss_write(0, opaque, data_bytes);
  thread_load(thread, THREAD_TLS, 0, 0, (data_bytes + 7) & ~(size_t)7);
  task->entry = 0;
  task->code_end = code_bytes;
  task->argument = tls_base;
  task->stack = tls_base + tls_bytes;
}

static const struct engine_binding binding
    __attribute__((section("engine_bindings"), used)) = {THREAD_SIGNATURE,
                                                         prepare};
