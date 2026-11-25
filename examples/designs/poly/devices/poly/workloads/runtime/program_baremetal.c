#include <baremetal.h>

static struct runtime runtime;
static struct runtime_job jobs[16];
static struct runtime_result results[16];
#define X(hartid) {DEVICE_HARTS_##hartid##_HART_ID, 1, 0, execute},
static int execute(void *context, int (*entry)(void *, void *),
                   void *argument) {
  return entry(context, argument);
}
static const struct runtime_binding binding
    __attribute__((section("runtime_bindings"), used)) = {1, execute};
static struct runtime_hart harts[] = {DEVICE_HARTS};
#undef X
static unsigned started, finished, value;
static size_t dependency;

static int increment(void *hart, void *argument) {
  (void)hart;
  value += (uintptr_t)argument;
  return 0;
}

static int fail(void *hart, void *argument) {
  (void)hart;
  (void)argument;
  return 7;
}

static int submit(void *hart, void *argument) {
  (void)hart;
  (void)argument;
  for (unsigned index = 0; index < 8; ++index) {
    uint64_t begin, now;
    asm volatile("rdcycle %0" : "=r"(begin));
    do {
      asm volatile("rdcycle %0" : "=r"(now));
    } while (now - begin < 64);
    struct runtime_job job = {
        .signature = 1, .hart = 1, .entry = increment, .argument = (void *)1};
    size_t submitted = runtime_submit(&runtime, &job);
    // Hart 0 waits for the job: its completion wakes hart 0.
    __atomic_fetch_or(&results[submitted].dependents, 1, __ATOMIC_SEQ_CST);
    runtime_baremetal_notify(&runtime, 2);
    for (;;) {
      runtime_baremetal_prepare_wait(harts[0].id);
      if (__atomic_load_n(&results[submitted].state, __ATOMIC_ACQUIRE) >=
          RUNTIME_SUCCEEDED)
        break;
      asm volatile("wfi" ::: "memory");
    }
    if (results[submitted].state != RUNTIME_SUCCEEDED || value != index + 1)
      return 1;
  }
  struct runtime_job job = {.signature = 1, .hart = 1, .entry = fail};
  dependency = runtime_submit(&runtime, &job);
  job.dependencies = &dependency;
  job.dependency_count = 1;
  runtime_submit(&runtime, &job);
  runtime_baremetal_notify(&runtime, 2);
  return 0;
}

int main(void) {
  uintptr_t hartid;
  asm volatile("csrr %0, mhartid" : "=r"(hartid));
  size_t hart = 0;
  while (harts[hart].id != hartid)
    ++hart;
  if (hart == 0) {
    jobs[0] = (struct runtime_job){.signature = 1, .hart = 0, .entry = submit};
    runtime_init(&runtime, harts, sizeof(harts) / sizeof(harts[0]), jobs, 1,
                 results);
    __atomic_store_n(&started, 1, __ATOMIC_RELEASE);
  }
  while (!__atomic_load_n(&started, __ATOMIC_ACQUIRE)) {
  }
  runtime_baremetal_run(&runtime, hart);
  __atomic_fetch_add(&finished, 1, __ATOMIC_RELEASE);
  while (__atomic_load_n(&finished, __ATOMIC_ACQUIRE) != runtime.hart_count) {
  }
  return value != 8 || runtime.remaining || results[0].status ||
         results[9].state != RUNTIME_FAILED ||
         results[10].state != RUNTIME_CANCELLED;
}
