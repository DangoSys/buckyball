#include <kernel.h>
#include <stream.h>

static unsigned called;
static unsigned char source = 0x7e, target = 0xa5;
static void record(void *argument) {
  (void)argument;
  ++called;
}

int main(void) {
  struct kernel owner;
  const size_t count = thread_query(0, THREAD_FIELD_COUNT);
  struct kernel_slot slots[count];
  kernel_init(&owner, slots, count);
  const size_t last = count - 1;
  stream_begin(2, 1, 1);
  stream_job(slots[0].signature, 0, record, 0, 1);
  stream_job(slots[last].signature, last, record, 0, 1);
  stream_after(0);
  stream_transfer(0, 1, &source, &target, 0);
  stream_submit();
  stream_wait();
  if (called || target != 0xa5 || source != 0x7e || owner.active)
    return 1;
  for (size_t thread = 0; thread < count; ++thread)
    if (slots[thread].loaded)
      return 2;
  return 0;
}
