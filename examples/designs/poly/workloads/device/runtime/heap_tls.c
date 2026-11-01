#include <errno.h>
#include <stdint.h>
#include <stdlib.h>

extern char _end[];
extern void *_sbrk(intptr_t bytes);
static __thread volatile uint64_t marker = UINT64_C(0x123456789abcdef0);

int main(void) {
  uintptr_t hart, tls;
  asm volatile("csrr %0, mhartid" : "=r"(hart));
  if (hart != WORKLOAD_CONTROL_HART)
    return 0;
  asm volatile("mv %0, tp" : "=r"(tls));
  const size_t bytes = (size_t)300 << 20;
  char *buffer = malloc(bytes);
  if (!buffer)
    return 1;
  if (tls >= (uintptr_t)buffer && tls < (uintptr_t)buffer + bytes)
    return 2;
  buffer[0] = 1;
  buffer[bytes - 1] = 2;
  if (marker != UINT64_C(0x123456789abcdef0))
    return 3;
  free(buffer);
  void *before = _sbrk(0);
  void *oversized = _sbrk(DEVICE_HEAP_LIMIT - (uintptr_t)before + 1);
  if (oversized != (void *)-1 || errno != ENOMEM || _sbrk(0) != before)
    return 4;
  return 0;
}
