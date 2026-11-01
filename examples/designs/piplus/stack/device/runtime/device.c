/* The piplus device runtime, on the device program runtime (program.h).
 *
 * The cores share the coherent DDR. A value moves to another core as a copy
 * into a buffer of that core: a value the program computed or reads itself,
 * the program copies at once (device_copy; lower-device waits for the job
 * that wrote it first); between two cores, a copy job on the receiving core
 * runs after the job that wrote the value (device_transfer).
 * The scheduler orders the jobs; its lock orders their memory. */
#include "device.h"

#include <string.h>

uint64_t device_transfer(uint64_t from, uint64_t to, uint64_t target,
                         uint64_t source, uint64_t bytes, uint64_t after) {
  (void)from;
  return program_copy(to, target, source, bytes, after);
}

void device_copy(uint64_t from, uint64_t to, uint64_t target, uint64_t source,
                 uint64_t bytes) {
  (void)from;
  (void)to;
  memcpy((void *)(uintptr_t)target, (const void *)(uintptr_t)source, bytes);
}
