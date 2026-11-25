/* The piplus device runtime, on the device program executor of the framework
 * (program.h), which moves a value between the harts as a copy over the
 * coherent DDR at the front of the entry that reads it. A value the program
 * computed or reads itself the program copies at once (device_copy;
 * lower-device waits for the entry that wrote it first). */
#include "device.h"

#include <string.h>

void device_copy(uint64_t from, uint64_t to, uint64_t target, uint64_t source,
                 uint64_t bytes) {
  (void)from;
  (void)to;
  memcpy((void *)(uintptr_t)target, (const void *)(uintptr_t)source, bytes);
}
