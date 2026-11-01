/* All poly cores access the same coherent DDR. A logical transfer is one
 * copy on the receiving core, after the job that produced its source. */
#include "device.h"

uint64_t device_transfer(uint64_t from, uint64_t to, uint64_t target,
                         uint64_t source, uint64_t bytes, uint64_t after) {
  (void)from;
  return program_copy(to, target, source, bytes, after);
}
