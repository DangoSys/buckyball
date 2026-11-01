#include "corelink.h"

#define COMMAND(op, field, value)                                              \
  ({                                                                           \
    uint64_t result;                                                           \
    __asm__ volatile(".insn r 0x2b, 7, " #op ", %0, %1, %2"                    \
                     : "=r"(result)                                            \
                     : "r"((uint64_t)(field)), "r"((uint64_t)(value))          \
                     : "memory");                                              \
    result;                                                                    \
  })

uint64_t corelink_query(enum corelink_field field) {
  return COMMAND(15, field, 0);
}

int corelink_connected(uint32_t target_core) {
  return COMMAND(15, 3, target_core) != 0;
}

void corelink_transfer(uint64_t source_byte_address, uint32_t target_core,
                       uint64_t target_byte_address, uint32_t bytes) {
  COMMAND(13, 0, source_byte_address);
  COMMAND(13, 1, target_core);
  COMMAND(13, 2, target_byte_address);
  COMMAND(13, 3, bytes);
  if (COMMAND(14, 0, 0))
    __builtin_trap();
}

uint64_t corelink_read64(uint64_t byte_address) {
  return COMMAND(16, byte_address, 0);
}

void corelink_write64(uint64_t byte_address, uint64_t data) {
  if (COMMAND(17, byte_address, data))
    __builtin_trap();
}
