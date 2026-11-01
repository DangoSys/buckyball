#ifndef BUCKYBALL_CORELINK_H
#define BUCKYBALL_CORELINK_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum corelink_field {
  CORELINK_CAPACITY,
  CORELINK_CORE_ID,
  CORELINK_CORE_COUNT
};

uint64_t corelink_read64(uint64_t byte_address);
void corelink_write64(uint64_t byte_address, uint64_t data);

uint64_t corelink_query(enum corelink_field field);
/* Reports a direct hardware route from the calling core to the target. */
int corelink_connected(uint32_t target_core);
/* Copies bytes between device communication buffers and returns after commit.
 */
void corelink_transfer(uint64_t source_byte_address, uint32_t target_core,
                       uint64_t target_byte_address, uint32_t bytes);

#ifdef __cplusplus
}
#endif

#endif
