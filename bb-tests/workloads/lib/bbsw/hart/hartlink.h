#ifndef BUCKYBALL_CORELINK_H
#define BUCKYBALL_CORELINK_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum hartlink_field {
  HARTLINK_CAPACITY,
  HARTLINK_HART_ID,
  HARTLINK_HART_COUNT
};

uint64_t hartlink_read64(uint64_t byte_address);
void hartlink_write64(uint64_t byte_address, uint64_t data);

uint64_t hartlink_query(enum hartlink_field field);
/* Reports a direct hardware route from the calling hart to the target. */
int hartlink_connected(uint32_t target_hart);
/* Copies bytes between device communication buffers and returns after commit.
 */
void hartlink_transfer(uint64_t source_byte_address, uint32_t target_hart,
                       uint64_t target_byte_address, uint32_t bytes);

#ifdef __cplusplus
}
#endif

#endif
