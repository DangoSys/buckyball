#ifndef BUCKYBALL_RUSHB_H
#define BUCKYBALL_RUSHB_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void rushb_init(void);
void rushb_destroy(void);
void rushb_select_accelerator(uint32_t accelerator_id, int32_t chip_id);
void rushb_mset(uint64_t xs1, uint64_t xs2);
void rushb_mvin(uint64_t xs1, uint64_t packed_xs2, const void *host_ptr);
void rushb_mvout(uint64_t xs1, uint64_t packed_xs2, void *host_ptr);
void rushb_custom(uint64_t xs1, uint64_t xs2, uint32_t funct7);
uint64_t rushb_cycles(void);

#ifdef __cplusplus
}
#endif

#endif
