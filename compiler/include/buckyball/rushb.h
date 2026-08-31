#ifndef BUCKYBALL_RUSHB_H
#define BUCKYBALL_RUSHB_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void rushb_init(void);
void rushb_destroy(void);
void rushb_mset(uint32_t core_id, uint64_t xs1, uint64_t xs2);
void rushb_mvin(uint32_t core_id, uint64_t xs1, uint64_t packed_xs2,
                const void *host_ptr);
void rushb_mvin_mmio(uint32_t core_id, uint64_t xs1, uint64_t packed_xs2,
                     const void *host_ptr);
void rushb_mvout(uint32_t core_id, uint64_t xs1, uint64_t packed_xs2,
                 void *host_ptr);
void rushb_custom(uint32_t core_id, uint64_t xs1, uint64_t xs2,
                  uint32_t funct7);
uint64_t rushb_cycles(uint32_t core_id);

#ifdef __cplusplus
}
#endif

#endif
