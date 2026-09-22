#include <buckyball/rushb.h>

#if defined(BUCKYBALL_RUSHB_BEMU)
#define RUSHB_BACKEND(name) bemu_##name
#elif defined(BUCKYBALL_RUSHB_VERILATOR)
#define RUSHB_BACKEND(name) verilator_##name
#else
#error "RushB backend is not selected"
#endif

extern void RUSHB_BACKEND(rushb_init)(void);
extern void RUSHB_BACKEND(rushb_destroy)(void);
extern void RUSHB_BACKEND(rushb_mset)(uint32_t core_id, uint64_t xs1,
                                      uint64_t xs2);
extern void RUSHB_BACKEND(rushb_mvin)(uint32_t core_id, uint64_t xs1,
                                      uint64_t packed_xs2,
                                      const void *host_ptr);
extern void RUSHB_BACKEND(rushb_mvin_mmio)(uint32_t core_id, uint64_t xs1,
                                           uint64_t packed_xs2,
                                           const void *host_ptr);
extern void RUSHB_BACKEND(rushb_mvout)(uint32_t core_id, uint64_t xs1,
                                       uint64_t packed_xs2, void *host_ptr);
extern void RUSHB_BACKEND(rushb_custom)(uint32_t core_id, uint64_t xs1,
                                        uint64_t xs2, uint32_t funct7);
extern uint64_t RUSHB_BACKEND(rushb_cycles)(uint32_t core_id);

void rushb_init(void) { RUSHB_BACKEND(rushb_init)(); }
void rushb_destroy(void) { RUSHB_BACKEND(rushb_destroy)(); }
void rushb_mset(uint32_t core_id, uint64_t xs1, uint64_t xs2) {
  RUSHB_BACKEND(rushb_mset)(core_id, xs1, xs2);
}
void rushb_mvin(uint32_t core_id, uint64_t xs1, uint64_t packed_xs2,
                const void *host_ptr) {
  RUSHB_BACKEND(rushb_mvin)(core_id, xs1, packed_xs2, host_ptr);
}
void rushb_mvin_mmio(uint32_t core_id, uint64_t xs1, uint64_t packed_xs2,
                     const void *host_ptr) {
  RUSHB_BACKEND(rushb_mvin_mmio)(core_id, xs1, packed_xs2, host_ptr);
}
void rushb_mvout(uint32_t core_id, uint64_t xs1, uint64_t packed_xs2,
                 void *host_ptr) {
  RUSHB_BACKEND(rushb_mvout)(core_id, xs1, packed_xs2, host_ptr);
}
void rushb_custom(uint32_t core_id, uint64_t xs1, uint64_t xs2,
                  uint32_t funct7) {
  RUSHB_BACKEND(rushb_custom)(core_id, xs1, xs2, funct7);
}
uint64_t rushb_cycles(uint32_t core_id) {
  return RUSHB_BACKEND(rushb_cycles)(core_id);
}

static void __attribute__((constructor)) rushb_runtime_init(void) {
  rushb_init();
}

static void __attribute__((destructor)) rushb_runtime_destroy(void) {
  rushb_destroy();
}
