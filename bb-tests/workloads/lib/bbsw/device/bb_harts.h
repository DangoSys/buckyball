/* The harts of the device, from DESIGN_DEFS (design.cmake): the X-list
 * DEVICE_HARTS of their hart ids, DEVICE_HARTS_<hart>_<FIELD> of each,
 * and PROFILES of the compiler profiles. A hart's profile is the index of
 * its profile in PROFILES, UINT32_MAX for a hart without one. */
#ifndef BB_HARTS_H
#define BB_HARTS_H

#include <stdint.h>

#define X(name) bb_profile_##name,
enum { PROFILES bb_num_profiles };
#undef X
#define bb_profile_ UINT32_MAX
#define BB_PROFILE_INDEX_(name) bb_profile_##name
#define BB_PROFILE_INDEX(name) BB_PROFILE_INDEX_(name)

typedef struct {
  uint32_t hartid, profile;
  uint64_t signature;
} bb_hart_instance;

#define X(hart)                                                                \
  {DEVICE_HARTS_##hart##_HART_ID,                                              \
   BB_PROFILE_INDEX(DEVICE_HARTS_##hart##_PROFILE),                            \
   DEVICE_HARTS_##hart##_SIGNATURE},
static const bb_hart_instance bb_harts[] = {DEVICE_HARTS};
#undef X
#define BB_NUM_HARTS (sizeof(bb_harts) / sizeof(bb_harts[0]))

static inline uint32_t bb_hart_slot(uint32_t hartid) {
  for (uint32_t i = 0; i < BB_NUM_HARTS; ++i)
    if (bb_harts[i].hartid == hartid)
      return i;
  __builtin_trap();
}

static inline uint32_t bb_profile_hart_count(uint32_t profile) {
  uint32_t count = 0;
  for (uint32_t i = 0; i < BB_NUM_HARTS; ++i)
    count += bb_harts[i].profile == profile;
  return count;
}

static inline uint32_t bb_hart_profile(uint32_t hartid) {
  for (uint32_t i = 0; i < BB_NUM_HARTS; ++i)
    if (bb_harts[i].hartid == hartid)
      return bb_harts[i].profile;
  return UINT32_MAX;
}

#endif
