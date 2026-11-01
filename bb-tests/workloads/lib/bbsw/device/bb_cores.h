/* The cores of the device, from DESIGN_DEFS (design.cmake): the X-list
 * DEVICE_CORES of their hart ids, DEVICE_CORES_<hart>_<FIELD> of each,
 * and PROFILES of the compiler profiles. A core's profile is the index of
 * its profile in PROFILES, UINT32_MAX for a core without one. */
#ifndef BB_CORES_H
#define BB_CORES_H

#include <stdint.h>

#define X(name) bb_profile_##name,
enum { PROFILES bb_num_profiles };
#undef X
#define bb_profile_ UINT32_MAX
#define BB_PROFILE_INDEX_(name) bb_profile_##name
#define BB_PROFILE_INDEX(name) BB_PROFILE_INDEX_(name)

typedef struct {
  uint32_t hart, profile;
  uint64_t signature;
} bb_core_instance;

#define X(hart)                                                                \
  {DEVICE_CORES_##hart##_HART_ID,                                              \
   BB_PROFILE_INDEX(DEVICE_CORES_##hart##_PROFILE),                            \
   DEVICE_CORES_##hart##_SIGNATURE},
static const bb_core_instance bb_cores[] = {DEVICE_CORES};
#undef X
#define BB_NUM_CORES (sizeof(bb_cores) / sizeof(bb_cores[0]))

static inline uint32_t bb_hart_slot(uint32_t hart) {
  for (uint32_t i = 0; i < BB_NUM_CORES; ++i)
    if (bb_cores[i].hart == hart)
      return i;
  __builtin_trap();
}

static inline uint32_t bb_profile_core_count(uint32_t profile) {
  uint32_t count = 0;
  for (uint32_t i = 0; i < BB_NUM_CORES; ++i)
    count += bb_cores[i].profile == profile;
  return count;
}

static inline uint32_t bb_hart_profile(uint32_t hart) {
  for (uint32_t i = 0; i < BB_NUM_CORES; ++i)
    if (bb_cores[i].hart == hart)
      return bb_cores[i].profile;
  return UINT32_MAX;
}

#endif
