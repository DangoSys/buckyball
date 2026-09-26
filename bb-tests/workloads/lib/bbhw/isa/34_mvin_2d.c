#ifndef _BB_MVIN_2D_H_
#define _BB_MVIN_2D_H_

#include "isa.h"

#define BB_MVIN_2D_FUNC7 34

// rs1: write bank in BANK2 and tile height; [19:0] reserved.
// rs2: [35:0] address / 8, [42:36] physical pixel bytes / 8,
//      [52:43] source width, [58:53] destination row base,
//      [61:59] tile width - 1, [62] valid bytes (0 = 16, 1 = 8).
#define bb_mvin_2d(mem_addr, bank_id, height, pixel_bytes, source_width,       \
                   dst_base, tile_width, valid_bytes)                          \
  do {                                                                         \
    const uintptr_t bb_mvin_2d_addr = (uintptr_t)(mem_addr);                   \
    const uint64_t bb_mvin_2d_valid = (valid_bytes);                           \
    if ((bb_mvin_2d_addr & 7) || (bb_mvin_2d_addr >> 39) ||                    \
        (bb_mvin_2d_valid != 8 && bb_mvin_2d_valid != 16))                     \
      __builtin_trap();                                                        \
    bb_dma_cache_flush();                                                      \
    BUCKYBALL_INSTRUCTION_R_R(                                                 \
        (BB_BANK2(bank_id) | BB_ITER(height)),                                 \
        (FIELD(bb_mvin_2d_addr >> 3, 0, 35) |                                  \
         FIELD((pixel_bytes) / 8, 36, 42) | FIELD(source_width, 43, 52) |      \
         FIELD(dst_base, 53, 58) | FIELD((tile_width) - 1, 59, 61) |           \
         FIELD(bb_mvin_2d_valid == 8, 62, 62)),                                \
        BB_MVIN_2D_FUNC7);                                                     \
  } while (0)

#endif // _BB_MVIN_2D_H_
