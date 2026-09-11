#ifndef _BB_MVIN_2D_H_
#define _BB_MVIN_2D_H_

#include "isa.h"

#define BB_MVIN_2D_FUNC7 34

// rs1: bank id and tile height.
// rs2: [31:0] address, [38:32] physical pixel bytes / 8,
//      [48:39] source width, [54:49] destination row base,
//      [57:55] tile width - 1, [61:58] valid bytes (0 encodes 16).
#define bb_mvin_2d(mem_addr, bank_id, height, pixel_bytes, source_width,       \
                   dst_base, tile_width, valid_bytes)                          \
  do {                                                                         \
    bb_dma_cache_flush();                                                      \
    BUCKYBALL_INSTRUCTION_R_R(                                                 \
        (BB_BANK0(bank_id) | BB_ITER(height)),                                 \
        (FIELD(mem_addr, 0, 31) | FIELD((pixel_bytes) / 8, 32, 38) |           \
         FIELD(source_width, 39, 48) | FIELD(dst_base, 49, 54) |               \
         FIELD((tile_width) - 1, 55, 57) |                                     \
         FIELD((valid_bytes) == 16 ? 0 : (valid_bytes), 58, 61)),              \
        BB_MVIN_2D_FUNC7);                                                     \
  } while (0)

#endif // _BB_MVIN_2D_H_
