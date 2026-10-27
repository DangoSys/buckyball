#ifndef _BB_IM2COL_H_
#define _BB_IM2COL_H_

#include <bbhw/isa/bb_func7.h>
#include <bbhw/isa/isa.h>

// Input virtual bank: group g is channel block g (channels 16g..16g+15);
// line input_base + y * input_size + x holds pixel (y, x) of that block.
// K is ordered (kernel row, kernel column, channel), so K tile
// t = (kr * kernel + kc) * input_groups + g is one input row (zero when the
// tap falls into padding). With K_TILES = kernel * kernel * input_groups, the
// output groups must divide K_TILES; output group j holds K tiles
// j * G .. j * G + G - 1 (G = K_TILES / output_groups) in the SMatMul A
// layout: window w, local tile kt -> line ((w / tile_rows) * G + kt) *
// tile_rows + w % tile_rows. Windows window_count .. up to the next multiple
// of tile_rows are zero. Window w is output pixel window_start + w in
// row-major order.
#define bb_im2col(input_bank, output_bank, input_size, kernel, stride,         \
                  padding, input_base, start_row, start_col, window_start,     \
                  window_count, tile_rows)                                     \
  BUCKYBALL_INSTRUCTION_R_R(                                                   \
      BB_BANK0(input_bank) | BB_BANK2(output_bank) | BB_ITER(input_size),      \
      FIELD(kernel, 0, 3) | FIELD(stride, 4, 7) | FIELD(padding, 8, 11) |      \
          FIELD(start_col, 12, 15) | FIELD(start_row, 16, 19) |                \
          FIELD(input_base, 20, 25) | FIELD(window_start, 26, 35) |            \
          FIELD(window_count, 36, 45) | FIELD(tile_rows, 46, 50),              \
      FUNCT7(im2col, IM2COL))

#endif // _BB_IM2COL_H_
