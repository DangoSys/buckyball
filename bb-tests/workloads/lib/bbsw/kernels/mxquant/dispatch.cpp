#include "../mxfp8/quant.h"
#include <bbhw/isa/isa.h>
#include <isa/mxquant.h>

extern "C" void mxfp8_quant(const uint32_t *input, uint8_t *output,
                            uint32_t count) {
  constexpr unsigned inputBank = 0, outputBank = 1;
  bb_mem_alloc(inputBank, 1, 1);
  bb_mem_alloc(outputBank, 1, 1);
  bb_mvin((uintptr_t)input, inputBank, count / 4, 1);
  bb_mxquant(inputBank, outputBank, count);
  bb_mvout((uintptr_t)output, outputBank, (count + count / 32 + 15) / 16, 1);
  bb_fence();
  bb_mem_release(inputBank);
  bb_mem_release(outputBank);
}
