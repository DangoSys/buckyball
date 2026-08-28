#ifndef _BB_FP2INT_H_
#define _BB_FP2INT_H_

#include <bbhw/isa/bb_func7.h>
#include <bbhw/isa/isa.h>

// bb_fp2int(bank_id, wr_bank_id, iter, act_scale_addr)
// The Ball scans the complete logical FP32 activation tensor, writes
// Da=maxAbs/127 to this absolute MMIO byte address, then emits packed INT8.
// Encoding: rs1 = banks | iter
//           rs2[12:0] = act_scale_addr
#define bb_fp2int(bank_id, wr_bank_id, iter, act_scale_addr)                   \
  BUCKYBALL_INSTRUCTION_R_R(                                                   \
      (BB_BANK0(bank_id) | BB_BANK2(wr_bank_id) | BB_ITER(iter)),              \
      (FIELD((uint64_t)(act_scale_addr), 0, 12)), BB_FUNC7(FP2INT))

#endif // _BB_FP2INT_H_
