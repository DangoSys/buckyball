#ifndef _BB_RELU_H_
#define _BB_RELU_H_

#include <bbhw/isa/bb_func7.h>
#include <bbhw/isa/isa.h>

#define bb_relu(bank_id, wr_bank_id, iter)                                     \
  BUCKYBALL_INSTRUCTION_R_R(                                                   \
      (BB_BANK0(bank_id) | BB_BANK2(wr_bank_id) | BB_ITER(iter)), 0,           \
      BB_FUNC7(RELU))

#endif // _BB_RELU_H_
