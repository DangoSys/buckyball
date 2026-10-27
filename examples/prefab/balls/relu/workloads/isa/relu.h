#ifndef _BB_RELU_H_
#define _BB_RELU_H_

#include <bbhw/isa/bb_func7.h>
#include <bbhw/isa/isa.h>

#define bb_relu(bank_id, group, iter, stride)                                  \
  BUCKYBALL_INSTRUCTION_R_R((BB_BANK0(bank_id) | BB_BANK1(group) |             \
                             BB_BANK2(bank_id) | BB_ITER(iter) |               \
                             BB_UPDATE_WR),                                    \
                            (stride), FUNCT7(relu, RELU))

#endif // _BB_RELU_H_
