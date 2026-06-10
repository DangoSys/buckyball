#ifndef _MEM_H_
#define _MEM_H_

#include "params.h"
#include <bbhw/isa/isa.h>

static inline void bb_boot_init(void) {
  for (int bank_id = 0; bank_id < BANK_NUM; ++bank_id) {
    bb_mem_release(bank_id);
    bb_mmio_set(bank_id, 0, 0);
  }
  bb_fence();
}

#endif // _MEM_H_
