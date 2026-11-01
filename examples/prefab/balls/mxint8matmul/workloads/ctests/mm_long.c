#include "task.h"
#include "vectors.h"
#include <bbhw/isa/isa.h>
#include <isa/mxint8matmul.h>

/* Thread program: the mm_long product on the Ball. */
uint64_t entry(const struct mxint8matmul_task *task) {
  bb_mem_alloc(1, 1, 1);
  bb_mem_alloc(2, 1, 1);
  bb_mem_alloc(3, 1, 1);
  bb_mvin(task->a, 1, MM_LONG_A_ROWS, 1);
  bb_mvin(task->b, 2, MM_LONG_B_ROWS, 1);
  if (MM_LONG_ACCUMULATE)
    bb_mvin(task->c, 3, MM_LONG_C_ROWS, 1);
  bb_mxint8mm(1, 2, 3, MM_LONG_M, MM_LONG_N, MM_LONG_K, MM_LONG_A_BASE,
              MM_LONG_B_BASE, MM_LONG_C_BASE, MM_LONG_ACCUMULATE);
  bb_mvout(task->c, 3, MM_LONG_C_ROWS, 1);
  bb_mem_release(1);
  bb_mem_release(2);
  bb_mem_release(3);
  return 0;
}
