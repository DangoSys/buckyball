#include "task.h"
#include "vectors.h"
#include <bbhw/isa/isa.h>
#include <isa/mxint8mm.h>

/* Thread program: the mm_accumulate product on the Ball. */
uint64_t entry(const struct ball_task *task) {
  bb_mem_alloc(1, 1, 1);
  bb_mem_alloc(2, 1, 1);
  bb_mem_alloc(3, 1, 1);
  bb_mvin(task->a, 1,
          MM_ACCUMULATE_A_BASE + MM_ACCUMULATE_M * MXINT8_ROWS(MM_ACCUMULATE_K),
          1);
  bb_mvin(task->b, 2,
          MM_ACCUMULATE_B_BASE + MM_ACCUMULATE_N * MXINT8_ROWS(MM_ACCUMULATE_K),
          1);
  if (MM_ACCUMULATE_ACCUMULATE)
    bb_mvin(task->c, 3,
            MM_ACCUMULATE_C_BASE + MM_ACCUMULATE_M * MM_ACCUMULATE_N / 4, 1);
  bb_mxint8mm(1, 2, 3, MM_ACCUMULATE_M, MM_ACCUMULATE_N, MM_ACCUMULATE_K,
              MM_ACCUMULATE_A_BASE, MM_ACCUMULATE_B_BASE, MM_ACCUMULATE_C_BASE,
              MM_ACCUMULATE_ACCUMULATE);
  bb_mvout(task->c, 3,
           MM_ACCUMULATE_C_BASE + MM_ACCUMULATE_M * MM_ACCUMULATE_N / 4, 1);
  bb_mem_release(1);
  bb_mem_release(2);
  bb_mem_release(3);
  return 0;
}
