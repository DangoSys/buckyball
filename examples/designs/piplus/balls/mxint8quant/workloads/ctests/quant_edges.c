#include "task.h"
#include "vectors.h"
#include <bbhw/isa/isa.h>
#include <isa/mxint8quant.h>

/* Thread program: quantizes the quant_edges rows on the Ball. */
uint64_t entry(const struct ball_task *task) {
  bb_mem_alloc(1, 1, 1);
  bb_mem_alloc(2, 1, 1);
  bb_mvin(task->input, 1,
          QUANT_EDGES_IN_BASE + QUANT_EDGES_ROWS * QUANT_EDGES_K / 4, 1);
  bb_mxint8quant(1, 2, QUANT_EDGES_ROWS, QUANT_EDGES_K, QUANT_EDGES_IN_BASE,
                 QUANT_EDGES_OUT_BASE);
  bb_mvout(task->output, 2,
           QUANT_EDGES_OUT_BASE + QUANT_EDGES_ROWS * MXINT8_ROWS(QUANT_EDGES_K),
           1);
  bb_mem_release(1);
  bb_mem_release(2);
  return 0;
}
