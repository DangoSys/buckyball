#include "isa.h"

// FLUSH指令实现
void bb_flush(void) {
  BuckyballInstruction inst = {0, 0};
  InstructionBuilder builder = create_builder(&inst, FLUSH_FUNC7);
  execute_builder(builder);
}
