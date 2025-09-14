#include "isa.h"
// FENCE指令实现
void bb_fence(void) {
  BuckyballInstruction inst = {0, 0};
  InstructionBuilder builder = create_builder(&inst, FENCE_FUNC7);
  execute_builder(builder);
}
