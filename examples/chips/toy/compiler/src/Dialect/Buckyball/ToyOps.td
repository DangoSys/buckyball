//===-- ToyOps.td - Toy Buckyball operations ---------------- tablegen --===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
// This file defines Toy compiler operations for the Buckyball dialect.
//
//===----------------------------------------------------------------------===//

#ifndef TOY_BUCKYBALL_OPS_EXTRA
#define TOY_BUCKYBALL_OPS_EXTRA

// Hardware-level matrix multiplication operation
def MatMulOp : Buckyball_Op<"matmul"> {
  let summary = "Hardware-level matrix multiplication";
  let description = [{
    Perform fixed-size matrix multiplication that fits in accelerator resources.
    This operation represents a complete matrix multiplication tile that:
    - Fits within the scratchpad and accumulator constraints
    - Performs data movement (mvin) for input tiles
    - Executes computation using warp-level parallelism
    - Writes back results (mvout) to main memory
  }];
  let arguments = (ins MemRefRankOf<[AnyType], [2]>:$aMemArray,
                       MemRefRankOf<[AnyType], [2]>:$bMemArray,
                       MemRefRankOf<[AnyType], [2]>:$cMemArray);
  let assemblyFormat = [{$aMemArray $bMemArray $cMemArray attr-dict `:`
    type($aMemArray) type($bMemArray) type($cMemArray)}];
}

#endif // TOY_BUCKYBALL_OPS_EXTRA
