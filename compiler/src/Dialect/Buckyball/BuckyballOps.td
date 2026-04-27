//===-- BuckyballOps.td - Additional Buckyball operations ---- tablegen --===//
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
// This file defines additional operations for the Buckyball dialect including
// MatMul, debugging, and profiling operations.
//
//===----------------------------------------------------------------------===//

#ifndef BUCKYBALL_ADDITIONAL_OPS
#define BUCKYBALL_ADDITIONAL_OPS

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

// Debugging and profiling operations
def PrintMemRefOp : Buckyball_Op<"print_memref"> {
  let summary = "Print memref contents for debugging";
  let description = [{
    Print the contents of a memref for debugging purposes.
  }];
  let arguments = (ins AnyMemRef:$input);
  let assemblyFormat = "$input attr-dict `:` type($input)";
}

def PrintScalarOp : Buckyball_Op<"print_scalar"> {
  let summary = "Print scalar value for debugging";
  let description = [{
    Print a scalar value for debugging purposes.
  }];
  let arguments = (ins AnyType:$value);
  let assemblyFormat = "$value attr-dict `:` type($value)";
}

def CounterStartOp : Buckyball_Op<"counter_start"> {
  let summary = "Start performance counter";
  let description = [{
    Start a performance counter for profiling.
  }];
  let arguments = (ins);
  let assemblyFormat = "attr-dict";
}

def CounterStopOp : Buckyball_Op<"counter_stop"> {
  let summary = "Stop performance counter";
  let description = [{
    Stop a performance counter and report results.
  }];
  let arguments = (ins);
  let assemblyFormat = "attr-dict";
}

#endif // BUCKYBALL_ADDITIONAL_OPS
