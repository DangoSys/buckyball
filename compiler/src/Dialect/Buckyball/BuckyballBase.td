//===-- BuckyballBase.td - Buckyball base definitions ----------*- tablegen -*-===//
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
// This file defines the Buckyball dialect and common base operations.
// Common operations include data movement instructions (mvin, mvout, mset, fence)
// that are used by all Ball implementations.
//
//===----------------------------------------------------------------------===//

#ifndef BUCKYBALL_BASE
#define BUCKYBALL_BASE

include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/PatternBase.td"

//===----------------------------------------------------------------------===//
// Buckyball dialect definition
//===----------------------------------------------------------------------===//

def Buckyball_Dialect : Dialect {
  let name = "buckyball";
  let cppNamespace = "::buddy::buckyball";
  let summary = "Dialect for RISC-V Buckyball NPU extension";
  let description = [{
    Buckyball is an NPU Design Framework with composable Ball units.
    Each Ball is a specialized compute unit that can be independently
    developed and combined with others.
  }];
}

//===----------------------------------------------------------------------===//
// Base operation class
//===----------------------------------------------------------------------===//

class Buckyball_Op<string mnemonic, list<Trait> traits = []> :
  Op<Buckyball_Dialect, mnemonic, traits> {}

//===----------------------------------------------------------------------===//
// Common data movement operations (used by all Balls)
//===----------------------------------------------------------------------===//

def MvinOp : Buckyball_Op<"mvin"> {
  let summary = "Load data from memory to scratchpad bank";
  let description = [{
    Loads data from DRAM to scratchpad bank.
    Corresponds to bb_mvin instruction (funct7=33).
  }];
  let arguments = (ins MemRefRankOf<[AnyType], [2]>:$input,
                       I64:$addr, I64:$depth, I64:$stride);
  let assemblyFormat = "$input $addr $depth $stride attr-dict `:` "
                       "type($input) type($addr) type($depth) type($stride)";
}

def MvoutOp : Buckyball_Op<"mvout"> {
  let summary = "Store data from scratchpad bank to memory";
  let description = [{
    Stores data from scratchpad bank to DRAM.
    Corresponds to bb_mvout instruction (funct7=16).
  }];
  let arguments = (ins MemRefRankOf<[AnyType], [2]>:$output,
                       I64:$addr, I64:$depth, I64:$stride);
  let assemblyFormat = "$output $addr $depth $stride attr-dict `:` "
                       "type($output) type($addr) type($depth) type($stride)";
}

def MsetOp : Buckyball_Op<"mset"> {
  let summary = "Allocate or release scratchpad bank memory";
  let description = [{
    Allocates or releases scratchpad bank memory.
    Corresponds to bb_mset instruction (funct7=32).
  }];
  let arguments = (ins I64:$bankId,
                       DefaultValuedAttr<BoolAttr, "true">:$alloc,
                       DefaultValuedAttr<I64Attr, "1">:$row,
                       DefaultValuedAttr<I64Attr, "1">:$col);
  let assemblyFormat = "$bankId attr-dict `:` type($bankId)";
}

def FenceOp : Buckyball_Op<"fence"> {
  let summary = "Memory fence for synchronization";
  let description = [{
    Ensures all previous memory operations complete before proceeding.
    Corresponds to bb_fence instruction (funct7=0).
  }];
  let arguments = (ins);
  let assemblyFormat = "attr-dict";
}

#endif // BUCKYBALL_BASE
