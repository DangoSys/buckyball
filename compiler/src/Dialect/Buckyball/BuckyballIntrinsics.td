//===-- BuckyballIntrinsics.td - Intrinsic operations ----*- tablegen -*-===//
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
// This file defines LLVM intrinsic wrapper operations for Buckyball.
// These operations map directly to RISC-V custom instructions via LLVM intrinsics.
//
//===----------------------------------------------------------------------===//

#ifndef BUCKYBALL_INTRINSICS
#define BUCKYBALL_INTRINSICS

//===----------------------------------------------------------------------===//
// Base class for intrinsic operations
//===----------------------------------------------------------------------===//

class Buckyball_IntrOpBase<string mnemonic, list<Trait> traits = []> :
  LLVM_IntrOpBase</*Dialect dialect=*/Buckyball_Dialect,
                  /*string opName=*/"intr." # mnemonic,
                  /*string enumName=*/"riscv_bb_" # !subst(".", "_", mnemonic),
                  /*list<int> overloadedResults=*/[],
                  /*list<int> overloadedOperands=*/[],
                  /*list<Trait> traits=*/traits,
                  /*int numResults=*/0>;

class Buckyball_CustomIntrOpBase<string mnemonic, string enumName,
                                 list<int> immArgPositions,
                                 list<string> immArgAttrNames,
                                 list<Trait> traits = []> :
  LLVM_IntrOpBase</*Dialect dialect=*/Buckyball_Dialect,
                  /*string opName=*/"intr." # mnemonic,
                  /*string enumName=*/enumName,
                  /*list<int> overloadedResults=*/[],
                  /*list<int> overloadedOperands=*/[],
                  /*list<Trait> traits=*/traits,
                  /*int numResults=*/0,
                  /*bit requiresAccessGroup=*/0,
                  /*bit requiresAliasAnalysis=*/0,
                  /*bit requiresFastmath=*/0,
                  /*bit requiresArgAndResultAttrs=*/0,
                  /*bit requiresOpBundles=*/0,
                  /*list<int> immArgPositions=*/immArgPositions,
                  /*list<string> immArgAttrNames=*/immArgAttrNames>;

//===----------------------------------------------------------------------===//
// Common intrinsic operations
//===----------------------------------------------------------------------===//

def MsetIntrOp : Buckyball_IntrOpBase<"mset">,
  Arguments<(ins LLVM_Type, LLVM_Type)>;

def FenceIntrOp : Buckyball_IntrOpBase<"fence">,
  Arguments<(ins LLVM_Type, LLVM_Type)>;

def MvinIntrOp : Buckyball_IntrOpBase<"mvin">,
  Arguments<(ins LLVM_Type, LLVM_Type)>;

def MvoutIntrOp : Buckyball_IntrOpBase<"mvout">,
  Arguments<(ins LLVM_Type, LLVM_Type)>;

def CustomIntrOp
    : Buckyball_CustomIntrOpBase<"custom", "riscv_buckyball_custom",
                                 [2], ["funct7"]>,
      Arguments<(ins LLVM_Type:$rs1, LLVM_Type:$rs2, I32Attr:$funct7)> {
  let assemblyFormat = "$rs1 `,` $rs2 attr-dict `:` type($rs1) `,` type($rs2)";
}

#endif // BUCKYBALL_INTRINSICS
