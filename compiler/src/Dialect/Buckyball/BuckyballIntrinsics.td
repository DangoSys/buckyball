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

def MulWarp16IntrOp : Buckyball_IntrOpBase<"mul.warp16">,
  Arguments<(ins LLVM_Type, LLVM_Type)>;

def TransposeIntrOp : Buckyball_IntrOpBase<"transpose">,
  Arguments<(ins LLVM_Type, LLVM_Type)>;

def Im2colIntrOp : Buckyball_IntrOpBase<"im2col">,
  Arguments<(ins LLVM_Type, LLVM_Type)>;

def QuantIntrOp : Buckyball_IntrOpBase<"quant">,
  Arguments<(ins LLVM_Type, LLVM_Type)>;

def DequantIntrOp : Buckyball_IntrOpBase<"dequant">,
  Arguments<(ins LLVM_Type, LLVM_Type)>;

def ReluIntrOp : Buckyball_IntrOpBase<"relu">,
  Arguments<(ins LLVM_Type, LLVM_Type)>;

def SystolicIntrOp : Buckyball_IntrOpBase<"bbfp.mul">,
  Arguments<(ins LLVM_Type, LLVM_Type)>;

#endif // BUCKYBALL_INTRINSICS
