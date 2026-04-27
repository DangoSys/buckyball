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

class Buckyball_IntrOp<string mnemonic, list<Trait> traits = []> :
  Buckyball_Op<mnemonic, traits> {}

//===----------------------------------------------------------------------===//
// Common intrinsic operations
//===----------------------------------------------------------------------===//

def MsetIntrOp : Buckyball_IntrOp<"mset_intr"> {
  let summary = "LLVM intrinsic for bb_mset instruction";
  let description = [{
    Wrapper for int_riscv_bb_mset LLVM intrinsic.
    Maps to bb_mset custom instruction (funct7=32).
  }];
  let arguments = (ins I64:$rs1, I64:$rs2);
  let assemblyFormat = "$rs1 `,` $rs2 attr-dict";
}

def FenceIntrOp : Buckyball_IntrOp<"fence_intr"> {
  let summary = "LLVM intrinsic for bb_fence instruction";
  let description = [{
    Wrapper for int_riscv_bb_fence LLVM intrinsic.
    Maps to bb_fence custom instruction (funct7=0).
  }];
  let arguments = (ins);
  let assemblyFormat = "attr-dict";
}

def MvinIntrOp : Buckyball_IntrOp<"mvin_intr"> {
  let summary = "LLVM intrinsic for bb_mvin instruction";
  let description = [{
    Wrapper for int_riscv_bb_mvin LLVM intrinsic.
    Maps to bb_mvin custom instruction (funct7=33).
  }];
  let arguments = (ins I64:$rs1, I64:$rs2);
  let assemblyFormat = "$rs1 `,` $rs2 attr-dict";
}

def MvoutIntrOp : Buckyball_IntrOp<"mvout_intr"> {
  let summary = "LLVM intrinsic for bb_mvout instruction";
  let description = [{
    Wrapper for int_riscv_bb_mvout LLVM intrinsic.
    Maps to bb_mvout custom instruction (funct7=16).
  }];
  let arguments = (ins I64:$rs1, I64:$rs2);
  let assemblyFormat = "$rs1 `,` $rs2 attr-dict";
}

def MulWarp16IntrOp : Buckyball_IntrOp<"mul_warp16_intr"> {
  let summary = "LLVM intrinsic for bb_mul_warp16 instruction";
  let description = [{
    Wrapper for int_riscv_bb_mul_warp16 LLVM intrinsic.
    Maps to bb_mul_warp16 custom instruction.
  }];
  let arguments = (ins I64:$rs1, I64:$rs2);
  let assemblyFormat = "$rs1 `,` $rs2 attr-dict";
}

def TransposeIntrOp : Buckyball_IntrOp<"transpose_intr"> {
  let summary = "LLVM intrinsic for bb_transpose instruction";
  let description = [{
    Wrapper for int_riscv_bb_transpose LLVM intrinsic.
    Maps to bb_transpose custom instruction.
  }];
  let arguments = (ins I64:$rs1, I64:$rs2);
  let assemblyFormat = "$rs1 `,` $rs2 attr-dict";
}

def Im2colIntrOp : Buckyball_IntrOp<"im2col_intr"> {
  let summary = "LLVM intrinsic for bb_im2col instruction";
  let description = [{
    Wrapper for int_riscv_bb_im2col LLVM intrinsic.
    Maps to bb_im2col custom instruction.
  }];
  let arguments = (ins I64:$rs1, I64:$rs2);
  let assemblyFormat = "$rs1 `,` $rs2 attr-dict";
}

def QuantIntrOp : Buckyball_IntrOp<"quant_intr"> {
  let summary = "LLVM intrinsic for bb_quant instruction";
  let description = [{
    Wrapper for int_riscv_bb_quant LLVM intrinsic.
    Maps to bb_quant custom instruction.
  }];
  let arguments = (ins I64:$rs1, I64:$rs2);
  let assemblyFormat = "$rs1 `,` $rs2 attr-dict";
}

def DequantIntrOp : Buckyball_IntrOp<"dequant_intr"> {
  let summary = "LLVM intrinsic for bb_dequant instruction";
  let description = [{
    Wrapper for int_riscv_bb_dequant LLVM intrinsic.
    Maps to bb_dequant custom instruction.
  }];
  let arguments = (ins I64:$rs1, I64:$rs2);
  let assemblyFormat = "$rs1 `,` $rs2 attr-dict";
}

def ReluIntrOp : Buckyball_IntrOp<"relu_intr"> {
  let summary = "LLVM intrinsic for bb_relu instruction";
  let description = [{
    Wrapper for int_riscv_bb_relu LLVM intrinsic.
    Maps to bb_relu custom instruction.
  }];
  let arguments = (ins I64:$rs1, I64:$rs2);
  let assemblyFormat = "$rs1 `,` $rs2 attr-dict";
}

#endif // BUCKYBALL_INTRINSICS
