//===-- BuckyballBankSSA.td - Bank SSA operations ------------ tablegen --===//
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
// This file defines Bank SSA operations for the Buckyball dialect.
// These are intermediate representation operations used for bank allocation
// and optimization passes.
//
//===----------------------------------------------------------------------===//

#ifndef BUCKYBALL_BANK_SSA
#define BUCKYBALL_BANK_SSA

//===----------------------------------------------------------------------===//
// Bank SSA Operations
//===----------------------------------------------------------------------===//

def BankAllocOp : Buckyball_Op<"bank_alloc"> {
  let summary = "Allocate a virtual on-chip bank resource";
  let description = [{
    Create a virtual bank handle in SSA form.
    - row/col encode bank-array shape used by allocator.
    - The result i64 is a virtual bank handle, not physical id.
  }];
  let arguments = (ins
    DefaultValuedAttr<I64Attr, "1">:$row,
    DefaultValuedAttr<I64Attr, "1">:$col);
  let results = (outs I64:$bank);
  let assemblyFormat = "attr-dict";
}

def BankReleaseOp : Buckyball_Op<"bank_release"> {
  let summary = "Release a virtual bank handle";
  let arguments = (ins I64:$bank);
  let assemblyFormat = "$bank attr-dict `:` type($bank)";
}

def BankMvinOp : Buckyball_Op<"bank_mvin"> {
  let summary = "Load into a virtual bank handle";
  let arguments = (ins MemRefRankOf<[AnyType], [2]>:$input, I64:$bank, I64:$depth,
                       I64:$stride);
  let results = (outs I64:$bankOut);
  let assemblyFormat =
      "$input $bank $depth $stride attr-dict `:` type($input) type($bank) type($depth) type($stride)";
}

def BankMvoutOp : Buckyball_Op<"bank_mvout"> {
  let summary = "Store from a virtual bank handle";
  let arguments = (ins MemRefRankOf<[AnyType], [2]>:$output, I64:$bank, I64:$depth,
                       I64:$stride);
  let results = (outs I64:$bankOut);
  let assemblyFormat =
      "$output $bank $depth $stride attr-dict `:` type($output) type($bank) type($depth) type($stride)";
}

def BankMulWarp16Op : Buckyball_Op<"bank_mul_warp16"> {
  let summary = "Warp-16 matmul using virtual bank handles";
  let arguments = (ins I64:$op1Bank, I64:$op2Bank, I64:$wrBank, I64:$iter, I64:$mode);
  let results = (outs I64:$wrBankOut);
  let assemblyFormat =
      "$op1Bank $op2Bank $wrBank $iter $mode attr-dict `:` type($op1Bank) type($op2Bank) type($wrBank) type($iter) type($mode)";
}

def BankTransposeOp : Buckyball_Op<"bank_transpose"> {
  let summary = "Transpose using virtual bank handles";
  let arguments = (ins I64:$inBank, I64:$outBank, I64:$iter, I64:$mode);
  let results = (outs I64:$outBankOut);
  let assemblyFormat = "$inBank $outBank $iter $mode attr-dict `:` type($inBank) type($outBank) type($iter) type($mode)";
}

def BankIm2colOp : Buckyball_Op<"bank_im2col"> {
  let summary = "Im2col using virtual bank handles";
  let arguments = (ins I64:$inBank, I64:$outBank, I64:$krow, I64:$kcol,
                       I64:$inrow, I64:$incol, I64:$startrow, I64:$startcol);
  let results = (outs I64:$outBankOut);
  let assemblyFormat =
      "$inBank $outBank $krow $kcol $inrow $incol $startrow $startcol attr-dict `:` type($inBank) type($outBank) type($krow) type($kcol) type($inrow) type($incol) type($startrow) type($startcol)";
}

def BankQuantOp : Buckyball_Op<"bank_quant"> {
  let summary = "Quantization using virtual bank handles";
  let arguments = (ins I64:$inBank, I64:$outBank, I64:$iter, I64:$scale);
  let results = (outs I64:$outBankOut);
  let assemblyFormat =
      "$inBank $outBank $iter $scale attr-dict `:` type($inBank) type($outBank) type($iter) type($scale)";
}

def BankDequantOp : Buckyball_Op<"bank_dequant"> {
  let summary = "Dequantization using virtual bank handles";
  let arguments = (ins I64:$inBank, I64:$outBank, I64:$iter, I64:$scale);
  let results = (outs I64:$outBankOut);
  let assemblyFormat =
      "$inBank $outBank $iter $scale attr-dict `:` type($inBank) type($outBank) type($iter) type($scale)";
}

#endif // BUCKYBALL_BANK_SSA
