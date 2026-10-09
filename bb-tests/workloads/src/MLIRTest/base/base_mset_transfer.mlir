// RUN: buddy-opt %s --target=toy -report-bank-usage="bank_num=4 verbose=true" 2>&1 | FileCheck %s --check-prefix=REPORT
// REPORT: transfer b1 -> b7 groups=1 cur=2/4
// REPORT: transfer b2 -> b7 groups=1 cur=2/4
// REPORT: peak=2/4 alloc=2 release=1 leaked=0

func.func private @check_result(memref<16x32xi8>)
func.func @main() -> i8 {
  %zero = arith.constant 0 : i8
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %i32_16 = arith.constant 16 : i32
  %i32_127 = arith.constant 127 : i32
  %source0 = arith.constant 1 : i64
  %source1 = arith.constant 2 : i64
  %target = arith.constant 7 : i64
  %depth = arith.constant 16 : i64
  %stride = arith.constant 1 : i64
  %input0 = memref.alloc() alignment = 64 : memref<16x16xi8>
  %input1 = memref.alloc() alignment = 64 : memref<16x16xi8>
  %output = memref.alloc() alignment = 64 : memref<16x32xi8>
  scf.for %i = %c0 to %c16 step %c1 {
    scf.for %j = %c0 to %c16 step %c1 {
      %ii = arith.index_cast %i : index to i32
      %jj = arith.index_cast %j : index to i32
      %base = arith.muli %ii, %i32_16 : i32
      %value = arith.addi %base, %jj : i32
      %other = arith.subi %i32_127, %value : i32
      %byte = arith.trunci %value : i32 to i8
      %other_byte = arith.trunci %other : i32 to i8
      memref.store %byte, %input0[%i, %j] : memref<16x16xi8>
      memref.store %other_byte, %input1[%i, %j] : memref<16x16xi8>
    }
  }
  buckyball.mset %source0 <row = 1, col = 1> : i64
  buckyball.mset %source1 <row = 1, col = 1> : i64
  buckyball.mvin %input0 %source0 %depth %stride : memref<16x16xi8> i64 i64 i64
  buckyball.mvin %input1 %source1 %depth %stride : memref<16x16xi8> i64 i64 i64
  buckyball.mset_transfer %source0 %target : i64 i64
  buckyball.mset_transfer %source1 %target : i64 i64
  buckyball.mvout %output %target %depth %stride : memref<16x32xi8> i64 i64 i64
  buckyball.mset %target <alloc = false, row = 0, col = 0> : i64
  func.call @check_result(%output) : (memref<16x32xi8>) -> ()
  memref.dealloc %input0 : memref<16x16xi8>
  memref.dealloc %input1 : memref<16x16xi8>
  memref.dealloc %output : memref<16x32xi8>
  return %zero : i8
}
