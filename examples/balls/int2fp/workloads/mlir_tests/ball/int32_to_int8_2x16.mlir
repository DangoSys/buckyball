func.func private @check_result(memref<2x16xi8>) -> ()

memref.global "private" constant @input_i32 : memref<2x16xi32> = dense<[
  [-1000, -257, -255, -5, -3, -1, 0, 1, 3, 5, 127, 253, 255, 257, 1000, 2],
  [-999, -511, -259, -9, -7, -3, 2, 4, 6, 9, 125, 251, 254, 258, 511, 999]
]>

func.func @main() -> i8 {
  %zero_i8 = arith.constant 0 : i8
  %depth = arith.constant 2 : i64
  %stride = arith.constant 1 : i64
  %scale = arith.constant 1056964608 : i64
  %input = memref.get_global @input_i32 : memref<2x16xi32>
  %output = memref.alloc() {alignment = 64 : i64} : memref<2x16xi8>

  %in = buckyball.bank_alloc {col = 4 : i64}
  %out = buckyball.bank_alloc
  %loaded = buckyball.bank_mvin %input %in %depth %stride
      : memref<2x16xi32> i64 i64 i64
  buckyball.int32_to_int8 %loaded, %out, %depth, %scale : i64
  %stored = buckyball.bank_mvout %output %out %depth %stride
      : memref<2x16xi8> i64 i64 i64
  buckyball.fence
  buckyball.bank_release %loaded : i64
  buckyball.bank_release %stored : i64

  func.call @check_result(%output) : (memref<2x16xi8>) -> ()
  memref.dealloc %output : memref<2x16xi8>
  return %zero_i8 : i8
}
