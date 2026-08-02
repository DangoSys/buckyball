// Bank-level Op: buckyball.bank_int2fp
// Lowering: assign-physical-banks -> bank_int2fp becomes int2fp.

func.func private @check_result(memref<4x4xf32>) -> ()

memref.global "private" constant @i32_in : memref<4x4xi32> =
    dense<[[1, 2, 3, -1],
           [-2, 0, 4, 5],
           [10, -10, 7, 100],
           [-100, 8, 16, -8]]>

func.func @main() -> i8 {
  %zero_i8 = arith.constant 0 : i8
  %depth = arith.constant 4 : i64
  %stride = arith.constant 1 : i64
  %iter = arith.constant 4 : i64
  %scale = arith.constant 1065353216 : i64

  %input = memref.get_global @i32_in : memref<4x4xi32>
  %output = memref.alloc() {alignment = 64 : i64} : memref<4x4xf32>

  %bin = buckyball.bank_alloc
  %bout = buckyball.bank_alloc
  %loaded = buckyball.bank_mvin %input %bin %depth %stride
      : memref<4x4xi32> i64 i64 i64
  %q = buckyball.bank_int2fp %loaded %bout %iter %scale
      : i64 i64 i64 i64
  %stored = buckyball.bank_mvout %output %q %depth %stride
      : memref<4x4xf32> i64 i64 i64
  buckyball.bank_release %loaded : i64
  buckyball.bank_release %stored : i64

  func.call @check_result(%output) : (memref<4x4xf32>) -> ()
  memref.dealloc %output : memref<4x4xf32>
  return %zero_i8 : i8
}
