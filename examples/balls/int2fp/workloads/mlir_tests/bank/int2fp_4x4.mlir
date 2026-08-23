// Bank-level Op: buckyball.bank_int2fp_tensor
// Lowering: assign-physical-banks -> tensor int2fp.

func.func private @prepare_scales() -> ()
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
  %da_addr = arith.constant 0 : i64
  %dw_addr = arith.constant 16 : i64

  func.call @prepare_scales() : () -> ()

  %input = memref.get_global @i32_in : memref<4x4xi32>
  %output = memref.alloc() {alignment = 64 : i64} : memref<4x4xf32>

  %bin = buckyball.bank_alloc
  %bout = buckyball.bank_alloc
  %loaded = buckyball.bank_mvin %input %bin %depth %stride
      : memref<4x4xi32> i64 i64 i64
  %q = buckyball.bank_int2fp_tensor %loaded %bout %iter %da_addr %dw_addr
      : i64 i64 i64 i64 i64
  %stored = buckyball.bank_mvout %output %q %depth %stride
      : memref<4x4xf32> i64 i64 i64
  buckyball.bank_release %loaded : i64
  buckyball.bank_release %stored : i64

  func.call @check_result(%output) : (memref<4x4xf32>) -> ()
  memref.dealloc %output : memref<4x4xf32>
  return %zero_i8 : i8
}
