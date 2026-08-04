// Already-lowered ball Op: buckyball.fp2int (physical-bank form).

func.func private @check_result(memref<2x16xi8>) -> ()

memref.global "private" constant @fp_in : memref<2x16xf32> = dense<[
  [0.125, -0.125, 0.25, -0.25, 0.75, -0.75, 1.25, -1.25,
   1.75, -1.75, 63.25, 63.75, -63.75, -64.75, 0.0, -0.0],
  [2.25, -2.25, 2.75, -2.75, 3.25, -3.25, 3.75, -3.75,
   10.125, -10.125, 20.25, -20.25, 0.375, -0.375, 64.25, -65.25]
]>

func.func @main() -> i8 {
  %zero_i8 = arith.constant 0 : i8
  %depth = arith.constant 2 : i64
  %stride = arith.constant 1 : i64
  %scale = arith.constant 1073741824 : i64

  %input = memref.get_global @fp_in : memref<2x16xf32>
  %output = memref.alloc() {alignment = 64 : i64} : memref<2x16xi8>

  %bin = buckyball.bank_alloc {col = 4 : i64}
  %bout = buckyball.bank_alloc
  %loaded = buckyball.bank_mvin %input %bin %depth %stride
      : memref<2x16xf32> i64 i64 i64
  buckyball.fp2int %loaded, %bout, %depth, %scale : i64
  %stored = buckyball.bank_mvout %output %bout %depth %stride
      : memref<2x16xi8> i64 i64 i64
  buckyball.fence
  buckyball.bank_release %loaded : i64
  buckyball.bank_release %stored : i64

  func.call @check_result(%output) : (memref<2x16xi8>) -> ()
  memref.dealloc %output : memref<2x16xi8>
  return %zero_i8 : i8
}
