// Chip integration: MobileNet tile0-like bank SSA chain
// (fp2int col=4 -> im2col k=1 -> mvout). Not a Ball unit test.

func.func private @check_result(memref<16x16xi32>) -> ()

memref.global "private" constant @fp_in : memref<1x4xf32> =
    dense<[[1.0, 2.0, 3.0, 4.0]]>

func.func @main() -> i8 {
  %zero_i8 = arith.constant 0 : i8
  %depth_in = arith.constant 1 : i64
  %depth_out = arith.constant 16 : i64
  %stride = arith.constant 1 : i64
  %iter = arith.constant 1 : i64
  %scale = arith.constant 1065353216 : i64
  %ksize = arith.constant 1 : i64
  %pad = arith.constant 0 : i64

  %input = memref.get_global @fp_in : memref<1x4xf32>
  %output = memref.alloc() {alignment = 64 : i64} : memref<16x16xi32>

  %bin = buckyball.bank_alloc
  %bq = buckyball.bank_alloc
  %bout = buckyball.bank_alloc
  %loaded = buckyball.bank_mvin %input %bin %depth_in %stride
      : memref<1x4xf32> i64 i64 i64
  %q = buckyball.bank_fp2int %loaded %bq %iter %scale
      : i64 i64 i64 i64
  %im = buckyball.bank_im2col %q %bout %iter %ksize %stride %pad
      : i64 i64 i64 i64 i64 i64
  %stored = buckyball.bank_mvout %output %im %depth_out %stride
      : memref<16x16xi32> i64 i64 i64
  buckyball.bank_release %loaded : i64
  buckyball.bank_release %q : i64
  buckyball.bank_release %stored : i64

  func.call @check_result(%output) : (memref<16x16xi32>) -> ()
  memref.dealloc %output : memref<16x16xi32>
  return %zero_i8 : i8
}
