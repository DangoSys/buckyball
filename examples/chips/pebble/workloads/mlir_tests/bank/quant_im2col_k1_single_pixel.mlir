// Chip integration: f32-to-i8 quantization -> im2col k=1 -> mvout.

func.func private @check_result(memref<16x16xi8>) -> ()

memref.global "private" constant @fp_in : memref<1x16xf32> =
    dense<[[1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]>

func.func @main() -> i8 {
  %zero_i8 = arith.constant 0 : i8
  %depth_in = arith.constant 1 : i64
  %depth_out = arith.constant 16 : i64
  %stride = arith.constant 1 : i64
  %quant_iter = arith.constant 4 : i64
  %input_size = arith.constant 1 : i64
  %scale = arith.constant 1.0 : f32
  %da_addr = arith.constant 0 : i64
  %ksize = arith.constant 1 : i64
  %pad = arith.constant 0 : i64

  %input = memref.get_global @fp_in : memref<1x16xf32>
  %output = memref.alloc() alignment = 64 : memref<16x16xi8>

  %bin = buckyball.bank_alloc {col = 4 : i64}
  %bq = buckyball.bank_alloc
  %bout = buckyball.bank_alloc
  %loaded = buckyball.bank_mvin %input %bin %depth_in %stride
      : memref<1x16xf32> i64 i64 i64
  %q = buckyball.bank_quant_f32_to_i8 %loaded %bq %quant_iter %scale
      : i64 i64 i64 f32
  %base = arith.constant 0 : i64
  %lane = arith.constant 0 : i64
  %im = buckyball.bank_im2col %q %bout %input_size %ksize %stride %pad %base
      %lane {startRow = 0 : i64, startCol = 0 : i64, windowStart = 0 : i64,
             windowCount = 1 : i64}
      : i64 i64 i64 i64 i64 i64 i64 i64
  %stored = buckyball.bank_mvout %output %im %depth_out %stride
      : memref<16x16xi8> i64 i64 i64
  buckyball.bank_release %loaded : i64
  buckyball.bank_release %q : i64
  buckyball.bank_release %stored : i64

  func.call @check_result(%output) : (memref<16x16xi8>) -> ()
  memref.dealloc %output : memref<16x16xi8>
  return %zero_i8 : i8
}
