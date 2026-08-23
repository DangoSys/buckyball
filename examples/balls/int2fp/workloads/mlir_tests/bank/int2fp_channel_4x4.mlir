func.func private @prepare_scales() -> ()
func.func private @check_result(memref<1x16xf32>) -> ()
memref.global "private" constant @input : memref<1x16xi32> = dense<8>

func.func @main() -> i8 {
  %z = arith.constant 0 : i8
  %one = arith.constant 1 : i64
  %zero = arith.constant 0 : i64
  %sixteen = arith.constant 16 : i64
  func.call @prepare_scales() : () -> ()
  %input = memref.get_global @input : memref<1x16xi32>
  %output = memref.alloc() {alignment = 64 : i64} : memref<1x16xf32>
  %in = buckyball.bank_alloc {col = 4 : i64}
  %out = buckyball.bank_alloc {col = 4 : i64}
  %loaded = buckyball.bank_mvin %input %in %one %one : memref<1x16xi32> i64 i64 i64
  %q = buckyball.bank_int2fp_channel %loaded %out %one %zero %sixteen : i64 i64 i64 i64 i64
  %stored = buckyball.bank_mvout %output %q %one %one : memref<1x16xf32> i64 i64 i64
  buckyball.bank_release %loaded : i64
  buckyball.bank_release %stored : i64
  func.call @check_result(%output) : (memref<1x16xf32>) -> ()
  memref.dealloc %output : memref<1x16xf32>
  return %z : i8
}
