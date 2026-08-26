// Already-lowered ball Op: buckyball.fp2int (physical-bank form).

func.func private @check_result(memref<1x16xi8>) -> ()

memref.global "private" constant @fp_in : memref<1x16xf32> =
    dense<[[1.0, 2.0, 3.0, -1.0, -2.0, 0.0, 4.0, 5.0,
            10.0, -10.0, 0.5, 100.0, -100.0, 7.0, 8.0, -8.0]]>

func.func @main() -> i8 {
  %zero_i8 = arith.constant 0 : i8
  %depth = arith.constant 1 : i64
  %stride = arith.constant 1 : i64
  %iter = arith.constant 1 : i64
  %da_addr = arith.constant 0 : i64

  %input = memref.get_global @fp_in : memref<1x16xf32>
  %output = memref.alloc() alignment = 64 : memref<1x16xi8>

  %bin = buckyball.bank_alloc {col = 4 : i64}
  %bout = buckyball.bank_alloc
  %loaded = buckyball.bank_mvin %input %bin %depth %stride
      : memref<1x16xf32> i64 i64 i64
  buckyball.fp2int %loaded, %bout, %iter, %da_addr : i64
  %stored = buckyball.bank_mvout %output %bout %depth %stride
      : memref<1x16xi8> i64 i64 i64
  buckyball.fence
  buckyball.bank_release %loaded : i64
  buckyball.bank_release %stored : i64

  func.call @check_result(%output) : (memref<1x16xi8>) -> ()
  memref.dealloc %output : memref<1x16xi8>
  return %zero_i8 : i8
}
