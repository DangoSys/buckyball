func.func @compute(%input: memref<4096xi32>, %table: memref<4096xi32>, %out: memref<4096xi32>) attributes {llvm.emit_c_interface} {
  %m = arith.constant 32 : i64
  %k = arith.constant 128 : i64
  %a = bbacc.bank_alloc <row = 1, col = 1>
  %b = bbacc.bank_alloc <row = 1, col = 1>
  %c = bbacc.bank_alloc <row = 1, col = 1>
  %x = bbacc.bank_mvin %input, %a <depth = 1024, stride = 1> : memref<4096xi32>
  %t = bbacc.bank_mvin %table, %b <depth = 1024, stride = 1> : memref<4096xi32>
  %y = intsoftmax_ball.bank_softmax %x, %t -> %c rows %m columns %k
  %done = bbacc.bank_mvout %out, %y <depth = 1024, stride = 1> : memref<4096xi32>
  bbacc.bank_release %x
  bbacc.bank_release %t
  bbacc.bank_release %done
  return
}
