func.func @compute(%input: memref<16xi32>, %table: memref<4096xi32>, %out: memref<16xi32>) attributes {llvm.emit_c_interface} {
  %m = arith.constant 2 : i64
  %k = arith.constant 7 : i64
  bbacc.mset <bank = 0, alloc = true, row = 1, col = 1>
  bbacc.mset <bank = 1, alloc = true, row = 1, col = 1>
  bbacc.mset <bank = 2, alloc = true, row = 1, col = 1>
  bbacc.mvin %input <bank = 0, depth = 4, stride = 1> : memref<16xi32>
  bbacc.mvin %table <bank = 1, depth = 1024, stride = 1> : memref<4096xi32>
  intsoftmax_ball.softmax %m, %k <input = 0, table = 1, output = 2>
  bbacc.mvout %out <bank = 2, depth = 4, stride = 1> : memref<16xi32>
  bbacc.mset <bank = 0, alloc = false, row = 0, col = 0>
  bbacc.mset <bank = 1, alloc = false, row = 0, col = 0>
  bbacc.mset <bank = 2, alloc = false, row = 0, col = 0>
  return
}
