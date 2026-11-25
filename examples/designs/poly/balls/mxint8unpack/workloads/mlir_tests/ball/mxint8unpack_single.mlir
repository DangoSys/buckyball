func.func @compute(%input: memref<48xi8>, %out: memref<32xi32>, %source_copy: memref<48xi8>) attributes {llvm.emit_c_interface} {
  %blocks = arith.constant 1 : i64
  bbacc.mset <bank = 1, alloc = true, row = 1, col = 1>
  bbacc.mset <bank = 2, alloc = true, row = 1, col = 1>
  bbacc.mvin %input <bank = 1, depth = 3, stride = 1> : memref<48xi8>
  mxint8unpack_ball.unpack %blocks <input = 1, output = 2>
  bbacc.mvout %out <bank = 2, depth = 8, stride = 1> : memref<32xi32>
  bbacc.mvout %source_copy <bank = 1, depth = 3, stride = 1> : memref<48xi8>
  bbacc.mset <bank = 1, alloc = false, row = 0, col = 0>
  bbacc.mset <bank = 2, alloc = false, row = 0, col = 0>
  return
}
