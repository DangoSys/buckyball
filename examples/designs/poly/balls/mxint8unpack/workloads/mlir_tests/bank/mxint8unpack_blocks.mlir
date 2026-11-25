func.func @compute(%input: memref<576xi8>, %out: memref<544xi32>, %source_copy: memref<576xi8>) attributes {llvm.emit_c_interface} {
  %blocks = arith.constant 17 : i64
  %a = bbacc.bank_alloc <row = 1, col = 1>
  %c = bbacc.bank_alloc <row = 1, col = 1>
  %x = bbacc.bank_mvin %input, %a <depth = 36, stride = 1> : memref<576xi8>
  %y = mxint8unpack_ball.bank_unpack %x -> %c blocks %blocks
  %done = bbacc.bank_mvout %out, %y <depth = 136, stride = 1> : memref<544xi32>
  %copied = bbacc.bank_mvout %source_copy, %x <depth = 36, stride = 1> : memref<576xi8>
  bbacc.bank_release %copied
  bbacc.bank_release %done
  return
}
