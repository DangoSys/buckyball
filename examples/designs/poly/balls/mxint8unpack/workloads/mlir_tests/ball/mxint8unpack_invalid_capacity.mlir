func.func @invalid() {
  %b = arith.constant 129 : i64
  // expected-error @+1 {{blocks must be between one and 128}}
  mxint8unpack_ball.unpack %b <input = 1, output = 2>
  return
}
