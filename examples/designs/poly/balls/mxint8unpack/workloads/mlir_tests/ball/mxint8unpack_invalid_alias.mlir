func.func @invalid() {
  %b = arith.constant 1 : i64
  // expected-error @+1 {{requires two distinct 16 KiB banks}}
  mxint8unpack_ball.unpack %b <input = 1, output = 1>
  return
}
