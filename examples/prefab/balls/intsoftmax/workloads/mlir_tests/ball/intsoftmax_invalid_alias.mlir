func.func @invalid() {
  %m = arith.constant 1 : i64
  %k = arith.constant 128 : i64
  // expected-error @+1 {{requires three distinct 16 KiB banks}}
  intsoftmax_ball.softmax %m, %k <input = 0, table = 1, output = 0>
  return
}
