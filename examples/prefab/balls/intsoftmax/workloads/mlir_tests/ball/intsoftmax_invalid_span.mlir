func.func @invalid() {
  %m = arith.constant 33 : i64
  %k = arith.constant 128 : i64
  // expected-error @+1 {{softmax span exceeds bank capacity}}
  intsoftmax_ball.softmax %m, %k <input = 0, table = 1, output = 2>
  return
}
