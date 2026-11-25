func.func @invalid() {
  %m = arith.constant 1 : i64
  %k = arith.constant 129 : i64
  // expected-error @+1 {{columns must be between one and 128}}
  intsoftmax_ball.softmax %m, %k <input = 0, table = 1, output = 2>
  return
}
