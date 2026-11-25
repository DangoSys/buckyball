func.func @invalid() {
  // expected-error @+1 {{fixed requires distinct banks}}
  fixed_ball.fixed <{a = 0 : i64, b = 1 : i64, c = 2 : i64, rows = 3 : i64, mode = 1 : i64, fraction = 24 : i64}>
  return
}
