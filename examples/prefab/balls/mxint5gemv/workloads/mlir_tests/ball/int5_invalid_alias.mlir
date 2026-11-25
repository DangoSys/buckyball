func.func @invalid_alias() {
  %base = arith.constant 4 : i64
  // expected-error @+1 {{GEMV requires distinct A, B and C banks}}
  mxint5gemv_ball.gemv_mxint5 %base <a = 1, b = 1, c = 3, k = 512, a_codes = 2, a_exponents = 40, b_base = 3, accumulate = false, finalize = true, update = false>
  return
}
