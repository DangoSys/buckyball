func.func @invalid_k() {
  %base = arith.constant 4 : i64
  // expected-error @+1 {{GEMV window exceeds the bank or K is not aligned to 32}}
  mxint5gemv_ball.gemv_mxint5 %base <a = 1, b = 2, c = 3, k = 31, a_codes = 2, a_exponents = 40, b_base = 3, accumulate = false, finalize = true, update = false>
  return
}
