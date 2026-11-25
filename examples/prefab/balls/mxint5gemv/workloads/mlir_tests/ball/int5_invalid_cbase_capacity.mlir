func.func @compute() {
  %base = arith.constant 1008 : i64
  // expected-error@+1 {{C base must be proven within the 17-row state window}}
  mxint5gemv_ball.gemv_mxint5 %base <a = 1, b = 2, c = 3, k = 512, a_codes = 0, a_exponents = 32, b_base = 0, accumulate = false, finalize = false, update = true>
  return
}
