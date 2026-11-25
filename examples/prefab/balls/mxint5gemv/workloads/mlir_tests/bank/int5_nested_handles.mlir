// BANK-LABEL: func.func @bank_numeric_loops
// BANK: scf.for {{.*}} iter_args({{.*}}) -> (i64)
// BANK: scf.for {{.*}} iter_args({{.*}}) -> (i64)
// BANK: mxint5gemv_ball.gemv_mxint5
// BANK: arith.addi
// BANK: scf.yield {{.*}} : i64
// BANK: scf.yield {{.*}} : i64
func.func @bank_numeric_loops(%n: index) -> i64 {
  %z = arith.constant 0 : index
  %one = arith.constant 1 : index
  %two = arith.constant 2 : index
  %zero = arith.constant 0 : i64
  %inc = arith.constant 1 : i64
  %a = bbacc.bank_alloc <row = 1, col = 1>
  %b = bbacc.bank_alloc <row = 1, col = 4>
  %c = bbacc.bank_alloc <row = 1, col = 1>
  %out:2 = scf.for %r = %z to %n step %one iter_args(%bank = %c, %sum = %zero) -> (i64,i64) {
    %inner:2 = scf.for %s = %z to %two step %one iter_args(%old = %bank, %acc = %sum) -> (i64,i64) {
      %new = mxint5gemv_ball.bank_gemv_mxint5 %a, %b -> %old at %zero <k = 32, a_codes = 0, a_exponents = 2, b_base = 0, accumulate = false, finalize = false, update = true>
      %next = arith.addi %acc, %inc : i64
      scf.yield %new, %next : i64,i64
    }
    scf.yield %inner#0, %inner#1 : i64,i64
  }
  bbacc.bank_release %a
  bbacc.bank_release %b
  bbacc.bank_release %out#0
  return %out#1 : i64
}
