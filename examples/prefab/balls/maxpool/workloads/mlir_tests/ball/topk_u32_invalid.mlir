func.func @empty_input() {
  // expected-error@+1 {{topk_u32 requires distinct banks, valid N/K and bank-sized rows/output}}
  maxpool_ball.topk_u32 <{input = 1 : i64, output = 2 : i64, rows = 1 : i64, count = 0 : i64, k = 1 : i64}>
  return
}

// -----

func.func @too_many_results() {
  // expected-error@+1 {{topk_u32 requires distinct banks, valid N/K and bank-sized rows/output}}
  maxpool_ball.topk_u32 <{input = 1 : i64, output = 2 : i64, rows = 1 : i64, count = 4 : i64, k = 5 : i64}>
  return
}

// -----

func.func @aliased_banks() {
  // expected-error@+1 {{topk_u32 requires distinct banks, valid N/K and bank-sized rows/output}}
  maxpool_ball.topk_u32 <{input = 1 : i64, output = 1 : i64, rows = 1 : i64, count = 4 : i64, k = 1 : i64}>
  return
}
