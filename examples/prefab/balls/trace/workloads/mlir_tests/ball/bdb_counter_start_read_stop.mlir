func.func @main() -> i8 {
  %zero8 = arith.constant 0 : i8
  %tag = arith.constant 48879 : i64
  trace_ball.counter_start %tag <{counter = 3 : i64}>
  trace_ball.counter_read <{counter = 3 : i64}>
  trace_ball.counter_stop <{counter = 3 : i64}>
  return %zero8 : i8
}
