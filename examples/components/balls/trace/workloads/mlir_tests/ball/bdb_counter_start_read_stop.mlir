func.func @main() -> i8 {
  %zero8 = arith.constant 0 : i8
  %tag = arith.constant 48879 : i64
  buckyball.custom "counter_start" args(%tag : i64) <{sources = array<i64>, destination = 0 : i64}> {counter = 3 : i64}
  buckyball.custom "counter_read" <{sources = array<i64>, destination = 0 : i64}> {counter = 3 : i64}
  buckyball.custom "counter_stop" <{sources = array<i64>, destination = 0 : i64}> {counter = 3 : i64}
  return %zero8 : i8
}
