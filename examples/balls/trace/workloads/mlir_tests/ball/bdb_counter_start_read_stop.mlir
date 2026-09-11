func.func @main() -> i8 {
  %zero8 = arith.constant 0 : i8
  %start = arith.constant 0 : i64
  %stop = arith.constant 1 : i64
  %read = arith.constant 2 : i64
  %counter = arith.constant 3 : i64
  %tag = arith.constant 48879 : i64
  buckyball.bdb_counter %start, %counter, %tag : i64
  buckyball.bdb_counter %read, %counter, %start : i64
  buckyball.bdb_counter %stop, %counter, %start : i64
  return %zero8 : i8
}
