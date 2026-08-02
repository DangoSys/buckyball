// TransposeBall ISA contract:
// bank_transpose -> transpose(iter, elem_bits)
// rs2 low 8 bits = elem_bits; funct7 = 49
//
// CHECK-PHYSICAL-DAG: arith.constant 16 : i64
// CHECK-PHYSICAL-DAG: arith.constant 8 : i64
// CHECK-PHYSICAL: buckyball.transpose {{.*}}, {{.*}}, {{.*}}, {{.*}} : i64
// CHECK-INTRIN: buckyball.intr.custom {{.*}}, {{.*}} {funct7 = 49 : i32}

func.func @transpose_ops(%in: i64, %out: i64) {
  %iter = arith.constant 16 : i64
  %elem_bits = arith.constant 8 : i64
  %r = buckyball.bank_transpose %in %out %iter %elem_bits
      : i64 i64 i64 i64
  return
}
