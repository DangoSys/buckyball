// RUN: buddy-opt %s --target=attention -lower-buckyball-to-bank-ssa | FileCheck %s --check-prefix=SSA
// RUN: buddy-opt %s --target=attention -lower-buckyball-to-bank-ssa -assign-physical-banks | FileCheck %s --check-prefix=BANK
// SSA-LABEL: func.func @packed
// SSA-COUNT-3: buckyball.bank_transfer
// SSA: buckyball.bank_kernel @rvv_packmm
// SSA: scf.for
// SSA: buckyball.bank_mvin {{.*}} <group = 0>
// SSA: buckyball.bank_mvin {{.*}} <group = 1>
// SSA: buckyball.bank_kernel @rvv_packmm
// SSA: buckyball.bank_mvout {{.*}} <group = 2>
// SSA: buckyball.bank_kernel @rvv_packmm
// SSA-COUNT-2: buckyball.bank_release
// SSA: return
// BANK-LABEL: func.func @packed
// BANK-COUNT-3: buckyball.mset_transfer
// BANK: call @rvv_packmm
// BANK: buckyball.mvin {{.*}} <group = 0>
// BANK: buckyball.mvin {{.*}} <group = 1>
// BANK: call @rvv_packmm
// BANK: buckyball.mvout {{.*}} <group = 2>
// BANK: call @rvv_packmm
// BANK-NOT: buckyball.bank_
// BANK: return
func.func @packed(%lhs: memref<16x8xf32>, %rhs: memref<8x13xf32>,
                  %output: memref<16x13xf32>) {
  buckyball.fp32_mem_matmul %lhs, %rhs, %output <fused = false> : memref<16x8xf32>, memref<8x13xf32>, memref<16x13xf32>
  return
}
