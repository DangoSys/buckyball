// RUN: buddy-opt %s --target=ffn -fuse-norm-quant-window | FileCheck %s --check-prefix=SSA
// RUN: buddy-opt %s --target=ffn -fuse-norm-quant-window -assign-physical-banks | FileCheck %s --check-prefix=BANK

// SSA-LABEL: func.func @mixed
// SSA-COUNT-3: buckyball.bank_transfer
// SSA: buckyball.bank_kernel @rvv_norm_window
// SSA: scf.for
// SSA: buckyball.bank_mvin {{.*}} <group = 0>
// SSA: buckyball.bank_kernel @rvv_norm_window
// SSA: buckyball.bank_mvin {{.*}} <group = 0>
// SSA: buckyball.bank_kernel @rvv_norm_window
// SSA: buckyball.bank_mvout {{.*}} <group = 1>
// SSA: buckyball.bank_kernel @rvv_norm_window
// SSA-COUNT-2: buckyball.bank_release
// SSA-NOT: memref.copy
// SSA: return
// BANK-LABEL: func.func @mixed
// BANK-COUNT-3: buckyball.mset_transfer
// BANK: call @rvv_norm_window
// BANK: scf.for
// BANK: buckyball.mvin {{.*}} <group = 0>
// BANK: call @rvv_norm_window
// BANK: buckyball.mvin {{.*}} <group = 0>
// BANK: call @rvv_norm_window
// BANK: buckyball.mvout {{.*}} <group = 1>
// BANK: call @rvv_norm_window
// BANK-NOT: buckyball.bank_
// BANK: return
module {
  func.func private @rvv_norm(memref<*xf32>, memref<*xf32>, memref<*xf32>, i32, i32)
  func.func private @mxfp8_quant(memref<*xf32>, memref<*xi8>, i64, i64, i64)
  func.func @mixed(%input: memref<1x64xf32>, %gamma: memref<64xf32>,
                   %weights: memref<131072xi8, strided<[1], offset: ?>>) -> memref<1x20xf32> {
    %norm = memref.alloc() : memref<1x64xf32>
    %packed = memref.alloc() : memref<65536xi8>
    %output = memref.alloc() : memref<1x20xf32>
    %norm_view = memref.cast %norm : memref<1x64xf32> to memref<*xf32>
    %input_view = memref.cast %input : memref<1x64xf32> to memref<*xf32>
    %gamma_view = memref.cast %gamma : memref<64xf32> to memref<*xf32>
    %packed_view = memref.cast %packed : memref<65536xi8> to memref<*xi8>
    %mean = arith.constant 1015021568 : i32
    %epsilon = arith.constant 897988541 : i32
    %one = arith.constant 1 : i64
    %chunk = arith.constant 32 : i64
    %bytes = arith.constant 32768 : i64
    call @rvv_norm(%norm_view, %input_view, %gamma_view, %mean, %epsilon) : (memref<*xf32>, memref<*xf32>, memref<*xf32>, i32, i32) -> ()
    call @mxfp8_quant(%norm_view, %packed_view, %one, %chunk, %bytes) : (memref<*xf32>, memref<*xi8>, i64, i64, i64) -> ()
    buckyball.mxfp8_mem_matmul %packed, %weights, %output <reduction_k = 64, tile_k = 32, tile_m = 16, tile_n = 16, bank_bytes = 32768, activation_tile_k=32> : memref<65536xi8>, memref<131072xi8, strided<[1], offset: ?>>, memref<1x20xf32>
    return %output : memref<1x20xf32>
  }
}
