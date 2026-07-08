// Tile Dialect matmul contract test: K-blocked 16x2048 @ 2048x16.
// Uses a small bank_depth in CMake to force K splitting.
//
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1024:.*]] = arith.constant 1024 : index
// CHECK-DAG: %[[C2048:.*]] = arith.constant 2048 : index
// CHECK: scf.for {{.*}} to {{.*}} step {{.*}} {
// CHECK:   scf.for {{.*}} to {{.*}} step {{.*}} {
// CHECK:     %[[C_TILE:.*]] = memref.subview
// CHECK:     %[[PARTIAL:.*]] = memref.alloc() : memref<16x16xf32>
// CHECK:     linalg.fill
// CHECK:     scf.for {{.*}} = %[[C0]] to %[[C2048]] step %[[C1024]] {
// CHECK:       buckyball.matmul {{.*}} %[[PARTIAL]]
// CHECK:       scf.for {{.*}} = %[[C0]] to {{.*}} step {{.*}} {
// CHECK:         scf.for {{.*}} = %[[C0]] to {{.*}} step {{.*}} {
// CHECK:           memref.load %[[C_TILE]]
// CHECK:           memref.load %[[PARTIAL]]
// CHECK:           arith.addf
// CHECK:           memref.store {{.*}}, %[[C_TILE]]
// CHECK:     memref.dealloc %[[PARTIAL]]

func.func @main() -> i8 {
  %zero_i8 = arith.constant 0 : i8
  %one_f32 = arith.constant 1.0 : f32
  %zero_f32 = arith.constant 0.0 : f32

  %a = memref.alloc() : memref<16x2048xf32>
  %b = memref.alloc() : memref<2048x16xf32>
  %c = memref.alloc() : memref<16x16xf32>

  linalg.fill ins(%one_f32 : f32) outs(%a : memref<16x2048xf32>)
  linalg.fill ins(%one_f32 : f32) outs(%b : memref<2048x16xf32>)
  linalg.fill ins(%zero_f32 : f32) outs(%c : memref<16x16xf32>)

  tile.tile_matmul %a %b %c
    : memref<16x2048xf32> memref<2048x16xf32> memref<16x16xf32>

  memref.dealloc %a : memref<16x2048xf32>
  memref.dealloc %b : memref<2048x16xf32>
  memref.dealloc %c : memref<16x16xf32>

  return %zero_i8 : i8
}
