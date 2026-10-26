// RUN: buddy-opt %s -lower-gather-rows | FileCheck %s

// CHECK-LABEL: func.func @scatter_rows
// CHECK-SAME: %[[INPUT:.*]]: memref<1x4x8x128xf32>
// CHECK: %[[OUTPUT:.*]] = memref.alloc() : memref<1x4x16x128xf32>
// CHECK: scf.for %[[HEAD:.*]] =
// CHECK: scf.for %[[TOKEN:.*]] =
// CHECK: %[[POSITION:.*]] = memref.load
// CHECK-NOT: vector.transfer
// CHECK: %[[SOURCE:.*]] = memref.subview %[[INPUT]][{{.*}}, %[[HEAD]], %[[TOKEN]], {{.*}}] [1, 1, 1, 128] [1, 1, 1, 1]
// CHECK: %[[DESTINATION:.*]] = memref.subview %[[OUTPUT]][{{.*}}, %[[HEAD]], %[[POSITION]], {{.*}}] [1, 1, 1, 128] [1, 1, 1, 1]
// CHECK: memref.copy %[[SOURCE]], %[[DESTINATION]]
// CHECK-NOT: vector.transfer
// CHECK: return %[[OUTPUT]]
func.func @scatter_rows(%input: memref<1x4x8x128xf32>, %positions: memref<8xindex>) -> memref<1x4x16x128xf32> {
  %output = memref.alloc() : memref<1x4x16x128xf32>
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %four = arith.constant 4 : index
  %eight = arith.constant 8 : index
  %padding = arith.constant 0.0 : f32
  scf.for %head = %zero to %four step %one {
    scf.for %token = %zero to %eight step %one {
      %position = memref.load %positions[%token] : memref<8xindex>
      %row = vector.transfer_read %input[%zero, %head, %token, %zero], %padding {in_bounds = [true]} : memref<1x4x8x128xf32>, vector<128xf32>
      vector.transfer_write %row, %output[%zero, %head, %position, %zero] {in_bounds = [true]} : vector<128xf32>, memref<1x4x16x128xf32>
    }
  }
  return %output : memref<1x4x16x128xf32>
}

// CHECK-LABEL: func.func @alias_view
// CHECK-NOT: memref.copy
// CHECK: vector.transfer_read
// CHECK: vector.transfer_write
func.func @alias_view(%row: index) -> memref<4x128xf32> {
  %buffer = memref.alloc() : memref<4x128xf32>
  %view = memref.cast %buffer : memref<4x128xf32> to memref<?x128xf32>
  %zero = arith.constant 0 : index
  %padding = arith.constant 0.0 : f32
  %value = vector.transfer_read %view[%zero, %zero], %padding {in_bounds = [true]} : memref<?x128xf32>, vector<128xf32>
  vector.transfer_write %value, %buffer[%row, %zero] {in_bounds = [true]} : vector<128xf32>, memref<4x128xf32>
  return %buffer : memref<4x128xf32>
}

// CHECK-LABEL: func.func @masked
// CHECK-NOT: memref.copy
// CHECK: vector.transfer_read
// CHECK: vector.transfer_write
func.func @masked(%input: memref<4x128xf32>, %mask: vector<128xi1>) -> memref<4x128xf32> {
  %output = memref.alloc() : memref<4x128xf32>
  %zero = arith.constant 0 : index
  %padding = arith.constant 0.0 : f32
  %value = vector.transfer_read %input[%zero, %zero], %padding, %mask {in_bounds = [true]} : memref<4x128xf32>, vector<128xf32>
  vector.transfer_write %value, %output[%zero, %zero], %mask {in_bounds = [true]} : vector<128xf32>, memref<4x128xf32>
  return %output : memref<4x128xf32>
}

// CHECK-LABEL: func.func @strided
// CHECK-NOT: memref.copy
// CHECK: vector.transfer_read
// CHECK: vector.transfer_write
func.func @strided(%input: memref<4x128xf32, strided<[256, 2], offset: ?>>) -> memref<4x128xf32> {
  %output = memref.alloc() : memref<4x128xf32>
  %zero = arith.constant 0 : index
  %padding = arith.constant 0.0 : f32
  %value = vector.transfer_read %input[%zero, %zero], %padding {in_bounds = [true]} : memref<4x128xf32, strided<[256, 2], offset: ?>>, vector<128xf32>
  vector.transfer_write %value, %output[%zero, %zero] {in_bounds = [true]} : vector<128xf32>, memref<4x128xf32>
  return %output : memref<4x128xf32>
}

// CHECK-LABEL: func.func @padding
// CHECK-NOT: memref.copy
// CHECK: vector.transfer_read
// CHECK: vector.transfer_write
func.func @padding(%input: memref<4x128xf32>, %offset: index) -> memref<4x128xf32> {
  %output = memref.alloc() : memref<4x128xf32>
  %zero = arith.constant 0 : index
  %padding = arith.constant 0.0 : f32
  %value = vector.transfer_read %input[%zero, %offset], %padding {in_bounds = [false]} : memref<4x128xf32>, vector<128xf32>
  vector.transfer_write %value, %output[%zero, %zero] {in_bounds = [true]} : vector<128xf32>, memref<4x128xf32>
  return %output : memref<4x128xf32>
}

// CHECK-LABEL: func.func @intervening_store
// CHECK-NOT: memref.copy
// CHECK: vector.transfer_read
// CHECK: memref.store
// CHECK: vector.transfer_write
func.func @intervening_store(%input: memref<4x128xf32>) -> memref<4x128xf32> {
  %output = memref.alloc() : memref<4x128xf32>
  %zero = arith.constant 0 : index
  %padding = arith.constant 0.0 : f32
  %value = vector.transfer_read %input[%zero, %zero], %padding {in_bounds = [true]} : memref<4x128xf32>, vector<128xf32>
  memref.store %padding, %input[%zero, %zero] : memref<4x128xf32>
  vector.transfer_write %value, %output[%zero, %zero] {in_bounds = [true]} : vector<128xf32>, memref<4x128xf32>
  return %output : memref<4x128xf32>
}
