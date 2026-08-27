// 513x17 is padded to 528x32, then packed linearly into five physical pbank
// chunks. Every ReLU command owns one pbank in place.
// CHECK: funct7 = 50
// CHECK: funct7 = 50
// CHECK: funct7 = 50
// CHECK: funct7 = 50
// CHECK: funct7 = 50

memref.global "private" constant @zero : memref<513x17xi32> = dense<0>

func.func @main(%input: memref<513x17xi32>, %output: memref<513x17xi32>) {
  %zero = memref.get_global @zero : memref<513x17xi32>
  linalg.generic {
    indexing_maps = [affine_map<(row, column) -> (row, column)>,
                     affine_map<(row, column) -> (row, column)>,
                     affine_map<(row, column) -> (row, column)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%input, %zero : memref<513x17xi32>, memref<513x17xi32>)
    outs(%output : memref<513x17xi32>) {
  ^bb0(%value: i32, %zeroValue: i32, %old: i32):
    %relu = arith.maxsi %value, %zeroValue : i32
    linalg.yield %relu : i32
  }
  return
}
