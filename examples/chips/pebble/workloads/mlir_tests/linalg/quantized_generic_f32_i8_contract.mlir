// CHECK: funct7 = 51
// CHECK: funct7 = 65
// CHECK: funct7 = 54

func.func @main(%a: memref<16x32xf32>, %b: memref<32x16xi8>,
                %c: memref<16x16xf32>) {
  linalg.generic {buckyball.quantized = true, dw_addr = 16 : i64,
                  dw_bytes = 64 : i64, per_channel = true,
                  indexing_maps = [affine_map<(m, n, k) -> (m, k)>,
                                   affine_map<(m, n, k) -> (k, n)>,
                                   affine_map<(m, n, k) -> (m, n)>],
                  iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%a, %b : memref<16x32xf32>, memref<32x16xi8>)
      outs(%c : memref<16x16xf32>) {
    ^bb0(%x: f32, %w: i8, %sum: f32):
      %wf = arith.sitofp %w : i8 to f32
      %product = arith.mulf %x, %wf : f32
      %next = arith.addf %sum, %product : f32
      linalg.yield %next : f32
  }
  return
}
