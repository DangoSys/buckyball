// BANK: maxpool_ball.bank_topk_u32
module {
  func.func private @check_result(memref<1x3xi32>, memref<1x3xi64>)
  func.func @main() -> i8 {
    %c0_i8 = arith.constant 0 : i8
    %c0_i32 = arith.constant 0 : i32
    %c2 = arith.constant 2 : index
    %cst = arith.constant -2147483648 : i32
    %true = arith.constant true
    %false = arith.constant false
    %c5 = arith.constant 5 : index
    %c3 = arith.constant 3 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst_0 = arith.constant dense<[[0, 8388608, 16777216, 4194304, 8388608]]> : tensor<1x5xi32>
    %0 = scf.execute_region -> memref<2x4xi32> no_inline {
      %alloc_2 = memref.alloc() : memref<1x3xi32>
      %alloc_3 = memref.alloc() : memref<1x3xi64>
      %2 = bufferization.to_buffer %cst_0 : tensor<1x5xi32> to memref<1x5xi32>
      %alloc_4 = memref.alloc() : memref<1x5xi1>
      scf.for %arg0 = %c0 to %c5 step %c1 {
        memref.store %false, %alloc_4[%c0, %arg0] : memref<1x5xi1>
      }
      scf.for %arg0 = %c0 to %c3 step %c1 {
        %alloca = memref.alloca() : memref<i32>
        %alloca_6 = memref.alloca() : memref<index>
        memref.store %cst, %alloca[] : memref<i32>
        memref.store %c0, %alloca_6[] : memref<index>
        scf.for %arg1 = %c0 to %c5 step %c1 {
          %6 = memref.load %alloc_4[%c0, %arg1] : memref<1x5xi1>
          %7 = arith.cmpi ne, %6, %true : i1
          scf.if %7 {
            %8 = memref.load %2[%c0, %arg1] : memref<1x5xi32>
            %9 = memref.load %alloca[] : memref<i32>
            %10 = arith.cmpi sgt, %8, %9 : i32
            scf.if %10 {
              memref.store %8, %alloca[] : memref<i32>
              memref.store %arg1, %alloca_6[] : memref<index>
            }
          }
        }
        %3 = memref.load %alloca[] : memref<i32>
        %4 = memref.load %alloca_6[] : memref<index>
        memref.store %3, %alloc_2[%c0, %arg0] : memref<1x3xi32>
        %5 = arith.index_cast %4 : index to i64
        memref.store %5, %alloc_3[%c0, %arg0] : memref<1x3xi64>
        memref.store %true, %alloc_4[%c0, %4] : memref<1x5xi1>
      }
      %alloc_5 = memref.alloc() : memref<2x4xi32>
      scf.for %arg0 = %c0 to %c3 step %c1 {
        %3 = arith.divui %arg0, %c2 : index
        %4 = arith.remui %arg0, %c2 : index
        %5 = arith.muli %4, %c2 : index
        %6 = memref.load %alloc_2[%c0, %arg0] : memref<1x3xi32>
        %7 = arith.addi %6, %c0_i32 : i32
        %8 = memref.load %alloc_3[%c0, %arg0] : memref<1x3xi64>
        %9 = arith.trunci %8 : i64 to i32
        memref.store %7, %alloc_5[%3, %5] : memref<2x4xi32>
        %10 = arith.addi %5, %c1 : index
        memref.store %9, %alloc_5[%3, %10] : memref<2x4xi32>
      }
      memref.store %c0_i32, %alloc_5[%c1, %c2] : memref<2x4xi32>
      memref.store %c0_i32, %alloc_5[%c1, %c3] : memref<2x4xi32>
      scf.yield %alloc_5 : memref<2x4xi32>
    } {pebble.topk_u32 = 3 : i64}
    %1 = bufferization.to_tensor %0 restrict : memref<2x4xi32> to tensor<2x4xi32>
    %alloc = memref.alloc() : memref<1x3xi32>
    %alloc_1 = memref.alloc() : memref<1x3xi64>
    scf.for %arg0 = %c0 to %c3 step %c1 {
      %2 = arith.divui %arg0, %c2 : index
      %3 = arith.remui %arg0, %c2 : index
      %4 = arith.muli %3, %c2 : index
      %extracted = tensor.extract %1[%2, %4] : tensor<2x4xi32>
      %5 = arith.addi %extracted, %c0_i32 : i32
      %6 = arith.addi %4, %c1 : index
      %extracted_2 = tensor.extract %1[%2, %6] : tensor<2x4xi32>
      %7 = arith.extui %extracted_2 : i32 to i64
      memref.store %5, %alloc[%c0, %arg0] : memref<1x3xi32>
      memref.store %7, %alloc_1[%c0, %arg0] : memref<1x3xi64>
    }
    call @check_result(%alloc, %alloc_1) : (memref<1x3xi32>, memref<1x3xi64>) -> ()
    return %c0_i8 : i8
  }
}
