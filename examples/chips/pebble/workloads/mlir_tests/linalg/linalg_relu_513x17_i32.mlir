memref.global "private" constant @zero : memref<513x17xi32> = dense<0>

func.func private @check_result(memref<513x17xi32>) -> ()

func.func @main() -> i8 {
  %zeroI8 = arith.constant 0 : i8
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %rows = arith.constant 513 : index
  %columns = arith.constant 17 : index
  %thirtySeven = arith.constant 37 : i32
  %eighteen = arith.constant 18 : i32
  %input = memref.alloc() alignment = 64 : memref<513x17xi32>
  %output = memref.alloc() alignment = 64 : memref<513x17xi32>
  %zeroInput = memref.get_global @zero : memref<513x17xi32>

  scf.for %row = %zero to %rows step %one {
    scf.for %column = %zero to %columns step %one {
      %rowI32 = arith.index_cast %row : index to i32
      %columnI32 = arith.index_cast %column : index to i32
      %value = arith.addi %rowI32, %columnI32 : i32
      %remainder = arith.remui %value, %thirtySeven : i32
      %signed = arith.subi %remainder, %eighteen : i32
      memref.store %signed, %input[%row, %column] : memref<513x17xi32>
    }
  }

  linalg.generic {
    indexing_maps = [affine_map<(row, column) -> (row, column)>,
                     affine_map<(row, column) -> (row, column)>,
                     affine_map<(row, column) -> (row, column)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%input, %zeroInput : memref<513x17xi32>, memref<513x17xi32>)
    outs(%output : memref<513x17xi32>) {
  ^bb0(%value: i32, %zeroValue: i32, %old: i32):
    %relu = arith.maxsi %value, %zeroValue : i32
    linalg.yield %relu : i32
  }

  func.call @check_result(%output) : (memref<513x17xi32>) -> ()
  memref.dealloc %input : memref<513x17xi32>
  memref.dealloc %output : memref<513x17xi32>
  return %zeroI8 : i8
}
