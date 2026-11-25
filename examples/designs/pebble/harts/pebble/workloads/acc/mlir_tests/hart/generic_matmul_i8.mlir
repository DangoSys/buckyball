func.func private @check_result(memref<1x32xi32>)

func.func @main() -> i8 {
  %exit = arith.constant 0 : i8
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %c7 = arith.constant 7 : index
  %c9 = arith.constant 9 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %a = memref.alloc() alignment = 64 : memref<1x32xi8>
  %b = memref.alloc() alignment = 64 : memref<32x32xi8>
  %c = memref.alloc() alignment = 64 : memref<1x32xi32>
  scf.for %k = %c0 to %c32 step %c1 {
    %rem = arith.remui %k, %c7 : index
    %value = arith.subi %rem, %c3 : index
    %a8 = arith.index_cast %value : index to i8
    memref.store %a8, %a[%c0, %k] : memref<1x32xi8>
    %initial = arith.subi %k, %c16 : index
    %c32value = arith.index_cast %initial : index to i32
    memref.store %c32value, %c[%c0, %k] : memref<1x32xi32>
    scf.for %column = %c0 to %c32 step %c1 {
      %k3 = arith.muli %k, %c3 : index
      %column5 = arith.muli %column, %c5 : index
      %sum = arith.addi %k3, %column5 : index
      %rem9 = arith.remui %sum, %c9 : index
      %signed = arith.subi %rem9, %c4 : index
      %b8 = arith.index_cast %signed : index to i8
      memref.store %b8, %b[%k, %column] : memref<32x32xi8>
    }
  }
  %lhs = bufferization.to_tensor %a restrict : memref<1x32xi8> to tensor<1x32xi8>
  %rhs = bufferization.to_tensor %b restrict : memref<32x32xi8> to tensor<32x32xi8>
  %initial = bufferization.to_tensor %c restrict : memref<1x32xi32> to tensor<1x32xi32>
  %product = linalg.generic {
    indexing_maps = [affine_map<(m,n,k)->(m,k)>, affine_map<(m,n,k)->(k,n)>, affine_map<(m,n,k)->(m,n)>],
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%lhs, %rhs: tensor<1x32xi8>, tensor<32x32xi8>) outs(%initial: tensor<1x32xi32>) {
  ^bb0(%x: i8, %y: i8, %acc: i32):
    %x32 = arith.extsi %x : i8 to i32
    %y32 = arith.extsi %y : i8 to i32
    %mul = arith.muli %x32, %y32 : i32
    %sum = arith.addi %acc, %mul : i32
    linalg.yield %sum : i32
  } -> tensor<1x32xi32>
  %output = bufferization.to_buffer %product : tensor<1x32xi32> to memref<1x32xi32>
  func.call @check_result(%output) : (memref<1x32xi32>) -> ()
  return %exit : i8
}
