// A value of poly hart 1 read on poly hart 2 crosses the device SHM in one
// transfer, routed by the hart links; its reader on hart 1 reads it in place.
// PLACE-LABEL: func.func @program(
// PLACE: %[[SOURCE:[0-9]+]] = device.launch @producer
// PLACE-SAME: hart = 1>
// PLACE-NOT: device.transfer
// PLACE: %[[MOVED:[0-9]+]] = device.transfer %[[SOURCE]]
// PLACE-SAME: <from = 1, to = 2>
// PLACE-NOT: device.transfer
// PLACE: device.launch @consumer(%[[MOVED]]
// PLACE-SAME: hart = 2>
// PLACE: device.launch @consumer(%[[SOURCE]]
// PLACE-SAME: hart = 1>
// LOWER-LABEL: func.func @program(
// LOWER: call @program_start
// LOWER: call @program_dispatch(%c0_i64
// The target of the transfer, before its reader.
// LOWER: llvm.getelementptr %{{.*}}[3]
// LOWER: call @program_dispatch(%c1_i64
// LOWER: call @program_dispatch(%c2_i64
// LOWER: call @program_finish
// LOWER: memref.dealloc
func.func private @producer(tensor<8xf32>) -> tensor<8xf32>
func.func private @consumer(tensor<8xf32>) -> tensor<8xf32>
func.func @program(%input: tensor<8xf32>) -> (tensor<8xf32>, tensor<8xf32>) {
  %a = tensor.empty() : tensor<8xf32>
  %b = tensor.empty() : tensor<8xf32>
  %c = tensor.empty() : tensor<8xf32>
  %source = "device.launch"(%input, %a) <{callee = @producer, hart_class = "poly", operandSegmentSizes = array<i32: 1, 1>}> {device.rank = 0 : i64} : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
  %first = "device.launch"(%source, %b) <{callee = @consumer, hart_class = "poly", operandSegmentSizes = array<i32: 1, 1>}> {device.rank = 1 : i64} : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
  %second = "device.launch"(%source, %c) <{callee = @consumer, hart_class = "poly", operandSegmentSizes = array<i32: 1, 1>}> {device.rank = 0 : i64} : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
  return %first, %second : tensor<8xf32>, tensor<8xf32>
}

// The first receiver waits for a later sender job. Sending the original
// value eagerly would block that job behind the second receiver.
// PLACE-LABEL: func.func @delayed_program(
// PLACE: %[[EARLY:[0-9]+]] = device.launch @producer
// PLACE: %[[LATE:[0-9]+]] = device.launch @consumer(%[[EARLY]]
// PLACE: device.transfer %[[LATE]]
// PLACE-NOT: device.defer
// PLACE: device.launch @consumer
// PLACE: device.transfer %[[EARLY]]
// PLACE-SAME: device.defer
// PLACE: device.launch @consumer
// LOWER-LABEL: func.func @delayed_program(
// LOWER: call @program_dispatch(%c0_i64
// LOWER: call @program_dispatch(%c1_i64
// LOWER: call @program_dispatch(%c2_i64
// LOWER: constant @program_transfer_job
// LOWER: call @program_dispatch(%c3_i64
// LOWER: call @program_dispatch(%c4_i64
// LOWER: call @program_finish
func.func @delayed_program(%input: tensor<8xf32>) -> tensor<8xf32> {
  %a = tensor.empty() : tensor<8xf32>
  %b = tensor.empty() : tensor<8xf32>
  %c = tensor.empty() : tensor<8xf32>
  %d = tensor.empty() : tensor<8xf32>
  %early = "device.launch"(%input, %a) <{callee = @producer, hart_class = "poly", operandSegmentSizes = array<i32: 1, 1>}> {device.rank = 0 : i64} : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
  %late = "device.launch"(%early, %b) <{callee = @consumer, hart_class = "poly", operandSegmentSizes = array<i32: 1, 1>}> {device.rank = 0 : i64} : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
  %first = "device.launch"(%late, %c) <{callee = @consumer, hart_class = "poly", operandSegmentSizes = array<i32: 1, 1>}> {device.rank = 1 : i64} : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
  %second = "device.launch"(%early, %d) <{callee = @consumer, hart_class = "poly", operandSegmentSizes = array<i32: 1, 1>}> {device.rank = 1 : i64} : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
  return %second : tensor<8xf32>
}

// Entry 0 on hart 1 pushes the transfer (words 2..5) and wakes entry 1 on
// hart 2, which pulls it; entry 2 on hart 1 reads the value in place.
// LOWER: llvm.mlir.global internal constant @program_table(dense<[3, 1, 3, 3, 0, 1, 1, 0, 0, 2, 0, 0, 0, 1, 1, 0, 1, 1, 2, 6, 1, 3, 2, 0, 2, 0, 2, 1, 3, 0, 1, 8, 0, 3, 3, 0, 3, 0, 3, 0, 3, 0, 1, 2, 2, 1, 0, 0]>
