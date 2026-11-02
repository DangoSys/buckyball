// PLACE-LABEL: func.func @program(
// PLACE: %[[SOURCE:[a-z0-9_]+]] = device.launch @producer
// PLACE-SAME: core = 1>
// PLACE: device.launch @consumer(%[[SOURCE]])
// PLACE-SAME: core = 2>
// PLACE: device.launch @consumer(%[[SOURCE]])
// PLACE-SAME: core = 1>
// LOWER-LABEL: func.func @program(
// LOWER: %[[PRODUCER:[a-z0-9_]+]] = call @program_job_submit
// LOWER: call @program_job_after(%[[PRODUCER]])
// LOWER: call @program_job_submit
// LOWER: call @program_job_after(%[[PRODUCER]])
// LOWER: call @program_job_submit
// LOWER: call @program_cleanup_begin
func.func private @producer(tensor<8xf32>) -> tensor<8xf32>
func.func private @consumer(tensor<8xf32>) -> tensor<8xf32>
func.func @program(%input: tensor<8xf32>) -> (tensor<8xf32>, tensor<8xf32>) {
  %a = tensor.empty() : tensor<8xf32>
  %b = tensor.empty() : tensor<8xf32>
  %c = tensor.empty() : tensor<8xf32>
  %source = "device.launch"(%input, %a) <{callee = @producer, core_class = "poly", operandSegmentSizes = array<i32: 1, 1>}> {device.rank = 0 : i64} : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
  %first = "device.launch"(%source, %b) <{callee = @consumer, core_class = "poly", operandSegmentSizes = array<i32: 1, 1>}> {device.rank = 1 : i64} : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
  %second = "device.launch"(%source, %c) <{callee = @consumer, core_class = "poly", operandSegmentSizes = array<i32: 1, 1>}> {device.rank = 0 : i64} : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
  return %first, %second : tensor<8xf32>, tensor<8xf32>
}
