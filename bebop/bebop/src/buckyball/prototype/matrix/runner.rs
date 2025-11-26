/// Matrix execution logic
use super::compute::MatrixCompute;
use super::super::common::VECLANE;
use crate::builtin::ball::Blink;

pub fn run_matrix(
  blink: &mut Blink,
  compute: &mut MatrixCompute,
  op1_addr: u32,
  op2_addr: u32,
  dst_addr: u32,
) -> bool {
  if compute.should_read() {
    let is_op1 = compute.read_counter < VECLANE as u32;
    let addr = if is_op1 { op1_addr + compute.read_counter }
           else { op2_addr + (compute.read_counter - VECLANE as u32) };
    blink.sram_read_req[0].valid = true;
    blink.sram_read_req[0].addr = addr;
    compute.read_counter += 1;
  }

  if blink.sram_read_resp[0].valid {
    let row = compute.resp_counter as usize;
    let is_op1 = row < VECLANE;
    let actual_row = if is_op1 { row } else { row - VECLANE };
    compute.store_data(actual_row, &blink.sram_read_resp[0].data, is_op1);
    compute.resp_counter += 1;
    if compute.is_read_done() { compute.compute_matmul(); }
  }

  if compute.is_read_done() && compute.should_write() {
    blink.sram_write_req[0].valid = true;
    blink.sram_write_req[0].addr = dst_addr + compute.write_counter;
    blink.sram_write_req[0].data = compute.get_result_row(compute.write_counter as usize)
      .iter().map(|&x| x as i8).collect();
    compute.write_counter += 1;
  }

  if compute.is_write_done() {
    blink.cmd_resp.valid = true;
    blink.cmd_resp.rob_id = blink.cmd_req.rob_id;
    true
  } else { false }
}
