/// Vector execution logic
use super::compute::VecCompute;
use super::super::common::VECLANE;
use crate::builtin::ball::Blink;

pub fn run_vector(
  blink: &mut Blink,
  compute: &mut VecCompute,
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
    if row < VECLANE {
      compute.store_op1(row, &blink.sram_read_resp[0].data);
    } else {
      compute.store_op2(row - VECLANE, &blink.sram_read_resp[0].data);
    }
    compute.resp_counter += 1;
    if compute.is_read_done() { compute.compute_add(); }
  }

  if compute.is_read_done() && compute.should_write() {
    blink.acc_write_req[0].valid = true;
    blink.acc_write_req[0].addr = dst_addr + compute.write_counter;
    blink.acc_write_req[0].data = compute.get_result_row(compute.write_counter as usize);
    compute.write_counter += 1;
  }

  if compute.is_write_done() {
    blink.cmd_resp.valid = true;
    blink.cmd_resp.rob_id = blink.cmd_req.rob_id;
    true
  } else { false }
}
