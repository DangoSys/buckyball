/// Transpose execution logic
use super::compute::TransposeCompute;
use crate::builtin::ball::Blink;

pub fn run_transpose(
  blink: &mut Blink,
  compute: &mut TransposeCompute,
  src_addr: u32,
  dst_addr: u32,
) -> bool {
  if compute.should_read() {
    blink.sram_read_req[0].valid = true;
    blink.sram_read_req[0].addr = src_addr + compute.read_counter;
    compute.read_counter += 1;
  }

  if blink.sram_read_resp[0].valid {
    compute.store_row(compute.resp_counter as usize, &blink.sram_read_resp[0].data);
    compute.resp_counter += 1;
  }

  if compute.is_read_done() && compute.should_write() {
    blink.sram_write_req[0].valid = true;
    blink.sram_write_req[0].addr = dst_addr + compute.write_counter;
    blink.sram_write_req[0].data = compute.get_transposed_row(compute.write_counter as usize);
    compute.write_counter += 1;
  }

  if compute.is_write_done() {
    blink.cmd_resp.valid = true;
    blink.cmd_resp.rob_id = blink.cmd_req.rob_id;
    true
  } else { false }
}
