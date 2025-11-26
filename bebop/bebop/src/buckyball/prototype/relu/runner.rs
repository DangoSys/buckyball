/// ReLU execution logic
use super::compute::ReluCompute;
use super::super::common::{MemoryInterface, SramReadReq, SramWriteReq};
use crate::builtin::Wire;

pub fn run_relu(
    mem: &mut MemoryInterface,
    compute: &mut ReluCompute,
    raddr: u32,
    waddr: u32,
    cmd_resp: &mut Wire<u32>,
) -> bool {
    mem.clear_requests();

    if compute.should_read() {
        mem.sram_read_req.set(SramReadReq {
            addr: raddr + compute.read_counter,
            bank: 0,
        });
        compute.read_counter += 1;
    }

    if mem.sram_read_resp.valid {
        compute.apply_relu(
            compute.resp_counter as usize,
            &mem.sram_read_resp.value.data
        );
        compute.resp_counter += 1;
    }

    if compute.is_read_done() && compute.should_write() {
        mem.sram_write_req.set(SramWriteReq {
            addr: waddr + compute.write_counter,
            bank: 0,
            data: compute.get_row(compute.write_counter as usize),
        });
        compute.write_counter += 1;
    }

    if compute.is_write_done() {
        cmd_resp.set(0);
        true
    } else {
        false
    }
}
