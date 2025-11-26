/// Im2col execution logic
use super::compute::Im2colCompute;
use super::super::common::{MemoryInterface, SramReadReq, SramWriteReq};
use crate::builtin::Wire;

pub fn run_im2col(
    mem: &mut MemoryInterface,
    compute: &mut Im2colCompute,
    src_addr: u32,
    dst_addr: u32,
    cmd_resp: &mut Wire<u32>,
) -> bool {
    mem.clear_requests();

    if compute.should_read() {
        mem.sram_read_req.set(SramReadReq {
            addr: src_addr + compute.read_counter,
            bank: 0,
        });
        compute.read_counter += 1;
    }

    if mem.sram_read_resp.valid {
        compute.store_input(
            compute.resp_counter as usize,
            &mem.sram_read_resp.value.data
        );
        compute.resp_counter += 1;

        if compute.is_read_done() {
            compute.convert_im2col();
        }
    }

    if compute.is_read_done() && compute.should_write() {
        mem.sram_write_req.set(SramWriteReq {
            addr: dst_addr + compute.write_counter,
            bank: 0,
            data: compute.get_output_row(compute.write_counter as usize),
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
