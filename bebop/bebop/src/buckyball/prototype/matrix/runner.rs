/// Matrix execution logic
use super::compute::MatrixCompute;
use super::super::common::{MemoryInterface, SramReadReq, SramWriteReq, VECLANE};
use crate::builtin::Wire;

pub fn run_matrix(
    mem: &mut MemoryInterface,
    compute: &mut MatrixCompute,
    op1_addr: u32,
    op2_addr: u32,
    dst_addr: u32,
    cmd_resp: &mut Wire<u32>,
) -> bool {
    mem.clear_requests();

    if compute.should_read() {
        let is_op1 = compute.read_counter < VECLANE as u32;
        let addr = if is_op1 {
            op1_addr + compute.read_counter
        } else {
            op2_addr + (compute.read_counter - VECLANE as u32)
        };
        mem.sram_read_req.set(SramReadReq { addr, bank: 0 });
        compute.read_counter += 1;
    }

    if mem.sram_read_resp.valid {
        let row = compute.resp_counter as usize;
        let is_op1 = row < VECLANE;
        let actual_row = if is_op1 { row } else { row - VECLANE };
        compute.store_data(actual_row, &mem.sram_read_resp.value.data, is_op1);
        compute.resp_counter += 1;

        if compute.is_read_done() {
            compute.compute_matmul();
        }
    }

    if compute.is_read_done() && compute.should_write() {
        mem.sram_write_req.set(SramWriteReq {
            addr: dst_addr + compute.write_counter,
            bank: 0,
            data: compute.get_result_row(compute.write_counter as usize)
                .iter().map(|&x| x as i8).collect(),
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
