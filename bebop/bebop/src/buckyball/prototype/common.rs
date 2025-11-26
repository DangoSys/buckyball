/// Common types for all balls
use crate::builtin::Wire;

pub const VECLANE: usize = 16;

#[derive(Clone, Default)]
pub struct SramReadReq {
    pub addr: u32,
    pub bank: u8,
}

#[derive(Clone, Default)]
pub struct SramReadResp {
    pub data: Vec<i8>,
}

#[derive(Clone, Default)]
pub struct SramWriteReq {
    pub addr: u32,
    pub bank: u8,
    pub data: Vec<i8>,
}

#[derive(Clone, Default)]
pub struct AccReadReq {
    pub addr: u32,
    pub bank: u8,
}

#[derive(Clone, Default)]
pub struct AccReadResp {
    pub data: Vec<i32>,
}

#[derive(Clone, Default)]
pub struct AccWriteReq {
    pub addr: u32,
    pub bank: u8,
    pub data: Vec<i32>,
}

pub struct MemoryInterface {
    pub sram_read_req: Wire<SramReadReq>,
    pub sram_read_resp: Wire<SramReadResp>,
    pub sram_write_req: Wire<SramWriteReq>,
    pub acc_read_req: Wire<AccReadReq>,
    pub acc_read_resp: Wire<AccReadResp>,
    pub acc_write_req: Wire<AccWriteReq>,
}

impl Default for MemoryInterface {
    fn default() -> Self {
        Self {
            sram_read_req: Wire::default(),
            sram_read_resp: Wire::default(),
            sram_write_req: Wire::default(),
            acc_read_req: Wire::default(),
            acc_read_resp: Wire::default(),
            acc_write_req: Wire::default(),
        }
    }
}

impl MemoryInterface {
    pub fn clear_requests(&mut self) {
        self.sram_read_req.clear();
        self.sram_write_req.clear();
        self.acc_read_req.clear();
        self.acc_write_req.clear();
    }
}
