/// Ball trait - Base trait for all Ball devices
/// Mirrors RTL: framework/blink/baseball.scala
use super::blink::*;

/// Blink interface bundle for a Ball
pub struct Blink {
  pub cmd_req: BallCmdReq,
  pub cmd_resp: BallCmdResp,
  pub sram_read_req: Vec<SramReadReq>,
  pub sram_read_resp: Vec<SramReadResp>,
  pub sram_write_req: Vec<SramWriteReq>,
  pub acc_read_req: Vec<AccReadReq>,
  pub acc_read_resp: Vec<AccReadResp>,
  pub acc_write_req: Vec<AccWriteReq>,
  pub status: Status,
}

impl Blink {
  pub fn new(num_sp_banks: usize, num_acc_banks: usize) -> Self {
    Self {
      cmd_req: BallCmdReq::default(),
      cmd_resp: BallCmdResp::default(),
      sram_read_req: vec![SramReadReq::default(); num_sp_banks],
      sram_read_resp: vec![SramReadResp::default(); num_sp_banks],
      sram_write_req: vec![SramWriteReq::default(); num_sp_banks],
      acc_read_req: vec![AccReadReq::default(); num_acc_banks],
      acc_read_resp: vec![AccReadResp::default(); num_acc_banks],
      acc_write_req: vec![AccWriteReq::default(); num_acc_banks],
      status: Status::default(),
    }
  }

  pub fn clear_requests(&mut self) {
    for req in &mut self.sram_read_req {
      req.valid = false;
    }
    for req in &mut self.sram_write_req {
      req.valid = false;
    }
    for req in &mut self.acc_read_req {
      req.valid = false;
    }
    for req in &mut self.acc_write_req {
      req.valid = false;
    }
    self.cmd_resp.valid = false;
  }
}

/// Ball trait - all balls must implement this
pub trait Ball {
  fn ball_id(&self) -> u8;
  fn blink(&self) -> &Blink;
  fn blink_mut(&mut self) -> &mut Blink;
  fn tick(&mut self);
  fn reset(&mut self);
}
