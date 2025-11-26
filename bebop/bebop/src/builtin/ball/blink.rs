/// Blink - Standard interface for Ball devices
/// Mirrors RTL: framework/blink/blink.scala

/// Ball device status
#[derive(Clone, Default)]
pub struct Status {
  pub ready: bool,
  pub valid: bool,
  pub idle: bool,
  pub running: bool,
  pub complete: bool,
  pub iter: u32,
}

/// Command request from RS - raw fields for ball-specific decoding
#[derive(Clone, Default)]
pub struct BallCmdReq {
  pub valid: bool,
  pub rob_id: u32,
  pub bid: u8,
  pub funct: u8,
  pub xs1: u64,  // Raw xs1 for ISA decoding
  pub xs2: u64,  // Raw xs2 for ISA decoding
}

/// Command response to RS
#[derive(Clone, Default)]
pub struct BallCmdResp {
  pub valid: bool,
  pub rob_id: u32,
}

/// SRAM read request
#[derive(Clone, Default)]
pub struct SramReadReq {
  pub valid: bool,
  pub addr: u32,
  pub bank: u8,
}

/// SRAM read response
#[derive(Clone, Default)]
pub struct SramReadResp {
  pub valid: bool,
  pub data: Vec<i8>,
}

/// SRAM write request
#[derive(Clone, Default)]
pub struct SramWriteReq {
  pub valid: bool,
  pub addr: u32,
  pub bank: u8,
  pub data: Vec<i8>,
}

/// Accumulator read/write (i32 data)
#[derive(Clone, Default)]
pub struct AccReadReq {
  pub valid: bool,
  pub addr: u32,
  pub bank: u8,
}

#[derive(Clone, Default)]
pub struct AccReadResp {
  pub valid: bool,
  pub data: Vec<i32>,
}

#[derive(Clone, Default)]
pub struct AccWriteReq {
  pub valid: bool,
  pub addr: u32,
  pub bank: u8,
  pub data: Vec<i32>,
}
