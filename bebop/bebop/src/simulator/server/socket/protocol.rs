/// Message protocol definitions for host-Bebop communication
/// Matches the C++ structures in customext/include/socket.h

/// Message types for socket communication
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MsgType {
  CmdReq = 0,       // Command request from client
  CmdResp = 1,      // Command response from server
  DmaReadReq = 2,   // DMA read request from server
  DmaReadResp = 3,  // DMA read response from client
  DmaWriteReq = 4,  // DMA write request from server
  DmaWriteResp = 5, // DMA write response from client
}

impl MsgType {
  pub fn from_u32(value: u32) -> Option<Self> {
    match value {
      0 => Some(MsgType::CmdReq),
      1 => Some(MsgType::CmdResp),
      2 => Some(MsgType::DmaReadReq),
      3 => Some(MsgType::DmaReadResp),
      4 => Some(MsgType::DmaWriteReq),
      5 => Some(MsgType::DmaWriteResp),
      _ => None,
    }
  }
}

/// Common message header (8 bytes)
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct MsgHeader {
  pub msg_type: u32,
  pub reserved: u32,
}
