/// CMD protocol definitions
use super::protocol::{MsgHeader, MsgType};

/// Command request from client (CMD path) - 24 bytes
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct CmdReq {
  pub header: MsgHeader, // 8 bytes
  pub funct: u32,        // 4 bytes
  pub padding: u32,      // 4 bytes
  pub xs1: u64,          // 8 bytes
  pub xs2: u64,          // 8 bytes
}

impl CmdReq {
  pub const SIZE: usize = 32; // 8 + 4 + 4 + 8 + 8

  pub fn from_bytes(bytes: &[u8; Self::SIZE]) -> Self {
    let msg_type = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    let reserved = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    let funct = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
    let padding = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);
    let xs1 = u64::from_le_bytes([
      bytes[16], bytes[17], bytes[18], bytes[19], bytes[20], bytes[21], bytes[22], bytes[23],
    ]);
    let xs2 = u64::from_le_bytes([
      bytes[24], bytes[25], bytes[26], bytes[27], bytes[28], bytes[29], bytes[30], bytes[31],
    ]);

    Self {
      header: MsgHeader { msg_type, reserved },
      funct,
      padding,
      xs1,
      xs2,
    }
  }
}

/// Command response from server (CMD path) - 16 bytes
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct CmdResp {
  pub header: MsgHeader, // 8 bytes
  pub result: u64,       // 8 bytes
}

impl CmdResp {
  pub const SIZE: usize = 16;

  pub fn new(result: u64) -> Self {
    Self {
      header: MsgHeader {
        msg_type: MsgType::CmdResp as u32,
        reserved: 0,
      },
      result,
    }
  }

  pub fn to_bytes(&self) -> [u8; Self::SIZE] {
    let mut bytes = [0u8; Self::SIZE];
    bytes[0..4].copy_from_slice(&self.header.msg_type.to_le_bytes());
    bytes[4..8].copy_from_slice(&self.header.reserved.to_le_bytes());
    bytes[8..16].copy_from_slice(&self.result.to_le_bytes());
    bytes
  }
}

// Backward compatibility aliases
pub type SocketMsg = CmdReq;
pub type SocketResp = CmdResp;
