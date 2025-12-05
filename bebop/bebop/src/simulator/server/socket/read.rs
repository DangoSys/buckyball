/// DMA read protocol definitions
use super::protocol::{MsgHeader, MsgType};

/// DMA read request from server (DMA path) - 24 bytes
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct DmaReadReq {
  pub header: MsgHeader, // 8 bytes
  pub size: u32,         // 4 bytes
  pub padding: u32,      // 4 bytes
  pub addr: u64,         // 8 bytes
}

impl DmaReadReq {
  pub const SIZE: usize = 24;

  pub fn new(addr: u64, size: u32) -> Self {
    Self {
      header: MsgHeader {
        msg_type: MsgType::DmaReadReq as u32,
        reserved: 0,
      },
      size,
      padding: 0,
      addr,
    }
  }

  pub fn to_bytes(&self) -> [u8; Self::SIZE] {
    let mut bytes = [0u8; Self::SIZE];
    bytes[0..4].copy_from_slice(&self.header.msg_type.to_le_bytes());
    bytes[4..8].copy_from_slice(&self.header.reserved.to_le_bytes());
    bytes[8..12].copy_from_slice(&self.size.to_le_bytes());
    bytes[12..16].copy_from_slice(&self.padding.to_le_bytes());
    bytes[16..24].copy_from_slice(&self.addr.to_le_bytes());
    bytes
  }
}

/// DMA read response from client (DMA path) - 16 bytes
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct DmaReadResp {
  pub header: MsgHeader, // 8 bytes
  pub data: u64,         // 8 bytes
}

impl DmaReadResp {
  pub const SIZE: usize = 16;

  pub fn from_bytes(bytes: &[u8; Self::SIZE]) -> Self {
    let msg_type = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    let reserved = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    let data = u64::from_le_bytes([
      bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15],
    ]);

    Self {
      header: MsgHeader { msg_type, reserved },
      data,
    }
  }
}
