/// Message protocol definitions for Spike-Bebop communication
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

/// DMA write request from server (DMA path) - 32 bytes
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct DmaWriteReq {
  pub header: MsgHeader, // 8 bytes
  pub size: u32,         // 4 bytes
  pub padding: u32,      // 4 bytes
  pub addr: u64,         // 8 bytes
  pub data: u64,         // 8 bytes
}

impl DmaWriteReq {
  pub const SIZE: usize = 32;

  pub fn new(addr: u64, data: u64, size: u32) -> Self {
    Self {
      header: MsgHeader {
        msg_type: MsgType::DmaWriteReq as u32,
        reserved: 0,
      },
      size,
      padding: 0,
      addr,
      data,
    }
  }

  pub fn to_bytes(&self) -> [u8; Self::SIZE] {
    let mut bytes = [0u8; Self::SIZE];
    bytes[0..4].copy_from_slice(&self.header.msg_type.to_le_bytes());
    bytes[4..8].copy_from_slice(&self.header.reserved.to_le_bytes());
    bytes[8..12].copy_from_slice(&self.size.to_le_bytes());
    bytes[12..16].copy_from_slice(&self.padding.to_le_bytes());
    bytes[16..24].copy_from_slice(&self.addr.to_le_bytes());
    bytes[24..32].copy_from_slice(&self.data.to_le_bytes());
    bytes
  }
}

/// DMA write response from client (DMA path) - 16 bytes
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct DmaWriteResp {
  pub header: MsgHeader, // 8 bytes
  pub reserved: u64,     // 8 bytes
}

impl DmaWriteResp {
  pub const SIZE: usize = 16;

  pub fn from_bytes(bytes: &[u8; Self::SIZE]) -> Self {
    let msg_type = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    let reserved_header = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    let reserved = u64::from_le_bytes([
      bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15],
    ]);

    Self {
      header: MsgHeader {
        msg_type,
        reserved: reserved_header,
      },
      reserved,
    }
  }
}

// Backward compatibility aliases
pub type SocketMsg = CmdReq;
pub type SocketResp = CmdResp;
