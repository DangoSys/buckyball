/// Socket communication module for Spike-Bebop interface
mod dma_client;
mod handler;
mod protocol;
mod server;

pub use dma_client::DmaClient;
pub use protocol::{
  CmdReq, CmdResp, DmaReadReq, DmaReadResp, DmaWriteReq, DmaWriteResp, MsgType, SocketMsg, SocketResp,
};
pub use server::SocketServer;
