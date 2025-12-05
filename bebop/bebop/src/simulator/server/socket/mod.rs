/// Socket communication module for host-Bebop interface
pub mod cmd;
pub mod protocol;
pub mod read;
pub mod write;

pub use cmd::{CmdReq, CmdResp, SocketMsg, SocketResp};
pub use protocol::MsgType;
