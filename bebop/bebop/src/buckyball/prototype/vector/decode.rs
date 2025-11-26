/// Vector Ball decoder
use crate::builtin::ball::BallCmdReq;
use super::isa::VecCmd;

pub fn decode(req: &BallCmdReq) -> VecCmd {
  VecCmd::from_fields(req.xs1, req.xs2)
}
