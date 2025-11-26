/// ReLU Ball decoder
use crate::builtin::ball::BallCmdReq;
use super::isa::ReluCmd;

pub fn decode(req: &BallCmdReq) -> ReluCmd {
  ReluCmd::from_fields(req.xs1, req.xs2)
}
