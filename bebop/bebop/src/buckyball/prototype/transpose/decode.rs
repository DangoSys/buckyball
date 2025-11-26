/// Transpose Ball decoder
use crate::builtin::ball::BallCmdReq;
use super::isa::TransposeCmd;

pub fn decode(req: &BallCmdReq) -> TransposeCmd {
  TransposeCmd::from_fields(req.xs1, req.xs2)
}
