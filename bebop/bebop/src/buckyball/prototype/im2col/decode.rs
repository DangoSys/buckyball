/// Im2col Ball decoder
use crate::builtin::ball::BallCmdReq;
use super::isa::Im2colCmd;

pub fn decode(req: &BallCmdReq) -> Im2colCmd {
  Im2colCmd::from_fields(req.xs1, req.xs2)
}
