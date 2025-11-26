/// Matrix Ball decoder
use crate::builtin::ball::BallCmdReq;
use super::isa::MatrixCmd;

pub fn decode(req: &BallCmdReq) -> MatrixCmd {
  MatrixCmd::from_fields(req.xs1, req.xs2)
}
