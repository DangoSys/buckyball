package prototype.vector.thread

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config._
import prototype.vector.bond.CanHaveVVVBond
import prototype.vector.op.{MulOp, CanHaveMulOp}

class MulThread(implicit p: Parameters) extends BaseThread
  with CanHaveMulOp
  with CanHaveVVVBond {

  // 连接MulOp和VVVBond
  for {
    op <- mulOp
    bond <- vvvBond
  } {
    op.io.in <> bond.in
    op.io.out <> bond.out
  }
}
