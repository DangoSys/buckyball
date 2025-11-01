package prototype.vector.thread

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config._
import prototype.vector.bond.CanHaveVVVBond
import prototype.vector.op.{CascadeOp, CanHaveCascadeOp}

class CasThread(implicit p: Parameters) extends BaseThread
  with CanHaveCascadeOp
  with CanHaveVVVBond {

  // Connect CascadeOp and VVVBond
  for {
    op <- cascadeOp
    bond <- vvvBond
  } {
    op.io.in <> bond.in
    op.io.out <> bond.out
  }
}
