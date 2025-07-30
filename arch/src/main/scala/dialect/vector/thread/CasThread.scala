package dialect.vector.thread

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config._
import dialect.vector.bond.CanHaveVVVBond
import dialect.vector.op.{CascadeOp, CanHaveCascadeOp}  

class CasThread(implicit p: Parameters) extends BaseThread
  with CanHaveCascadeOp
  with CanHaveVVVBond {

  // 连接CascadeOp和VVVBond
  for {
    op <- cascadeOp
    bond <- vvvBond
  } {
    op.io.in <> bond.in
    op.io.out <> bond.out
  }
}