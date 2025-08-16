package prototype.vector.op

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config._
import prototype.vector.thread.{ThreadKey, ThreadOpKey, ThreadBondKey, BaseThread}
import prototype.vector.bond.VVV

class CascadeOp(implicit p: Parameters) extends Module {
  val lane = p(ThreadKey).get.lane
  val bondParam = p(ThreadBondKey).get
  val outputWidth = bondParam.outputWidth

  val io = IO(new VVV()(p))

  val reg1       = RegInit(VecInit(Seq.fill(lane)(0.U(outputWidth.W))))
  val reg2       = RegInit(VecInit(Seq.fill(lane)(0.U(outputWidth.W))))
  val valid1      = RegInit(false.B)
  val valid2      = RegInit(false.B)
  
  io.in.ready := io.out.ready
  
  when (io.in.valid) {
    valid1 := true.B
    reg1 := io.in.bits.in1.zip(io.in.bits.in2).map { case (a, b) => a + b }
  }.elsewhen(!io.in.ready){
    valid1 := valid1
  }.otherwise {
    valid1 := false.B
  }


  val valid = valid1 

  when (io.out.ready && valid) {
    io.out.valid := true.B
    io.out.bits.out := reg1
  }.otherwise {
    io.out.valid := false.B
    io.out.bits.out := VecInit(Seq.fill(lane)(0.U(outputWidth.W)))
  }
}

trait CanHaveCascadeOp { this: BaseThread =>
  val cascadeOp = params(ThreadOpKey).filter(_.OpType == "cascade").map { opParam =>
    // println(s"[CascadeOp] Creating OpType: ${opParam.OpType}")

    Module(new CascadeOp()(params))
  }

  def getCascadeOp = cascadeOp
}

