package prototype.vector.op

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config._
import prototype.vector.thread.{ThreadKey, ThreadOpKey, ThreadBondKey, BaseThread}
import prototype.vector.bond.VVV

class MulOp(implicit p: Parameters) extends Module {
  val lane = p(ThreadKey).get.lane
  val bondParam = p(ThreadBondKey).get
  val inputWidth = bondParam.inputWidth

  val io = IO(new VVV()(p))

  val reg1 = RegInit(VecInit(Seq.fill(lane)(0.U(inputWidth.W))))
  val reg2 = RegInit(VecInit(Seq.fill(lane)(0.U(inputWidth.W))))

  val cnt = RegInit(0.U(log2Ceil(lane).W))
  val active = RegInit(false.B)

  io.out.valid := active && io.out.ready
  io.in.ready := io.out.ready

  when (io.in.valid) {
    reg1 := io.in.bits.in1
    reg2 := io.in.bits.in2
    cnt := 0.U
    active := true.B
  } .elsewhen (active && io.out.ready) {
    cnt := cnt + 1.U
    when (cnt === (lane-1).U) {
      active := false.B
    }
  }

  for (i <- 0 until lane) {
    io.out.bits.out(i) := reg1(cnt) * reg2(i)
  }

}

trait CanHaveMulOp { this: BaseThread =>
  val mulOp = params(ThreadOpKey).filter(_.OpType == "mul").map { opParam =>
    // println(s"[MulOp] Creating OpType: ${opParam.OpType}")

    Module(new MulOp()(params))
  }

  def getMulOp = mulOp
}

