package examples.balls.vector

import chisel3._
import chisel3.util._
import framework.balldomain.blink.BankRead
import framework.top.GlobalConfig

class MulWarp16AccLoad(val b: GlobalConfig, lane: Int, accWidth: Int) extends Module {
  val io = IO(new Bundle {
    val start    = Input(Bool())
    val wrBank   = Input(UInt(log2Up(b.memDomain.bankNum).W))
    val robId    = Input(UInt(log2Up(b.frontend.rob_entries).W))
    val bankRead = Vec(2, Flipped(new BankRead(b)))
    val acc      = Output(Vec(lane, Vec(lane, UInt(accWidth.W))))
    val done     = Output(Bool())
    val busy     = Output(Bool())
  })

  val sIdle :: sReq :: sResp :: Nil = Enum(3)
  val state = RegInit(sIdle)

  val loadRow   = RegInit(0.U(log2Ceil(lane + 1).W))
  val loadPhase = RegInit(0.U(1.W))
  val reqDone   = RegInit(VecInit(Seq.fill(2)(false.B)))
  val respDone  = RegInit(VecInit(Seq.fill(2)(false.B)))
  val acc       = Reg(Vec(lane, Vec(lane, UInt(accWidth.W))))

  def loadGroup(port: Int): UInt = Cat(loadPhase, port.U(1.W))
  def clr2() = VecInit(Seq.fill(2)(false.B))

  io.acc  := acc
  io.done := false.B
  io.busy := state =/= sIdle

  for (i <- 0 until 2) {
    io.bankRead(i).rob_id           := io.robId
    io.bankRead(i).ball_id          := 0.U
    io.bankRead(i).bank_id          := 0.U
    io.bankRead(i).group_id         := 0.U
    io.bankRead(i).io.req.valid     := false.B
    io.bankRead(i).io.req.bits.addr := 0.U
    io.bankRead(i).io.resp.ready    := false.B
  }

  switch(state) {
    is(sIdle) {
      when(io.start) {
        loadRow   := 0.U
        loadPhase := 0.U
        reqDone   := clr2()
        respDone  := clr2()
        state     := sReq
      }
    }

    is(sReq) {
      for (p <- 0 until 2) {
        io.bankRead(p).bank_id          := io.wrBank
        io.bankRead(p).group_id         := loadGroup(p)
        io.bankRead(p).io.req.valid     := !reqDone(p)
        io.bankRead(p).io.req.bits.addr := loadRow
        when(io.bankRead(p).io.req.fire) {
          reqDone(p) := true.B
        }
      }
      when((0 until 2).map(p => reqDone(p) || io.bankRead(p).io.req.fire).reduce(_ && _)) {
        respDone := clr2()
        state    := sResp
      }
    }

    is(sResp) {
      for (p <- 0 until 2) {
        io.bankRead(p).bank_id       := io.wrBank
        io.bankRead(p).group_id      := loadGroup(p)
        io.bankRead(p).io.resp.ready := !respDone(p)
        when(io.bankRead(p).io.resp.fire) {
          val g     = loadGroup(p)
          val elems = io.bankRead(p).io.resp.bits.data.asTypeOf(Vec(4, UInt(accWidth.W)))
          val row   = loadRow(log2Ceil(lane) - 1, 0)
          for (i <- 0 until 4) {
            acc(row)(g * 4.U + i.U) := elems(i)
          }
          respDone(p) := true.B
        }
      }
      when((0 until 2).map(p => respDone(p) || io.bankRead(p).io.resp.fire).reduce(_ && _)) {
        when(loadPhase === 0.U) {
          loadPhase := 1.U
          reqDone   := clr2()
          respDone  := clr2()
          state     := sReq
        }.elsewhen(loadRow === (lane - 1).U) {
          io.done := true.B
          state   := sIdle
        }.otherwise {
          loadRow   := loadRow + 1.U
          loadPhase := 0.U
          reqDone   := clr2()
          respDone  := clr2()
          state     := sReq
        }
      }
    }
  }
}
