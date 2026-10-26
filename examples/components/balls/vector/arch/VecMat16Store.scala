package examples.balls.vector

import chisel3.experimental.hierarchy.{instantiable, public}

import framework.acc.balldomain.configs.BallDomainParam
import framework.acc.frontend.configs.FrontendParam
import framework.acc.memdomain.configs.MemDomainParam

import examples.balls.vector.configs.VectorBallParam

import chisel3._
import chisel3.util._
import framework.acc.memdomain.BankWrite

@instantiable
class VecMat16Store(
  frontendParam: FrontendParam,
  memParam:      MemDomainParam,
  routingParam:  BallDomainParam,
  val accWidth:  Int,
  lane:          Int,
  val ballParam: VectorBallParam)
    extends Module {

  @public
  val io = IO(new Bundle {
    val start  = Input(Bool())
    val wrBank = Input(UInt(frontendParam.bank_id_len.W))
    val robId  = Input(UInt(log2Up(frontendParam.rob_entries).W))
    val acc    = Input(Vec(lane, Vec(lane, UInt(accWidth.W))))

    val bankWrite = Vec(
      4,
      Flipped(new BankWrite(
        frontendParam,
        memParam
      ))
    )

    val done = Output(Bool())
    val busy = Output(Bool())
  })

  val sIdle :: sReq :: sResp :: Nil = Enum(3)
  val state                         = RegInit(sIdle)

  val writeRow    = RegInit(0.U(log2Ceil(lane + 1).W))
  val writeIssued = RegInit(VecInit(Seq.fill(4)(false.B)))
  val writeAcked  = RegInit(VecInit(Seq.fill(4)(false.B)))

  def writeData(group: Int): UInt = {
    val row   = writeRow(log2Ceil(lane) - 1, 0)
    val elems = (0 until 4).map(i => io.acc(row)(group * 4 + i))
    Cat(elems.reverse)
  }

  def clr4() = VecInit(Seq.fill(4)(false.B))

  io.done := false.B
  io.busy := state =/= sIdle

  for (i <- 0 until 4) {
    io.bankWrite(i).rob_id           := io.robId
    io.bankWrite(i).bank_id          := io.wrBank
    io.bankWrite(i).group_id         := i.U
    io.bankWrite(i).io.req.valid     := false.B
    io.bankWrite(i).io.req.bits.addr := 0.U
    io.bankWrite(i).io.req.bits.data := 0.U
    io.bankWrite(i).io.req.bits.mask := VecInit(
      Seq.fill(memParam.bankMaskLen)(false.B)
    )
    io.bankWrite(i).io.resp.ready    := false.B
  }

  switch(state) {
    is(sIdle) {
      when(io.start) {
        writeRow    := 0.U
        writeIssued := clr4()
        writeAcked  := clr4()
        state       := sReq
      }
    }

    is(sReq) {
      for (g <- 0 until 4) {
        io.bankWrite(g).bank_id          := io.wrBank
        io.bankWrite(g).group_id         := g.U
        io.bankWrite(g).io.req.valid     := !writeIssued(g)
        io.bankWrite(g).io.req.bits.addr := writeRow
        io.bankWrite(g).io.req.bits.data := writeData(g)
        io.bankWrite(g).io.req.bits.mask := VecInit(
          Seq.fill(memParam.bankMaskLen)(true.B)
        )
        when(io.bankWrite(g).io.req.fire) {
          writeIssued(g) := true.B
        }
      }
      when(
        (0 until 4)
          .map(g => writeIssued(g) || io.bankWrite(g).io.req.fire)
          .reduce(_ && _)
      ) {
        writeAcked := clr4()
        state      := sResp
      }
    }

    is(sResp) {
      for (g <- 0 until 4) {
        io.bankWrite(g).bank_id       := io.wrBank
        io.bankWrite(g).group_id      := g.U
        io.bankWrite(g).io.resp.ready := !writeAcked(g)
        when(io.bankWrite(g).io.resp.fire) {
          writeAcked(g) := true.B
        }
      }
      when(
        (0 until 4)
          .map(g => writeAcked(g) || io.bankWrite(g).io.resp.fire)
          .reduce(_ && _)
      ) {
        when(writeRow === (lane - 1).U) {
          io.done := true.B
          state   := sIdle
        }.otherwise {
          writeRow    := writeRow + 1.U
          writeIssued := clr4()
          writeAcked  := clr4()
          state       := sReq
        }
      }
    }
  }
}
