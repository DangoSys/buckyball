package framework.balldomain.prototype.gemmini

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.top.GlobalConfig
import framework.balldomain.blink.SubRobRow
import framework.frontend.decoder.PostGDCmd
import framework.frontend.scoreboard.BankAccessInfo
import framework.frontend.decoder.DomainId

/** Converts LoopCmd (logical) → SubRobRow (hardware PostGDCmd) for SubROB. */
@instantiable
class LoopCmdEncoder(val b: GlobalConfig) extends Module {
  val bankIdLen = log2Up(b.memDomain.bankNum)

  @public
  val io = IO(new Bundle {
    val cmd         = Flipped(Decoupled(new LoopCmd(b)))
    val subRobRow   = Decoupled(new SubRobRow(b))
    val ballId      = Input(UInt(log2Up(b.ballDomain.ballNum).W))
    val masterRobId = Input(UInt(log2Up(b.frontend.rob_entries).W))
  })

  // Passthrough handshake
  io.subRobRow.valid := io.cmd.valid
  io.cmd.ready       := io.subRobRow.ready

  io.subRobRow.bits.ball_id       := io.ballId
  io.subRobRow.bits.master_rob_id := io.masterRobId

  for (i <- 0 until 4) {
    val slot = io.subRobRow.bits.slots(i)
    val lsub = io.cmd.bits.slots(i)
    slot.valid := lsub.valid

    // Defaults — all unused RoCCCommandBB fields zero
    slot.cmd.domain_id    := 0.U
    slot.cmd.cmd.raw_inst := 0.U
    slot.cmd.cmd.pc       := 0.U
    slot.cmd.cmd.funct    := 0.U
    slot.cmd.cmd.funct3   := 0.U
    slot.cmd.cmd.rs2      := 0.U
    slot.cmd.cmd.rs1      := 0.U
    slot.cmd.cmd.xd       := false.B
    slot.cmd.cmd.xs1      := false.B
    slot.cmd.cmd.xs2      := false.B
    slot.cmd.cmd.rd       := 0.U
    slot.cmd.cmd.opcode   := 0.U
    slot.cmd.cmd.rs1Data  := 0.U
    slot.cmd.cmd.rs2Data  := 0.U
    slot.cmd.bankAccess   := BankAccessInfo.none(bankIdLen)
    slot.cmd.isFence      := false.B
    slot.cmd.isBarrier    := false.B

    when(lsub.valid) {
      switch(lsub.bits.cmdType) {
        is(LoopSubCmdType.MSET_ALLOC) {
          slot.cmd.domain_id   := DomainId.MEM
          slot.cmd.cmd.funct   := 0x20.U // MSET (enable=010, opcode=0)
          slot.cmd.cmd.rs1Data := lsub.bits.bank_id
          slot.cmd.cmd.rs2Data := lsub.bits.bank_row.asUInt |
            (lsub.bits.bank_col.asUInt << 5) |
            (1.U << 10) // alloc=1
          slot.cmd.bankAccess.wr_bank_valid := true.B
          slot.cmd.bankAccess.wr_bank_id    := lsub.bits.bank_id
        }
        is(LoopSubCmdType.MSET_FREE) {
          slot.cmd.domain_id                := DomainId.MEM
          slot.cmd.cmd.funct                := 0x20.U // MSET
          slot.cmd.cmd.rs1Data              := lsub.bits.bank_id
          slot.cmd.cmd.rs2Data              := 0.U    // alloc=0
          slot.cmd.bankAccess.wr_bank_valid := true.B
          slot.cmd.bankAccess.wr_bank_id    := lsub.bits.bank_id
        }
        is(LoopSubCmdType.MVIN) {
          slot.cmd.domain_id                := DomainId.MEM
          slot.cmd.cmd.funct                := 0x21.U // MVIN (enable=010, opcode=1)
          slot.cmd.cmd.rs1Data              := lsub.bits.bank_id | (lsub.bits.iter << 30)
          slot.cmd.cmd.rs2Data              := lsub.bits.dram_addr
          slot.cmd.bankAccess.wr_bank_valid := true.B
          slot.cmd.bankAccess.wr_bank_id    := lsub.bits.bank_id
        }
        is(LoopSubCmdType.MVOUT) {
          slot.cmd.domain_id                  := DomainId.MEM
          slot.cmd.cmd.funct                  := 0x10.U // MVOUT (enable=001, opcode=0)
          slot.cmd.cmd.rs1Data                := lsub.bits.bank_id | (lsub.bits.iter << 30)
          slot.cmd.cmd.rs2Data                := lsub.bits.dram_addr
          slot.cmd.bankAccess.rd_bank_0_valid := true.B
          slot.cmd.bankAccess.rd_bank_0_id    := lsub.bits.bank_id
        }
        is(LoopSubCmdType.PRELOAD) {
          slot.cmd.domain_id                  := DomainId.BALL
          slot.cmd.cmd.funct                  := 0x35.U // GEMMINI_PRELOAD (enable=011, opcode=5)
          slot.cmd.cmd.rs1Data                := lsub.bits.op1_bank |
            (lsub.bits.wr_bank << 20) |
            (lsub.bits.iter << 30)
          slot.cmd.cmd.rs2Data                := 1.U
          slot.cmd.bankAccess.rd_bank_0_valid := true.B
          slot.cmd.bankAccess.rd_bank_0_id    := lsub.bits.op1_bank
          slot.cmd.bankAccess.wr_bank_valid   := true.B
          slot.cmd.bankAccess.wr_bank_id      := lsub.bits.wr_bank
        }
        is(LoopSubCmdType.COMPUTE) {
          slot.cmd.domain_id := DomainId.BALL
          slot.cmd.cmd.funct := Mux(
            lsub.bits.compute_mode === 0.U,
            0x42.U, // COMPUTE_PRELOADED (enable=100, opcode=2)
            0x43.U  // COMPUTE_ACCUMULATED (enable=100, opcode=3)
          )
          slot.cmd.cmd.rs1Data                := lsub.bits.op1_bank |
            (lsub.bits.op2_bank << 10) |
            (lsub.bits.wr_bank << 20) |
            (lsub.bits.iter << 30)
          slot.cmd.cmd.rs2Data                := Mux(
            lsub.bits.compute_mode === 0.U,
            2.U,
            3.U
          )
          slot.cmd.bankAccess.rd_bank_0_valid := true.B
          slot.cmd.bankAccess.rd_bank_0_id    := lsub.bits.op1_bank
          slot.cmd.bankAccess.rd_bank_1_valid := true.B
          slot.cmd.bankAccess.rd_bank_1_id    := lsub.bits.op2_bank
          slot.cmd.bankAccess.wr_bank_valid   := true.B
          slot.cmd.bankAccess.wr_bank_id      := lsub.bits.wr_bank
        }
      }
    }
  }
}
