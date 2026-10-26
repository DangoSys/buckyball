package framework.core.execution.instruction

import chisel3._
import chisel3.util._
import framework.ip.bus.chi.rnf.CacheAtomic
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.core.execution.instruction.configs.{Params => ModuleParam}

/** Expands RV64C instructions and decodes their fields and immediate operand. */
class InstructionIO(b: ModuleParam) extends Bundle {
  val p = b

  val bits             = Input(UInt(32.W))
  val expanded         = Output(UInt(32.W))
  val compressed       = Output(Bool())
  val illegal          = Output(Bool())
  val opcode           = Output(UInt(7.W))
  val funct3           = Output(UInt(3.W))
  val funct7           = Output(UInt(7.W))
  val rs1              = Output(UInt(5.W))
  val rs2              = Output(UInt(5.W))
  val rd               = Output(UInt(5.W))
  val csrAddress       = Output(UInt(12.W))
  val immediate        = Output(UInt(p.xLen.W))
  val atomicCode       = Output(UInt(4.W))
  val integerFunction  = Output(UInt(5.W))
  val integerWord      = Output(Bool())
  val integerImmediate = Output(Bool())

}

@instantiable
class Instruction(b: ModuleParam) extends Module {
  val p = b

  @public
  val io = IO(new InstructionIO(b))

  val compressedIllegal = WireDefault(false.B)
  val c                 = io.bits(15, 0)
  val rd                = c(11, 7)
  val rs2               = c(6, 2)
  val compactRd         = Cat(1.U(2.W), c(4, 2))
  val compactRs1        = Cat(1.U(2.W), c(9, 7))
  val signedImmediate   = Cat(Fill(7, c(12)), c(6, 2))
  val loadWordOffset    = Cat(0.U(5.W), c(5), c(12, 10), c(6), 0.U(2.W))
  val loadDoubleOffset  = Cat(0.U(4.W), c(6, 5), c(12, 10), 0.U(3.W))
  val jumpOffset        = Cat(Fill(10, c(12)), c(12), c(8), c(10, 9), c(6), c(7), c(2), c(11), c(5, 3), 0.U(1.W))
  val branchOffset      = Cat(Fill(5, c(12)), c(6, 5), c(2), c(11, 10), c(4, 3), 0.U(1.W))
  val addSpOffset       = Cat(Fill(3, c(12)), c(4, 3), c(5), c(2), c(6), 0.U(4.W))
  val addSpUnsigned     = Cat(0.U(2.W), c(10, 7), c(12, 11), c(5), c(6), 0.U(2.W))
  val stackWordLoad     = Cat(0.U(4.W), c(3, 2), c(12), c(6, 4), 0.U(2.W))
  val stackDoubleLoad   = Cat(0.U(3.W), c(4, 2), c(12), c(6, 5), 0.U(3.W))
  val stackWordStore    = Cat(0.U(4.W), c(8, 7), c(12, 9), 0.U(2.W))
  val stackDoubleStore  = Cat(0.U(3.W), c(9, 7), c(12, 10), 0.U(3.W))

  io.compressed     := c(1, 0) =/= 3.U
  io.expanded       := io.bits
  compressedIllegal := false.B
  when(io.compressed) {
    io.expanded       := "h00000013".U
    compressedIllegal := true.B
    switch(c(1, 0)) {
      is(0.U) {
        switch(c(15, 13)) {
          is(0.U) { // C.ADDI4SPN
            io.expanded       := Cat(addSpUnsigned, 2.U(5.W), 0.U(3.W), compactRd, "h13".U(7.W))
            compressedIllegal := !addSpUnsigned.orR
          }
          is(1.U, 2.U, 3.U) { // C.FLD, C.LW, C.LD
            val word = c(15, 13) === 2.U
            io.expanded       := Cat(
              Mux(word, loadWordOffset, loadDoubleOffset),
              compactRs1,
              Mux(word, 2.U(3.W), 3.U(3.W)),
              compactRd,
              Mux(c(15, 13) === 1.U, "h07".U(7.W), "h03".U(7.W))
            )
            compressedIllegal := false.B
          }
          is(5.U, 6.U, 7.U) { // C.FSD, C.SW, C.SD
            val word   = c(15, 13) === 6.U
            val offset = Mux(word, loadWordOffset, loadDoubleOffset)
            io.expanded       := Cat(
              offset(11, 5),
              compactRd,
              compactRs1,
              Mux(word, 2.U(3.W), 3.U(3.W)),
              offset(4, 0),
              Mux(c(15, 13) === 5.U, "h27".U(7.W), "h23".U(7.W))
            )
            compressedIllegal := false.B
          }
        }
      }
      is(1.U) {
        switch(c(15, 13)) {
          is(0.U, 1.U) { // C.ADDI/NOP and C.ADDIW
            io.expanded       := Cat(signedImmediate, rd, 0.U(3.W), rd, Mux(c(13), "h1b".U(7.W), "h13".U(7.W)))
            compressedIllegal := c(13) && rd === 0.U
          }
          is(2.U) { // C.LI, including hints
            io.expanded       := Cat(signedImmediate, 0.U(5.W), 0.U(3.W), rd, "h13".U(7.W))
            compressedIllegal := false.B
          }
          is(3.U) { // C.ADDI16SP / C.LUI
            io.expanded       := Mux(
              rd === 2.U,
              Cat(addSpOffset, 2.U(5.W), 0.U(3.W), 2.U(5.W), "h13".U(7.W)),
              Cat(Fill(15, c(12)), c(6, 2), rd, "h37".U(7.W))
            )
            compressedIllegal := !Cat(c(12), c(6, 2)).orR
          }
          is(4.U) {
            compressedIllegal := false.B
            switch(c(11, 10)) {
              is(0.U, 1.U) { // C.SRLI / C.SRAI
                io.expanded := Cat(
                  0.U(1.W),
                  c(10),
                  0.U(4.W),
                  c(12),
                  c(6, 2),
                  compactRs1,
                  5.U(3.W),
                  compactRs1,
                  "h13".U(7.W)
                )
              }
              is(2.U) { // C.ANDI
                io.expanded := Cat(signedImmediate, compactRs1, 7.U(3.W), compactRs1, "h13".U(7.W))
              }
              is(3.U) {
                val sub   = c(6, 5) === 0.U
                val funct = MuxLookup(c(6, 5), 0.U(3.W))(Seq(0.U -> 0.U, 1.U -> 4.U, 2.U -> 6.U, 3.U -> 7.U))
                io.expanded       := Cat(
                  Mux(sub, 32.U(7.W), 0.U(7.W)),
                  compactRd,
                  compactRs1,
                  Mux(c(12), 0.U(3.W), funct),
                  compactRs1,
                  Mux(c(12), "h3b".U(7.W), "h33".U(7.W))
                )
                compressedIllegal := c(12) && c(6, 5) >= 2.U
              }
            }
          }
          is(5.U) { // C.J
            io.expanded       := Cat(
              jumpOffset(20),
              jumpOffset(10, 1),
              jumpOffset(11),
              jumpOffset(19, 12),
              0.U(5.W),
              "h6f".U(7.W)
            )
            compressedIllegal := false.B
          }
          is(6.U, 7.U) { // C.BEQZ / C.BNEZ
            io.expanded       := Cat(
              branchOffset(12),
              branchOffset(10, 5),
              0.U(5.W),
              compactRs1,
              Cat(0.U(2.W), c(13)),
              branchOffset(4, 1),
              branchOffset(11),
              "h63".U(7.W)
            )
            compressedIllegal := false.B
          }
        }
      }
      is(2.U) {
        switch(c(15, 13)) {
          is(0.U) { // C.SLLI, including hints
            io.expanded       := Cat(0.U(6.W), c(12), c(6, 2), rd, 1.U(3.W), rd, "h13".U(7.W))
            compressedIllegal := false.B
          }
          is(1.U, 2.U, 3.U) { // C.FLDSP / C.LWSP / C.LDSP
            val word = c(15, 13) === 2.U
            io.expanded       := Cat(
              Mux(word, stackWordLoad, stackDoubleLoad),
              2.U(5.W),
              Mux(word, 2.U(3.W), 3.U(3.W)),
              rd,
              Mux(c(15, 13) === 1.U, "h07".U(7.W), "h03".U(7.W))
            )
            compressedIllegal := c(15, 13) =/= 1.U && rd === 0.U
          }
          is(4.U) {
            compressedIllegal := false.B
            when(rs2 =/= 0.U) { // C.MV / C.ADD, including hints
              io.expanded := Cat(0.U(7.W), rs2, Mux(c(12), rd, 0.U(5.W)), 0.U(3.W), rd, "h33".U(7.W))
            }.elsewhen(c(12) && rd === 0.U) {
              io.expanded := "h00100073".U // C.EBREAK
            }.otherwise { // C.JR / C.JALR
              io.expanded       := Cat(0.U(12.W), rd, 0.U(3.W), Mux(c(12), 1.U(5.W), 0.U(5.W)), "h67".U(7.W))
              compressedIllegal := rd === 0.U
            }
          }
          is(5.U, 6.U, 7.U) { // C.FSDSP / C.SWSP / C.SDSP
            val word   = c(15, 13) === 6.U
            val offset = Mux(word, stackWordStore, stackDoubleStore)
            io.expanded       := Cat(
              offset(11, 5),
              rs2,
              2.U(5.W),
              Mux(word, 2.U(3.W), 3.U(3.W)),
              offset(4, 0),
              Mux(c(15, 13) === 5.U, "h27".U(7.W), "h23".U(7.W))
            )
            compressedIllegal := false.B
          }
        }
      }
    }
  }

  val insn = io.expanded
  io.opcode     := insn(6, 0)
  io.funct3     := insn(14, 12)
  io.funct7     := insn(31, 25)
  io.rs1        := insn(19, 15)
  io.rs2        := insn(24, 20)
  io.rd         := insn(11, 7)
  io.csrAddress := insn(31, 20)

  val immediateI = Cat(Fill(p.xLen - 12, insn(31)), insn(31, 20))
  val immediateS = Cat(Fill(p.xLen - 12, insn(31)), insn(31, 25), insn(11, 7))
  val immediateB = Cat(Fill(p.xLen - 12, insn(31)), insn(7), insn(30, 25), insn(11, 8), 0.U(1.W))
  val immediateJ = Cat(Fill(p.xLen - 20, insn(31)), insn(19, 12), insn(20), insn(30, 21), 0.U(1.W))
  val immediateU = Cat(Fill(p.xLen - 32, insn(31)), insn(31, 12), 0.U(12.W))
  io.immediate := MuxLookup(io.opcode, immediateI)(Seq(
    "h23".U -> immediateS,
    "h27".U -> immediateS,
    "h63".U -> immediateB,
    "h6f".U -> immediateJ,
    "h17".U -> immediateU,
    "h37".U -> immediateU
  ))

  io.atomicCode       := MuxLookup(io.funct7(6, 2), CacheAtomic.None.U(4.W))(Seq(
    0.U  -> CacheAtomic.Add.U,
    1.U  -> CacheAtomic.Swap.U,
    2.U  -> CacheAtomic.LR.U,
    3.U  -> CacheAtomic.SC.U,
    4.U  -> CacheAtomic.Xor.U,
    8.U  -> CacheAtomic.Or.U,
    12.U -> CacheAtomic.And.U,
    16.U -> CacheAtomic.Min.U,
    20.U -> CacheAtomic.Max.U,
    24.U -> CacheAtomic.MinU.U,
    28.U -> CacheAtomic.MaxU.U
  ))
  io.integerWord      := io.opcode === "h1b".U || io.opcode === "h3b".U
  io.integerImmediate := io.opcode === "h13".U || io.opcode === "h1b".U
  val subtract = (io.opcode === "h33".U || io.opcode === "h3b".U) && io.funct7(5) && io.funct3 === 0.U
  io.integerFunction := MuxLookup(io.funct3, 0.U(5.W))(Seq(
    0.U -> Mux(subtract, 10.U, 0.U),
    1.U -> 1.U,
    2.U -> 12.U,
    3.U -> 14.U,
    4.U -> 4.U,
    5.U -> Mux(io.funct7(5), 11.U, 5.U),
    6.U -> 6.U,
    7.U -> 7.U
  ))

  val shift = io.funct3 === 1.U || io.funct3 === 5.U

  val immediateLegal = Mux(io.integerWord, io.funct3 === 0.U || shift, true.B) &&
    (!shift || Mux(io.integerWord, io.funct7 === 0.U, io.funct7(6, 1) === 0.U) ||
      (io.funct3 === 5.U && Mux(io.integerWord, io.funct7 === 32.U, io.funct7(6, 1) === 16.U)))

  val registerLegal = Mux(
    io.funct7 === 1.U,
    !io.integerWord || io.funct3 === 0.U || io.funct3 >= 4.U,
    (io.funct7 === 0.U || (io.funct7 === 32.U && (io.funct3 === 0.U || io.funct3 === 5.U))) &&
      (!io.integerWord || io.funct3 === 0.U || io.funct3 === 1.U || io.funct3 === 5.U)
  )

  val systemLegal = Seq("h00000073", "h00100073", "h30200073", "h10200073", "h10500073")
    .map(code => insn === code.U(32.W)).reduce(_ || _) || (io.funct7 === "h09".U && io.rd === 0.U)

  val encodingLegal = MuxLookup(io.opcode, false.B)(Seq(
    "h37".U -> true.B,
    "h17".U -> true.B,
    "h6f".U -> true.B,
    "h67".U -> (io.funct3 === 0.U),
    "h63".U -> (io.funct3 =/= 2.U && io.funct3 =/= 3.U),
    "h13".U -> immediateLegal,
    "h1b".U -> immediateLegal,
    "h33".U -> registerLegal,
    "h3b".U -> registerLegal,
    "h03".U -> (io.funct3 =/= 7.U),
    "h23".U -> (io.funct3 <= 3.U),
    "h07".U -> (io.funct3 === 2.U || io.funct3 === 3.U),
    "h27".U -> (io.funct3 === 2.U || io.funct3 === 3.U),
    "h2f".U -> ((io.funct3 === 2.U || io.funct3 === 3.U) && io.atomicCode =/= CacheAtomic.None.U &&
      (io.funct7(6, 2) =/= 2.U || io.rs2 === 0.U)),
    "h43".U -> true.B,
    "h47".U -> true.B,
    "h4b".U -> true.B,
    "h4f".U -> true.B,
    "h53".U -> true.B,
    "h0f".U -> (io.funct3 === 0.U || io.funct3 === 1.U),
    "h73".U -> Mux(io.funct3 =/= 0.U, io.funct3 =/= 4.U, systemLegal),
    "h2b".U -> (io.funct3 === 7.U),
    "h7b".U -> true.B
  ))

  io.illegal := compressedIllegal || !encodingLegal

}
