package framework.rvv

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}

@instantiable
class ScalarExecution extends Module {

  @public
  val io = IO(new Bundle {
    val instruction = Input(UInt(32.W))
    val pc          = Input(UInt(32.W))
    val source1     = Input(UInt(32.W))
    val source2     = Input(UInt(32.W))
    val legal       = Output(Bool())
    val write       = Output(Bool())
    val destination = Output(UInt(5.W))
    val result      = Output(UInt(32.W))
    val nextPc      = Output(UInt(32.W))
  })

  val opcode = io.instruction(6, 0)
  val funct3 = io.instruction(14, 12)
  val funct7 = io.instruction(31, 25)
  val rs2    = io.instruction(24, 20)
  val immI   = Cat(Fill(20, io.instruction(31)), io.instruction(31, 20))

  val immB = Cat(
    Fill(19, io.instruction(31)),
    io.instruction(31),
    io.instruction(7),
    io.instruction(30, 25),
    io.instruction(11, 8),
    0.U(1.W)
  )

  val immJ = Cat(
    Fill(11, io.instruction(31)),
    io.instruction(31),
    io.instruction(19, 12),
    io.instruction(20),
    io.instruction(30, 21),
    0.U(1.W)
  )

  io.legal       := false.B
  io.write       := false.B
  io.destination := io.instruction(11, 7)
  io.result      := 0.U
  io.nextPc      := io.pc + 4.U

  switch(opcode) {
    is("h37".U) {
      io.legal  := true.B
      io.write  := true.B
      io.result := Cat(io.instruction(31, 12), 0.U(12.W))
    }
    is("h17".U) {
      io.legal  := true.B
      io.write  := true.B
      io.result := io.pc + Cat(io.instruction(31, 12), 0.U(12.W))
    }
    is("h13".U) {
      io.legal  := funct3 =/= 1.U && funct3 =/= 5.U ||
        funct3 === 1.U && funct7 === 0.U ||
        funct3 === 5.U && (funct7 === 0.U || funct7 === 32.U)
      io.write  := io.legal
      io.result := MuxLookup(funct3, 0.U)(Seq(
        0.U -> (io.source1 + immI),
        1.U -> (io.source1 << rs2),
        2.U -> (io.source1.asSInt < immI.asSInt).asUInt,
        3.U -> (io.source1 < immI),
        4.U -> (io.source1 ^ immI),
        5.U -> Mux(funct7 === 32.U, (io.source1.asSInt >> rs2).asUInt, io.source1 >> rs2),
        6.U -> (io.source1 | immI),
        7.U -> (io.source1 & immI)
      ))
    }
    is("h33".U) {
      io.legal  := funct7 === 0.U || funct7 === 32.U && (funct3 === 0.U || funct3 === 5.U)
      io.write  := io.legal
      io.result := MuxLookup(funct3, 0.U)(Seq(
        0.U -> Mux(funct7 === 32.U, io.source1 - io.source2, io.source1 + io.source2),
        1.U -> (io.source1 << io.source2(4, 0)),
        2.U -> (io.source1.asSInt < io.source2.asSInt).asUInt,
        3.U -> (io.source1 < io.source2).asUInt,
        4.U -> (io.source1 ^ io.source2),
        5.U -> Mux(funct7 === 32.U, (io.source1.asSInt >> io.source2(4, 0)).asUInt, io.source1 >> io.source2(4, 0)),
        6.U -> (io.source1 | io.source2),
        7.U -> (io.source1 & io.source2)
      ))
    }
    is("h63".U) {
      io.legal := funct3 === 0.U || funct3 === 1.U || funct3 >= 4.U
      val taken = MuxLookup(funct3, false.B)(Seq(
        0.U -> (io.source1 === io.source2),
        1.U -> (io.source1 =/= io.source2),
        4.U -> (io.source1.asSInt < io.source2.asSInt),
        5.U -> (io.source1.asSInt >= io.source2.asSInt),
        6.U -> (io.source1 < io.source2),
        7.U -> (io.source1 >= io.source2)
      ))
      io.nextPc := Mux(taken, io.pc + immB, io.pc + 4.U)
    }
    is("h6f".U) {
      io.legal  := true.B
      io.write  := true.B
      io.result := io.pc + 4.U
      io.nextPc := io.pc + immJ
    }
    is("h67".U) {
      io.legal  := funct3 === 0.U
      io.write  := io.legal
      io.result := io.pc + 4.U
      io.nextPc := (io.source1 + immI) & "hfffffffe".U
    }
  }
}
