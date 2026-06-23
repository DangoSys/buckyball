package framework.balldomain.prototype.systolicarray

import chisel3._
import chisel3.util._
import chisel3.stage._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.top.GlobalConfig
import framework.balldomain.prototype.systolicarray.configs.SystolicBallParam

class ctrl_ex_req(b: GlobalConfig) extends Bundle {
  val iter   = UInt(b.frontend.iter_len.W)
  val config = UInt(64.W)
}

class ld_ex_req(b: GlobalConfig) extends Bundle {
  val config = SystolicBallParam()
  val op1    = Vec(config.lane, UInt(config.inputWidth.W))
  val op2    = Vec(config.lane, UInt(config.inputWidth.W))
  val iter   = UInt(b.frontend.iter_len.W)
}

class ex_st_req(b: GlobalConfig) extends Bundle {
  val config = SystolicBallParam()
  val result = Vec(config.lane, UInt(config.outputWidth.W))
}

class PE(val inputWidth: Int, val outputWidth: Int) extends Module {

  val io = IO(new Bundle {
    val in_a  = Flipped(Decoupled(UInt(inputWidth.W)))
    val in_b  = Flipped(Decoupled(UInt(inputWidth.W)))
    val out_a = Decoupled(UInt(inputWidth.W))
    val out_b = Decoupled(UInt(inputWidth.W))
    val out_c = Output(UInt(outputWidth.W))
    val clear = Input(Bool())
  })

  io.out_a.valid := RegNext(io.in_a.valid)
  io.out_a.bits  := RegNext(io.in_a.bits)
  io.in_a.ready  := io.out_a.ready

  io.out_b.valid := RegNext(io.in_b.valid)
  io.out_b.bits  := RegNext(io.in_b.bits)
  io.in_b.ready  := io.out_b.ready

  val acc_reg = RegInit(0.U(outputWidth.W))

  when(io.clear) {
    acc_reg := 0.U
  }.elsewhen(io.in_a.fire && io.in_b.fire) {
    acc_reg := io.in_a.bits * io.in_b.bits + acc_reg
  }

  io.out_c := acc_reg

}

@instantiable
class SystolicArrayEX(val b: GlobalConfig) extends Module {
  val config      = SystolicBallParam()
  val inputWidth  = config.inputWidth
  val outputWidth = config.outputWidth
  val arraySize   = config.lane
  val wsModeBit   = 0

  @public
  val io = IO(new Bundle {
    val ctrl_ex_i = Flipped(Decoupled(new ctrl_ex_req(b)))
    val ld_ex_i   = Flipped(Decoupled(new ld_ex_req(b)))

    val ex_st_o = Decoupled(new ex_st_req(b))
  })

  val idle :: busy :: Nil = Enum(2)
  val state               = RegInit(idle)

  val cfgIter          = RegInit(0.U(b.frontend.iter_len.W))
  val cfgMode          = RegInit(0.U(64.W))
  val iter_counter     = RegInit(0.U(b.frontend.iter_len.W))
  val store_counter    = RegInit(0.U(log2Ceil(config.lane + 1).W))
  val in_counter       = RegInit(0.U(b.frontend.iter_len.W))
  val osDrainActive    = RegInit(false.B)
  val isWsMode         = cfgMode(wsModeBit)
  val maxIter          = arraySize.U(b.frontend.iter_len.W)
  val effectiveIter    = Mux(cfgIter > maxIter, maxIter, cfgIter)
  val osDrainThreshold = effectiveIter + (2 * arraySize - 2).U(b.frontend.iter_len.W)
  val wsRowCounter     = RegInit(0.U(b.frontend.iter_len.W))
  val wsKCounter       = RegInit(0.U(b.frontend.iter_len.W))
  val wsAcc            = RegInit(VecInit(Seq.fill(arraySize)(0.U(outputWidth.W))))

  // Use Reg with Vec type for proper register behavior
  val in_a_buffer = Reg(Vec(arraySize, Vec(arraySize, UInt(inputWidth.W))))
  val in_b_buffer = Reg(Vec(arraySize, Vec(arraySize, UInt(inputWidth.W))))
  val pes         = VecInit(Seq.fill(arraySize)(VecInit(Seq.fill(arraySize)(Module(new PE(inputWidth, outputWidth)).io))))
  val clearPes    = WireDefault(false.B)

  // default values
  io.ctrl_ex_i.ready := state === idle
  io.ld_ex_i.ready   := state === busy && (in_counter < effectiveIter)

  io.ex_st_o.valid       := false.B
  io.ex_st_o.bits.result := VecInit(Seq.fill(arraySize)(0.U(outputWidth.W)))

  for (row <- 0 until arraySize) {
    for (col <- 0 until arraySize) {
      pes(row)(col).clear := clearPes
    }
  }

  when(io.ctrl_ex_i.fire) {
    clearPes      := true.B
    cfgIter       := io.ctrl_ex_i.bits.iter
    cfgMode       := io.ctrl_ex_i.bits.config
    iter_counter  := 0.U
    store_counter := 0.U
    in_counter    := 0.U
    osDrainActive := false.B
    wsRowCounter  := 0.U
    wsKCounter    := 0.U
    wsAcc         := VecInit(Seq.fill(arraySize)(0.U(outputWidth.W)))
    for (row <- 0 until arraySize) {
      for (col <- 0 until arraySize) {
        in_a_buffer(row)(col) := 0.U
        in_b_buffer(row)(col) := 0.U
      }
    }
    state := busy
  }

  // input data to buffer
  when(io.ld_ex_i.fire) {
    in_counter := in_counter + 1.U
    when(in_counter < arraySize.U) {
      for (i <- 0 until arraySize) {
        in_a_buffer(in_counter)(i) := io.ld_ex_i.bits.op1(i)
        in_b_buffer(in_counter)(i) := io.ld_ex_i.bits.op2(i)
      }
    }
  }

  // PEs connection
  val inputsReady  = state === busy && (effectiveIter =/= 0.U) && (in_counter >= effectiveIter)
  val osDrainStart = inputsReady && !isWsMode && !osDrainActive && (iter_counter >= osDrainThreshold)

  when(osDrainStart) {
    osDrainActive := true.B
  }

  when(inputsReady && !isWsMode && !osDrainActive) {
    iter_counter := iter_counter + 1.U

    for (row <- 0 until arraySize) {
      for (col <- 0 until arraySize) {
        if (row == 0 && col == 0) {
          when(iter_counter < arraySize.U) {
            pes(row)(col).in_a.valid := true.B
            pes(row)(col).in_a.bits  := in_a_buffer(0)(iter_counter)
            pes(row)(col).in_b.valid := true.B
            pes(row)(col).in_b.bits  := in_b_buffer(iter_counter)(0)
          }.otherwise {
            pes(row)(col).in_a.valid := false.B
            pes(row)(col).in_a.bits  := 0.U
            pes(row)(col).in_b.valid := false.B
            pes(row)(col).in_b.bits  := 0.U
          }

        } else if (row == 0 && col > 0) {
          when((iter_counter >= col.U) && (iter_counter < arraySize.U + col.U)) {
            pes(row)(col).in_b.valid := true.B
            pes(row)(col).in_b.bits  := in_b_buffer(iter_counter - col.U)(col)
            pes(row)(col).in_a <> pes(row)(col - 1).out_a
          }.otherwise {
            pes(row)(col).in_b.valid := false.B
            pes(row)(col).in_b.bits  := 0.U
            pes(row)(col).in_a <> pes(row)(col - 1).out_a
          }

        } else if (col == 0 && row > 0) {
          when((iter_counter >= row.U) && (iter_counter < arraySize.U + row.U)) {
            pes(row)(col).in_a.valid := true.B
            pes(row)(col).in_a.bits  := in_a_buffer(row)(iter_counter - row.U)
            pes(row)(col).in_b <> pes(row - 1)(col).out_b
          }.otherwise {
            pes(row)(col).in_a.valid := false.B
            pes(row)(col).in_a.bits  := 0.U
            pes(row)(col).in_b <> pes(row - 1)(col).out_b
          }

        } else if (row > 0 && col > 0) {
          pes(row)(col).in_a <> pes(row)(col - 1).out_a
          pes(row)(col).in_b <> pes(row - 1)(col).out_b
        }

        if (row == arraySize - 1 || col == arraySize - 1) {
          pes(row)(col).out_a.ready := io.ex_st_o.ready
          pes(row)(col).out_b.ready := io.ex_st_o.ready
        }
      }
    }

  }.otherwise {
    for (row <- 0 until arraySize) {
      for (col <- 0 until arraySize) {
        pes(row)(col).in_a.valid  := false.B
        pes(row)(col).in_a.bits   := 0.U
        pes(row)(col).in_b.valid  := false.B
        pes(row)(col).in_b.bits   := 0.U
        pes(row)(col).out_a.ready := io.ex_st_o.ready
        pes(row)(col).out_b.ready := io.ex_st_o.ready
      }
    }
  }

  when(inputsReady && isWsMode && wsRowCounter < effectiveIter) {
    when(wsKCounter < effectiveIter) {
      for (col <- 0 until arraySize) {
        val product = in_a_buffer(wsRowCounter)(wsKCounter) * in_b_buffer(wsKCounter)(col)
        wsAcc(col) := wsAcc(col) + product
      }
      wsKCounter := wsKCounter + 1.U
    }.otherwise {
      io.ex_st_o.valid       := true.B
      io.ex_st_o.bits.result := wsAcc
      when(io.ex_st_o.ready) {
        val nextRow = wsRowCounter + 1.U
        wsRowCounter := nextRow
        wsKCounter   := 0.U
        wsAcc        := VecInit(Seq.fill(arraySize)(0.U(outputWidth.W)))
        when(nextRow >= effectiveIter) {
          clearPes      := true.B
          cfgIter       := 0.U
          cfgMode       := 0.U
          iter_counter  := 0.U
          store_counter := 0.U
          in_counter    := 0.U
          osDrainActive := false.B
          state         := idle
        }
      }
    }
  }

  // output data from PEs
  when(osDrainActive) {
    when(store_counter < effectiveIter) {
      io.ex_st_o.valid       := true.B
      io.ex_st_o.bits.result := VecInit(pes(store_counter).map(_.out_c))
      when(io.ex_st_o.ready) {
        store_counter := store_counter + 1.U
      }
    }.otherwise {
      // back to idle
      clearPes      := true.B
      cfgIter       := 0.U
      cfgMode       := 0.U
      iter_counter  := 0.U
      store_counter := 0.U
      in_counter    := 0.U
      osDrainActive := false.B
      wsRowCounter  := 0.U
      wsKCounter    := 0.U
      wsAcc         := VecInit(Seq.fill(arraySize)(0.U(outputWidth.W)))
      state         := idle
    }
  }

}
