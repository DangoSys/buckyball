package examples.balls.mxmm

import chisel3.experimental.hierarchy.{instantiable, public}

import framework.acc.memdomain.configs.MemDomainParam

import examples.balls.mxmm.configs.MxmmBallParam

import framework.ip.memory.bank.{SramReadIO, SramReadReq, SramReadResp, SramWriteIO, SramWriteReq, SramWriteResp}

import chisel3._
import chisel3.util._
import framework.ip.memory.bank.{SramReadReq, SramReadResp}

@instantiable
class InputWindow(memParam: MemDomainParam, val ballParam: MxmmBallParam) extends Module {
  val addressWidth = log2Ceil(memParam.bankEntries)

  @public
  val io = IO(new Bundle {
    val start      = Input(Bool())
    val enable     = Input(Bool())
    val rows       = Input(UInt(12.W))
    val count      = Input(UInt(34.W))
    val fullK      = Input(UInt(16.W))
    val startK     = Input(UInt(16.W))
    val request    = Decoupled(new SramReadReq(memParam))
    val response   = Flipped(Decoupled(new SramReadResp(memParam)))
    val done       = Output(Bool())
    val write      = Output(Bool())
    val scaleWrite = Output(Bool())
    val lane       = Output(UInt(4.W))
    val address    = Output(UInt(addressWidth.W))
    val word       = Output(UInt(128.W))
    val scale      = Output(UInt(8.W))
  })

  val request :: response :: scales :: done :: Nil = Enum(4)
  val state                                        = RegInit(done)
  val rows                                         = Reg(UInt(12.W))
  val count                                        = Reg(UInt(34.W))
  val fullK                                        = Reg(UInt(16.W))
  val startK                                       = Reg(UInt(16.W))
  val row                                          = Reg(UInt(12.W))
  val offset                                       = Reg(UInt(addressWidth.W))
  val scaleMode                                    = Reg(Bool())
  val word                                         = Reg(UInt(128.W))
  val scaleAddress                                 = rows * fullK + row * (fullK >> 5) + (startK >> 5) + offset
  val byte                                         = scaleAddress(3, 0)
  io.request.valid            := io.enable && state === request
  io.request.bits.addr        := Mux(scaleMode, scaleAddress >> 4, row * (fullK >> 4) + (startK >> 4) + offset)
  io.response.ready           := io.enable && state === response
  io.done                     := state === done
  io.write                    := io.response.fire && !scaleMode
  io.scaleWrite               := io.enable && state === scales
  io.lane                     := row(3, 0)
  io.address                  := (row >> 4) * Mux(scaleMode, count >> 5, count >> 4) + offset
  io.word                     := io.response.bits.data
  io.scale                    := (word >> (byte << 3))(7, 0)
  when(io.start) {
    rows      := io.rows
    count     := io.count
    fullK     := io.fullK
    startK    := io.startK
    row       := 0.U
    offset    := 0.U
    scaleMode := false.B
    state     := request
  }
  when(io.request.fire)(state := response)
  when(io.response.fire) {
    when(scaleMode) { word := io.response.bits.data; state := scales }.otherwise {
      when(offset + 1.U === (count >> 4)) {
        offset           := 0.U
        when(row + 1.U === rows) { row := 0.U; scaleMode := true.B }
          .otherwise(row := row + 1.U)
      }.otherwise(offset := offset + 1.U)
      state              := request
    }
  }
  when(io.enable && state === scales) {
    when(offset + 1.U === (count >> 5)) {
      offset := 0.U
      when(row + 1.U === rows)(state := done)
        .otherwise { row := row + 1.U; state := request }
    }.otherwise {
      offset                    := offset + 1.U
      when(byte === 15.U)(state := request)
    }
  }
}
