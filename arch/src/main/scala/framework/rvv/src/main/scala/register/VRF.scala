package framework.rvv

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.top.GlobalConfig

@instantiable
class VRF(val b: GlobalConfig) extends Module {
  val p                = b.rvv
  val wordBits         = p.wordBits
  val wordsPerRegister = p.vLen / wordBits
  val bankCount        = p.registerBanks
  val bankBits         = log2Ceil(bankCount)
  val addressBits      = log2Ceil(32 * wordsPerRegister)
  val bankDepth        = 32 * wordsPerRegister / bankCount
  require(isPow2(bankCount))

  @public
  val io = IO(new Bundle {
    val initialize = Input(Bool())
    val busy       = Output(Bool())

    val request = Vec(
      bankCount,
      Flipped(Decoupled(new Bundle {
        val address = UInt(addressBits.W)
        val write   = Bool()
        val data    = UInt(wordBits.W)
        val mask    = UInt(wordBits.W)
      }))
    )

    val response = Vec(bankCount, Decoupled(UInt(wordBits.W)))
  })

  val clearing                                        = RegInit(true.B)
  val clearAddress                                    = RegInit(0.U(log2Ceil(bankDepth).W))
  val idle :: reading :: writing :: responding :: Nil = Enum(4)
  val states                                          = RegInit(VecInit(Seq.fill(bankCount)(idle)))

  when(io.initialize) {
    clearing     := true.B
    clearAddress := 0.U
  }.elsewhen(clearing) {
    when(clearAddress === (bankDepth - 1).U) {
      clearing := false.B
    }.otherwise {
      clearAddress := clearAddress + 1.U
    }
  }

  io.busy := io.initialize || clearing || states.map(_ =/= idle).reduce(_ || _)

  for (bank <- 0 until bankCount) {
    val memory   = SyncReadMem(bankDepth, UInt(wordBits.W))
    val address  = Reg(UInt(log2Ceil(bankDepth).W))
    val write    = Reg(Bool())
    val data     = Reg(UInt(wordBits.W))
    val mask     = Reg(UInt(wordBits.W))
    val result   = Reg(UInt(wordBits.W))
    val request  = io.request(bank)
    val response = io.response(bank)

    request.ready  := !io.initialize && !clearing && states(bank) === idle
    response.valid := !io.initialize && !clearing && states(bank) === responding
    response.bits  := result

    val portAddress = Mux(clearing, clearAddress, Mux(request.fire, request.bits.address >> bankBits, address))
    val portWrite   = clearing || states(bank) === writing
    val portData    = Mux(clearing, 0.U, result)
    val portEnable  = !io.initialize && (clearing || request.fire || states(bank) === writing)
    val readData    = memory.readWrite(portAddress, portData, portEnable, portWrite)

    when(io.initialize) {
      states(bank) := idle
    }.elsewhen(!clearing) {
      switch(states(bank)) {
        is(idle) {
          when(request.fire) {
            if (bankBits > 0) {
              assert(request.bits.address(bankBits - 1, 0) === bank.U)
            }
            address      := request.bits.address >> bankBits
            write        := request.bits.write
            data         := request.bits.data
            mask         := request.bits.mask
            states(bank) := reading
          }
        }
        is(reading) {
          result       := Mux(write, (readData & ~mask) | (data & mask), readData)
          states(bank) := Mux(write, writing, responding)
        }
        is(writing) {
          states(bank) := responding
        }
        is(responding) {
          when(response.fire) {
            states(bank) := idle
          }
        }
      }
    }
  }
}
