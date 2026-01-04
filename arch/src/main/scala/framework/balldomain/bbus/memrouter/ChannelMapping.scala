package framework.balldomain.bbus.memrouter

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.top.GlobalConfig

@instantiable
class ChannelMappingTable(val b: GlobalConfig) extends Module {
  val totalReadChannels    = b.ballDomain.ballIdMappings.map(_.inBW).sum
  val bbusProducerChannels = b.top.ballMemChannelProducer

  @public
  val io = IO(new Bundle {

    val write = Flipped(Decoupled(new Bundle {
      val idx   = UInt(log2Up(totalReadChannels).W)
      val outCh = UInt(log2Up(bbusProducerChannels).W)
    }))

    val routeMap   = Output(Vec(totalReadChannels, UInt(log2Up(bbusProducerChannels).W)))
    val routeValid = Output(Vec(totalReadChannels, Bool()))
  })

  val routeMap   = Reg(Vec(totalReadChannels, UInt(log2Up(bbusProducerChannels).W)))
  val routeValid = RegInit(VecInit(Seq.fill(totalReadChannels)(false.B)))

  when(io.write.fire) {
    routeMap(io.write.bits.idx)   := io.write.bits.outCh
    routeValid(io.write.bits.idx) := true.B
  }

  io.routeMap    := routeMap
  io.routeValid  := routeValid
  io.write.ready := true.B
}
