package framework.balldomain.bbus.memrouter

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.top.GlobalConfig

@instantiable
class ChannelMappingTable(val b: GlobalConfig) extends Module {
  val EntryNum       = b.ballDomain.ballIdMappings.map(_.inBW).sum
  val MappedChannels = b.top.ballMemChannelProducer

  @public
  val io = IO(new Bundle {

    val write = Flipped(Decoupled(new Bundle {
      val idx   = UInt(log2Up(EntryNum).W)
      val outCh = UInt(log2Up(MappedChannels).W)
    }))

    val routeMap   = Output(Vec(EntryNum, UInt(log2Up(MappedChannels).W)))
    val routeValid = Output(Vec(EntryNum, Bool()))
  })

  val routeMap   = Reg(Vec(EntryNum, UInt(log2Up(MappedChannels).W)))
  val routeValid = RegInit(VecInit(Seq.fill(EntryNum)(false.B)))

  when(io.write.fire) {
    routeMap(io.write.bits.idx)   := io.write.bits.outCh
    routeValid(io.write.bits.idx) := true.B
  }

  io.routeMap    := routeMap
  io.routeValid  := routeValid
  io.write.ready := true.B
}
