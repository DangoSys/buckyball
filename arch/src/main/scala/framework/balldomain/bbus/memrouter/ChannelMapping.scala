package framework.balldomain.bbus.memrouter

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.top.GlobalConfig

@instantiable
class ChannelMappingTable(val b: GlobalConfig) extends Module {
  val EntryNum       = b.ballDomain.ballIdMappings.map(_.inBW).sum
  val MappedChannels = b.top.ballMemChannelNum

  // 多写口数量：与输出通道数一致（一次最多派发这么多条）
  val WritePorts = MappedChannels

  @public
  val io = IO(new Bundle {
    val write = Flipped(Vec(WritePorts, Decoupled(new Bundle {
      val idx   = UInt(log2Up(EntryNum).W)
      val outCh = UInt(log2Up(MappedChannels).W)
    })))

    // 条目级清理：每个写口对应一条可能的派发通道（通常用 outCh 作为端口号）
    val invalidate = Input(Vec(WritePorts, Valid(UInt(log2Up(EntryNum).W))))

    val routeMap   = Output(Vec(EntryNum, UInt(log2Up(MappedChannels).W)))
    val routeValid = Output(Vec(EntryNum, Bool()))
  })

  val routeMap   = Reg(Vec(EntryNum, UInt(log2Up(MappedChannels).W)))
  val routeValid = RegInit(VecInit(Seq.fill(EntryNum)(false.B)))

  for (w <- 0 until WritePorts) {
    io.write(w).ready := true.B

    // 先处理清理：把对应 idx 的 valid 清掉
    when(io.invalidate(w).valid) {
      routeValid(io.invalidate(w).bits) := false.B
    }

    // 再处理写入：同拍既清又写时，让写入优先（最终为 valid=true）
    when(io.write(w).fire) {
      routeMap(io.write(w).bits.idx)   := io.write(w).bits.outCh
      routeValid(io.write(w).bits.idx) := true.B
    }
  }

  io.routeMap   := routeMap
  io.routeValid := routeValid

  // Assert: in the same cycle, different write ports must not target the same idx
  for (p <- 0 until WritePorts) {
    for (q <- p + 1 until WritePorts) {
      when(io.write(p).valid && io.write(q).valid) {
        assert(
          io.write(p).bits.idx =/= io.write(q).bits.idx,
          "ChannelMappingTable: multiple write ports target the same idx in one cycle"
        )
      }
    }
  }
}
