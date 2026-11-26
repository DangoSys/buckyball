package examples.balls.spikeio

import chisel3._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}
import chisel3.util._
import framework.ball.bbus.{BallFactory, BallPorts}
import framework.ball.blink.{BlinkIO, HasBlink}
import framework.ball.sub_rob.ExpansionParam
import framework.generic.contract.blink.BlinkParam
import framework.generic.contract.configs.BallRouteParam

@instantiable
class SpikeIoBall(link: BlinkParam, route: BallRouteParam) extends Module with HasBlink {
  route.parameters.unpack(classOf[buckyball.balls.spikeio.Params])
  require(link.bankWidth == 128 && link.bankEntries >= 8 && route.inBW == 1 && route.outBW == 1)
  val codes = route.instructions.map(i => i.mnemonic -> i.funct7).toMap
  require(Set("SPIKE_ARGMAX", "SPIKE_ENCODE").subsetOf(codes.keySet) && codes.keySet.subsetOf(Set("SPIKE_ARGMAX", "SPIKE_ENCODE", "SPIKE_COUNT")) && codes.values.forall(_ >> 4 == 3))
  @public val blink = IO(new BlinkIO(link, route))
  val idle :: reading :: receiving :: writing :: acknowledging :: complete :: Nil = Enum(6)
  val state = RegInit(idle)
  val command = Reg(chiselTypeOf(blink.cmdReq.bits))
  val row = RegInit(0.U(3.W))
  val best = Reg(SInt(16.W))
  val winner = Reg(UInt(4.W))
  val result = Reg(UInt(16.W))
  val count = codes.get("SPIKE_COUNT").map(code => command.funct === code.U).getOrElse(false.B)
  val encode = command.funct === codes("SPIKE_ENCODE").U
  blink.cmdReq.ready := state === idle
  when(blink.cmdReq.fire) {
    assert(blink.cmdReq.bits.iter === 1.U)
    command := blink.cmdReq.bits
    row := 0.U
    best := (-32768).S
    winner := 0.U
    state := reading
  }
  val read = blink.bankRead(0)
  read.bank_id := command.op1_bank
  read.group_id := command.op1_group
  read.rob_id := command.rob_id
  read.io.req.valid := state === reading
  read.io.req.bits.addr := row
  read.io.resp.ready := state === receiving
  when(read.io.req.fire) { state := receiving }
  val data = read.io.resp.bits.data
  val spikes = VecInit((0 until 16).map(i => data(i*8+7, i*8).asSInt > command.rs2(7,0).asSInt))
  val lo = data(15,0).asSInt
  val hi = data(31,16).asSInt
  val lane = hi > lo
  val candidate = Mux(lane, hi, lo)
  val index = Cat(row, lane)
  when(read.io.resp.fire) {
    when(count) {
      result := PopCount(data(15,0))
      state := writing
    }.elsewhen(encode) {
      result := spikes.asUInt
      state := writing
    }.otherwise {
      when(candidate > best) { best := candidate; winner := index }
      when(row === 7.U) {
        result := Mux(candidate > best, index, winner)
        state := writing
      }.otherwise { row := row + 1.U; state := reading }
    }
  }
  val write = blink.bankWrite(0)
  write.bank_id := command.wr_bank
  write.group_id := command.wr_group
  write.rob_id := command.rob_id
  write.io.req.valid := state === writing
  write.io.req.bits.addr := 0.U
  write.io.req.bits.data := result
  write.io.req.bits.mask.foreach(_ := true.B)
  write.io.resp.ready := state === writing || state === acknowledging
  when(write.io.req.fire) { state := acknowledging }
  when(write.io.resp.fire) { assert(write.io.resp.bits.ok); state := complete }
  blink.cmdResp.valid := state === complete
  blink.cmdResp.bits.rob_id := command.rob_id
  when(blink.cmdResp.fire) { state := idle }
  blink.status.idle := state === idle
  blink.status.running := state =/= idle
}
object SpikeIoBall extends BallFactory {
  def instantiate(link: BlinkParam, route: BallRouteParam, expansion: Option[ExpansionParam]): BallPorts =
    BallPorts(Instantiate(new SpikeIoBall(link, route)).blink, None)
}
