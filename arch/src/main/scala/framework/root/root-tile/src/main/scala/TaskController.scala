package hier.tile

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.system.core.rocket.RoCCIO

/** Linux submits compiled kernels; workers consume contexts without joining Linux SMP. */
@instantiable
class TaskController(coreIds: Seq[Int], signatures: Seq[BigInt], cpuCount: Int) extends Module {
  require(coreIds == (1 to signatures.size) && signatures.nonEmpty)
  require(cpuCount == coreIds.size + 1, "task workers must include every non-controller CPU")

  @public val io = IO(new Bundle {
    val ports = Vec(cpuCount, new RoCCIO(64))
    val satp  = Input(UInt(64.W))
  })

  val count        = coreIds.size
  val kinds        = VecInit(signatures.map(_.U(64.W)))
  val contexts     = Reg(Vec(count, Vec(7, UInt(64.W))))
  val roots        = Reg(Vec(count, UInt(64.W)))
  val active       = RegInit(VecInit(Seq.fill(count)(false.B)))
  val available    = RegInit(VecInit(Seq.fill(count)(false.B)))
  val results      = RegInit(VecInit(Seq.fill(count)(1.U(2.W))))
  val staged       = Reg(Vec(7, UInt(64.W)))
  val stagedRoot   = Reg(UInt(64.W))
  val stagedFields = RegInit(0.U(3.W))
  val workspace    = RegInit(0.U(64.W))

  for (cpu <- 0 until cpuCount) {
    val port      = io.ports(cpu)
    val worker    = coreIds.indexOf(cpu)
    val pending   = RegInit(false.B)
    val rd        = Reg(UInt(5.W))
    val response  = Reg(UInt(64.W))
    val command   = port.cmd.bits
    val op        = command.funct
    val index     = command.rs1Data
    val value     = command.rs2Data
    val slot      = index(math.max(1, log2Ceil(count)) - 1, 0)
    val field     = index(2, 0)
    val inRange   = index < count.U
    val free      = VecInit((0 until count).map(i => !active(i) && kinds(i) === value)).asUInt.orR
    val waitReady =
      if (cpu == 0)
        Mux(op === 1.U, Mux(index === count.U, free, !inRange || !active(slot)), true.B)
      else if (worker >= 0) Mux(op === 8.U, available(worker), true.B)
      else true.B
    port.cmd.ready               := !pending && waitReady
    port.resp.valid              := pending
    port.resp.bits.rd            := rd
    port.resp.bits.data          := response
    port.busy                    := pending
    port.interrupt               := false.B
    when(port.resp.fire)(pending := false.B)
    when(port.cmd.fire) {
      pending  := true.B
      rd       := command.rd
      response := 2.U
      if (cpu == 0) {
        switch(op) {
          is(0.U) {
            when(inRange && !active(slot) && stagedFields === 7.U &&
              stagedRoot === io.satp && kinds(slot) === value && staged(6) === value) {
              contexts(slot)  := staged
              roots(slot)     := stagedRoot
              active(slot)    := true.B
              available(slot) := true.B
              results(slot)   := 0.U
              stagedFields    := 0.U
              response        := 0.U
            }
          }
          is(1.U)(response  := Mux(index === count.U, 0.U, Mux(inRange, results(slot), 2.U)))
          is(3.U)(response  := count.U)
          is(4.U)(response  := Mux(inRange, kinds(slot), 0.U))
          is(5.U)(response  := workspace)
          is(6.U) { workspace := index; response := 0.U }
          is(7.U) {
            when(index === 0.U || (index < 7.U && index === stagedFields && stagedRoot === io.satp)) {
              staged(field)                  := value
              stagedFields                   := field + 1.U
              when(index === 0.U)(stagedRoot := io.satp)
              response                       := 0.U
            }
          }
          is(10.U)(response := Mux(inRange, results(slot), 2.U))
        }
      } else if (worker >= 0) {
        switch(op) {
          is(2.U) {
            when(active(worker) && !available(worker)) {
              active(worker)  := false.B
              results(worker) := Mux(index === 0.U, 1.U, 2.U)
              response        := 0.U
            }
          }
          is(5.U)(response := contexts(worker)(4))
          is(11.U)(when(active(worker) && !available(worker))(response := 0.U))
          is(8.U) { available(worker) := false.B; response := 1.U }
          is(9.U)(response := Mux(index === 7.U, roots(worker), Mux(index < 7.U, contexts(worker)(field), 0.U)))
        }
      }
    }
  }
}
