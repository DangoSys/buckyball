package framework.core.execution.privileged

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.core.clink.CoreInterrupts
import framework.core.memory.access.PhysicalAuthorization
import framework.core.memory.access.preflight.PmpEntry
import framework.core.memory.access.preflight.configs.PmpParams
import framework.core.execution.privileged.configs.{Params => ModuleParam}
import framework.core.execution.privileged.configs.Params

class Trap extends Bundle {
  val pc    = UInt(64.W)
  val cause = UInt(64.W)
  val value = UInt(64.W)
}

/** M/S/U architectural state, precise traps and physical memory permissions. */
class PrivilegedIO(b: ModuleParam) extends Bundle {
  val p = b
  implicit val permission: PmpParams = p.permission

  val hartId     = Input(UInt(64.W))
  val interrupts = Input(new CoreInterrupts)

  val csr = new Bundle {
    val address    = Input(UInt(12.W))
    val operand    = Input(UInt(64.W))
    val operation  = Input(UInt(2.W))
    val write      = Input(Bool())
    val commit     = Input(Bool())
    val data       = Output(UInt(64.W))
    val illegal    = Output(Bool())
    val memoryRead = Output(Bool())
  }

  val trap            = Flipped(Valid(new Trap))
  val returning       = Flipped(Valid(UInt(2.W)))
  val redirect        = Valid(UInt(64.W))
  val interrupt       = Valid(UInt(64.W))
  val wake            = Output(Bool())
  val retired         = Input(Bool())
  val floating        = Flipped(Valid(UInt(5.W)))
  val privilege       = Output(UInt(2.W))
  val dataPrivilege   = Output(UInt(2.W))
  val satp            = Output(UInt(64.W))
  val sum             = Output(Bool())
  val mxr             = Output(Bool())
  val floatingEnabled = Output(Bool())
  val rounding        = Output(UInt(3.W))
  val tvm             = Output(Bool())
  val tw              = Output(Bool())
  val tsr             = Output(Bool())
  val authorization   = Input(Vec(2, new PhysicalAuthorization))
  val permitted       = Output(Vec(2, Bool()))

  val pmp = Output(Vec(
    permission.entries,
    new PmpEntry(b.permission)
  ))

}

@instantiable
class Privileged(b: ModuleParam) extends Module {
  val p = b
  implicit val permission: PmpParams = p.permission
  require(permission.entries == 16)

  @public
  val io = IO(new PrivilegedIO(b))

  val privilege       = RegInit(3.U(2.W))
  val status          = RegInit(0.U(64.W))
  val supervisorMask  = BigInt("80000003000de122", 16)
  val statusMask      = BigInt("00000000007e79aa", 16)
  val statusRead      = status | (BigInt(2).U(64.W) << 32) | (BigInt(2).U(64.W) << 34) |
    Mux(status(14, 13) === 3.U, BigInt("8000000000000000", 16).U, 0.U)
  val mie             = RegInit(0.U(64.W))
  val softwarePending = RegInit(0.U(64.W))
  val medeleg         = RegInit(0.U(64.W))
  val mideleg         = RegInit(0.U(64.W))
  val mtvec           = RegInit(0.U(64.W))
  val stvec           = RegInit(0.U(64.W))
  val mscratch        = RegInit(0.U(64.W))
  val sscratch        = RegInit(0.U(64.W))
  val mepc            = RegInit(0.U(64.W))
  val sepc            = RegInit(0.U(64.W))
  val mcause          = RegInit(0.U(64.W))
  val scause          = RegInit(0.U(64.W))
  val mtval           = RegInit(0.U(64.W))
  val stval           = RegInit(0.U(64.W))
  val satp            = RegInit(0.U(64.W))
  val mcounteren      = RegInit(0.U(3.W))
  val scounteren      = RegInit(0.U(3.W))
  val inhibit         = RegInit(0.U(3.W))
  val cycles          = RegInit(0.U(64.W))
  val instructions    = RegInit(0.U(64.W))
  val fflags          = RegInit(0.U(5.W))
  val frm             = RegInit(0.U(3.W))
  val pmpConfig       = RegInit(VecInit(Seq.fill(16)(0.U(8.W))))
  val pmpAddress      = RegInit(VecInit(Seq.fill(16)(0.U((permission.addressBits - 2).W))))

  when(!inhibit(0))(cycles                     := cycles + 1.U)
  when(io.retired && !inhibit(2))(instructions := instructions + 1.U)
  when(io.floating.valid) {
    fflags := fflags | io.floating.bits
    status := status | (3.U(64.W) << 13)
  }

  val pending = softwarePending |
    (io.interrupts.software.asUInt << 3) | (io.interrupts.timer.asUInt << 7) |
    (io.interrupts.external.asUInt << 11) | (io.interrupts.supervisorExternal.asUInt << 9)

  val enabled           = pending & mie
  val machineEnabled    = privilege < 3.U || status(3)
  val supervisorEnabled = privilege === 0.U || (privilege === 1.U && status(1))
  val eligible          = enabled & (Mux(machineEnabled, ~mideleg, 0.U(64.W)) |
    Mux(supervisorEnabled, mideleg, 0.U(64.W)))
  val interruptPriority = Seq(11, 3, 7, 9, 1, 5)
  io.interrupt.valid := eligible.orR
  io.interrupt.bits  := BigInt("8000000000000000", 16).U |
    PriorityMux(interruptPriority.map(i => eligible(i) -> i.U(64.W)) :+ (true.B -> 0.U(64.W)))
  io.wake            := enabled.orR
  io.privilege       := privilege
  io.dataPrivilege   := Mux(status(17), status(12, 11), privilege)
  io.satp            := satp
  io.sum             := status(18)
  io.mxr             := status(19)
  io.floatingEnabled := status(14, 13).orR
  io.rounding        := frm
  io.tvm             := status(20)
  io.tw              := status(21)
  io.tsr             := status(22)

  val csr            = io.csr
  val floatingCsr    = csr.address >= 1.U && csr.address <= 3.U
  val counterCsr     = csr.address >= "hc00".U && csr.address <= "hc02".U
  val counterBit     = csr.address(1, 0)
  val counterAllowed = privilege === 3.U || (mcounteren(counterBit) &&
    (privilege === 1.U || scounteren(counterBit)))
  val known          = WireDefault(false.B)
  csr.data := 0.U

  val read = Seq(
    0x001 -> fflags.pad(64),
    0x002 -> frm.pad(64),
    0x003 -> Cat(frm, fflags).pad(64),
    0x100 -> (statusRead & supervisorMask.U(64.W)),
    0x104 -> (mie & mideleg),
    0x105 -> stvec,
    0x106 -> scounteren.pad(64),
    0x140 -> sscratch,
    0x141 -> sepc,
    0x142 -> scause,
    0x143 -> stval,
    0x144 -> (pending & mideleg),
    0x180 -> satp,
    0x300 -> statusRead,
    0x301 -> BigInt("800000000014112d", 16).U(64.W),
    0x302 -> medeleg,
    0x303 -> mideleg,
    0x304 -> mie,
    0x305 -> mtvec,
    0x306 -> mcounteren.pad(64),
    0x310 -> 0.U(64.W),
    0x320 -> inhibit.pad(64),
    0x340 -> mscratch,
    0x341 -> mepc,
    0x342 -> mcause,
    0x343 -> mtval,
    0x344 -> pending,
    0x3a0 -> pmpConfig.take(8).reverse.reduce(Cat(_, _)),
    0x3a2 -> pmpConfig.drop(8).reverse.reduce(Cat(_, _)),
    0xb00 -> cycles,
    0xb02 -> instructions,
    0xc00 -> cycles,
    0xc02 -> instructions,
    0xf11 -> 0.U(64.W),
    0xf12 -> 0.U(64.W),
    0xf13 -> 0.U(64.W),
    0xf14 -> io.hartId,
    0xf15 -> 0.U(64.W)
  ) ++ (0 until 16).map(i => (0x3b0 + i) -> pmpAddress(i).pad(64))

  for ((address, data) <- read) {
    when(csr.address === address.U) { known := true.B; csr.data := data }
  }
  csr.memoryRead := csr.address === "hc01".U
  when(csr.memoryRead)(known := true.B)
  csr.illegal                := !known || privilege < csr.address(9, 8) ||
    (csr.write && csr.address(11, 10) === 3.U) ||
    (floatingCsr && !io.floatingEnabled) || (counterCsr && !counterAllowed) ||
    (csr.address === "h180".U && privilege === 1.U && status(20))

  val writeData = MuxLookup(csr.operation, csr.operand)(Seq(
    2.U -> (csr.data | csr.operand),
    3.U -> (csr.data & ~csr.operand)
  ))

  when(csr.commit && csr.write && !csr.illegal) {
    switch(csr.address) {
      is("h001".U) { fflags := writeData(4, 0); status := status | (3.U(64.W) << 13) }
      is("h002".U) { frm := writeData(2, 0); status := status | (3.U(64.W) << 13) }
      is("h003".U) { fflags := writeData(4, 0); frm := writeData(7, 5); status := status | (3.U(64.W) << 13) }
      is("h100".U) {
        status := (status & ~supervisorMask.U(64.W)) | (writeData & supervisorMask.U(64.W) & statusMask.U(64.W))
      }
      is("h104".U)(mie             := (mie & ~mideleg) | (writeData & mideleg))
      is("h105".U)(stvec           := Cat(writeData(63, 2), 0.U(1.W), writeData(1, 0) === 1.U))
      is("h106".U)(scounteren      := writeData(2, 0))
      is("h140".U)(sscratch        := writeData)
      is("h141".U)(sepc            := writeData & ~1.U(64.W))
      is("h142".U)(scause          := writeData)
      is("h143".U)(stval           := writeData)
      is("h144".U)(softwarePending := (softwarePending & ~2.U(64.W)) | (writeData & mideleg & 2.U(64.W)))
      is("h180".U)(when(writeData(63, 60) === 0.U || writeData(63, 60) === 8.U)(satp := writeData))
      is("h300".U) {
        status := (writeData & statusMask.U(64.W) & ~(3.U(64.W) << 11)) |
          (Mux(writeData(12, 11) === 2.U, 0.U(2.W), writeData(12, 11)).pad(64) << 11)
      }
      is("h302".U)(medeleg         := writeData & "hb3ff".U)
      is("h303".U)(mideleg         := writeData & "h222".U)
      is("h304".U)(mie             := writeData & "haaa".U)
      is("h305".U)(mtvec           := Cat(writeData(63, 2), 0.U(1.W), writeData(1, 0) === 1.U))
      is("h306".U)(mcounteren      := writeData(2, 0))
      is("h320".U)(inhibit         := writeData(2, 0) & 5.U)
      is("h340".U)(mscratch        := writeData)
      is("h341".U)(mepc            := writeData & ~1.U(64.W))
      is("h342".U)(mcause          := writeData)
      is("h343".U)(mtval           := writeData)
      is("h344".U)(softwarePending := writeData & "h222".U)
      is("hb00".U)(cycles          := writeData)
      is("hb02".U)(instructions    := writeData)
    }
    for (i <- 0 until 16) {
      when(csr.address === (0x3a0 + (i / 8) * 2).U && !pmpConfig(i)(7)) {
        val cfg = writeData(8 * (i % 8) + 7, 8 * (i % 8))
        pmpConfig(i) := cfg & "h9d".U | Mux(cfg(0) && cfg(1), 2.U, 0.U)
      }
      val lockedByTor = if (i == 15) false.B else pmpConfig(i + 1)(7) && pmpConfig(i + 1)(4, 3) === 1.U
      when(csr.address === (0x3b0 + i).U && !pmpConfig(i)(7) && !lockedByTor) {
        pmpAddress(i) := writeData(permission.addressBits - 3, 0)
      }
    }
  }

  val supervisorTrap =
    privilege =/= 3.U && Mux(io.trap.bits.cause(63), mideleg(io.trap.bits.cause(5, 0)), medeleg(io.trap.bits.cause(5, 0)))
  val trapVector     = Mux(supervisorTrap, stvec, mtvec)
  io.redirect.valid := io.trap.valid || io.returning.valid
  io.redirect.bits  := Mux(
    io.trap.valid,
    (trapVector & ~3.U(64.W)) + Mux(io.trap.bits.cause(63) && trapVector(0), io.trap.bits.cause(5, 0) << 2, 0.U),
    Mux(io.returning.bits === 3.U, mepc, sepc)
  )
  when(io.trap.valid) {
    when(supervisorTrap) {
      sepc      := io.trap.bits.pc & ~1.U(64.W)
      scause    := io.trap.bits.cause
      stval     := io.trap.bits.value
      status    := (status & ~"h122".U(64.W)) | (status(1).asUInt << 5) | ((privilege === 1.U).asUInt << 8)
      privilege := 1.U
    }.otherwise {
      mepc      := io.trap.bits.pc & ~1.U(64.W)
      mcause    := io.trap.bits.cause
      mtval     := io.trap.bits.value
      status    := (status & ~"h1888".U(64.W)) | (status(3).asUInt << 7) | (privilege.pad(64) << 11)
      privilege := 3.U
    }
  }.elsewhen(io.returning.valid) {
    when(io.returning.bits === 3.U) {
      privilege := status(12, 11)
      status    := (status & ~"h1888".U(64.W) & Mux(status(12, 11) === 3.U, ~0.U(64.W), ~"h20000".U(64.W))) |
        (status(7).asUInt << 3) | "h80".U
    }.otherwise {
      privilege := Mux(status(8), 1.U, 0.U)
      status    := (status & ~"h20122".U(64.W)) | (status(5).asUInt << 1) | "h20".U
    }
  }

  for (i <- 0 until 16) {
    io.pmp(i).cfg  := pmpConfig(i).asTypeOf(io.pmp(i).cfg)
    io.pmp(i).addr := pmpAddress(i)
    io.pmp(i).mask := ((pmpAddress(i) ^ (pmpAddress(i) + 1.U)) << 2) | 3.U
  }
  // PMP bounds and the authorization are registered: permitted answers the authorization of the
  // previous cycle against the PMP state of that same cycle. Bounds are inclusive so the
  // decision path only compares.
  val bounds = Reg(Vec(16, new PmpBound))
  val requests = RegNext(io.authorization)
  // Last byte of each authorization: paddr + 2^size - 1.
  val lasts    = RegNext(VecInit(io.authorization.map(a => a.paddr +& ((1.U(8.W) << a.size) - 1.U))))
  for (i    <- 0 until 16) {
    val encoded = (pmpAddress(i) << 2).pad(64)
    val mask    = io.pmp(i).mask.pad(64)
    val mode    = pmpConfig(i)(4, 3)
    val below   = if (i == 0) 0.U(64.W) else (pmpAddress(i - 1) << 2).pad(64)
    bounds(i).low     := Mux(mode === 1.U, below, Mux(mode === 3.U, encoded & ~mask, encoded))
    bounds(i).last    := MuxLookup(mode, encoded | 3.U)(Seq(
      1.U -> ((pmpAddress(i) - 1.U) << 2 | 3.U).pad(64),
      3.U -> (encoded | mask)
    ))
    bounds(i).enabled := mode =/= 0.U && (mode =/= 1.U || below < encoded)
    bounds(i).config  := pmpConfig(i)
  }
  for (port <- 0 until 2) {
    val request = requests(port)
    val last    = lasts(port)
    val matches = bounds.map { bound =>
      val hit     = bound.enabled && request.paddr <= bound.last && last >= bound.low
      val allowed = (request.privilege === 3.U && !bound.config(7)) ||
        ((!request.read || bound.config(0)) && (!request.write || bound.config(1)) &&
          (!request.execute || bound.config(2)))
      hit -> (allowed && request.paddr >= bound.low && last <= bound.last)
    }
    io.permitted(port) := PriorityMux(matches :+ (true.B -> (request.privilege === 3.U)))
  }
}

/** One PMP entry as an inclusive byte range [low, last]. */
class PmpBound extends Bundle {
  val low     = UInt(64.W)
  val last    = UInt(64.W)
  val enabled = Bool()
  val config  = UInt(8.W)
}
