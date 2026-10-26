package framework.acc.frontend.scheduler.rob.global

import framework.acc.frontend.configs.FrontendParam
import framework.acc.gpdomain.configs.GpDomainParam
import framework.acc.memdomain.configs.MemDomainParam
import framework.core.memory.access.preflight.configs.PmpParams

import chisel3._
import chisel3.experimental.hierarchy.{instantiable, public}
import chisel3.util._
import framework.acc.frontend.decoder.PostGDCmd
import framework.acc.memdomain.DISA.MSET_BITPAT
import framework.acc.memdomain.MvoverISA

@instantiable
class DependencyResolver(
  val frontendParam:   FrontendParam,
  val memParam:        MemDomainParam,
  val gpParam:         GpDomainParam,
  val permissionParam: PmpParams)
    extends Module {
  val commandDataBits = frontendParam.commandDataBits
  val robDepth        = frontendParam.rob_entries

  val resourceCount =
    memParam.virtualBankCount + (if (gpParam.enable) 2
                                 else 0)

  @public
  val io = IO(new Bundle {
    val alloc        = Input(new PostGDCmd(frontendParam, memParam))
    val entries      = Input(Vec(robDepth, new GlobalRobEntry(frontendParam, memParam, permissionParam)))
    val valid        = Input(Vec(robDepth, Bool()))
    val complete     = Input(Vec(robDepth, Bool()))
    val dependencies = Output(UInt(robDepth.W))
    val movementLive = Output(Bool())
  })

  val entryReads    = Wire(Vec(robDepth, UInt(resourceCount.W)))
  val entryWrites   = Wire(Vec(robDepth, UInt(resourceCount.W)))
  val entryIsConfig = Wire(Vec(robDepth, Bool()))

  for (slot <- 0 until robDepth) {
    val access    = io.entries(slot).cmd.bankAccess
    val read0     = Mux(
      access.rd_bank_0_valid && access.rd_bank_0_id < resourceCount.U,
      UIntToOH(access.rd_bank_0_id, resourceCount),
      0.U(resourceCount.W)
    )
    val read1     = Mux(
      access.rd_bank_1_valid && access.rd_bank_1_id < resourceCount.U,
      UIntToOH(access.rd_bank_1_id, resourceCount),
      0.U(resourceCount.W)
    )
    val readWrite = Mux(
      access.rd_wr_valid && access.rd_wr_id < resourceCount.U,
      UIntToOH(access.rd_wr_id, resourceCount),
      0.U(resourceCount.W)
    )
    entryReads(slot)    := read0 | read1 | readWrite
    entryWrites(slot)   := Mux(
      access.wr_bank_valid && access.wr_bank_id < resourceCount.U,
      UIntToOH(access.wr_bank_id, resourceCount),
      0.U(resourceCount.W)
    )
    entryIsConfig(slot) := io.entries(slot).cmd.cmd.raw_inst(31, 25) === MSET_BITPAT
  }

  val allocAccess = io.alloc.bankAccess

  val allocRead0 = Mux(
    allocAccess.rd_bank_0_valid && allocAccess.rd_bank_0_id < resourceCount.U,
    UIntToOH(allocAccess.rd_bank_0_id, resourceCount),
    0.U(resourceCount.W)
  )

  val allocRead1 = Mux(
    allocAccess.rd_bank_1_valid && allocAccess.rd_bank_1_id < resourceCount.U,
    UIntToOH(allocAccess.rd_bank_1_id, resourceCount),
    0.U(resourceCount.W)
  )

  val allocReadWrite = Mux(
    allocAccess.rd_wr_valid && allocAccess.rd_wr_id < resourceCount.U,
    UIntToOH(allocAccess.rd_wr_id, resourceCount),
    0.U(resourceCount.W)
  )

  val allocReads = allocRead0 | allocRead1 | allocReadWrite

  val allocWrites = Mux(
    allocAccess.wr_bank_valid && allocAccess.wr_bank_id < resourceCount.U,
    UIntToOH(allocAccess.wr_bank_id, resourceCount),
    0.U(resourceCount.W)
  )

  val allocIsConfig = io.alloc.cmd.raw_inst(31, 25) === MSET_BITPAT

  val unrenamed =
    if (frontendParam.bank_alias_enable && memParam.bankLayoutEnable)
      ~((BigInt(1) << memParam.virtualBankCount) - 1).U(resourceCount.W)
    else Fill(resourceCount, 1.U(1.W))

  val dependency = Wire(Vec(robDepth, Bool()))
  for (older <- 0 until robDepth) {
    val olderAccess    = io.entries(older).cmd.bankAccess
    val olderUse       = entryReads(older) | entryWrites(older)
    val bankConflict   =
      (((allocReads & entryWrites(older)) | (allocWrites & olderUse)) & unrenamed).orR ||
        (allocAccess.wr_update && ((allocReads & entryWrites(older)) | (allocWrites & olderUse)).orR) ||
        (olderAccess.wr_update && (allocReads & entryWrites(older)).orR)
    val configConflict = (allocIsConfig && entryIsConfig(older)) ||
      ((allocIsConfig || entryIsConfig(older)) && ((allocReads | allocWrites) & olderUse).orR)
    dependency(older) :=
      (io.valid(older) && !io.complete(older) && bankConflict) ||
        (io.valid(older) && configConflict) ||
        (io.valid(older) && !io.complete(older) && io.alloc.cmd.raw_inst(31, 25) === MvoverISA.Funct.U)
  }

  io.dependencies := dependency.asUInt
  io.movementLive := VecInit((0 until robDepth).map(slot =>
    io.valid(slot) && !io.complete(slot) && io.entries(slot).cmd.cmd.raw_inst(31, 25) === MvoverISA.Funct.U
  )).asUInt.orR
}
