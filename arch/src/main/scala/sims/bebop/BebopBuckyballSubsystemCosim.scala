package sims.bebop

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import org.chipsalliance.diplomacy.lazymodule._

import freechips.rocketchip.diplomacy.{AddressSet, IdRange}
import freechips.rocketchip.devices.tilelink.TLTestRAM
import freechips.rocketchip.rocket.{HStatus, MStatus}
import freechips.rocketchip.tilelink._

import framework.core.bbtile.{BarrierUnit, BuckyballAccelerator, RoCCCommandBB}
import framework.memdomain.backend.shared.SharedMemBackend
import framework.top.GlobalConfig

object BebopBuckyballSubsystemCosim {
  val custom3Opcode: UInt = 0x7b.U(7.W)

  def roccCmd(
    funct: UInt,
    xs1:   UInt,
    xs2:   UInt,
    xLen:  Int
  ): RoCCCommandBB = {
    val cmd = Wire(new RoCCCommandBB(xLen))
    val f7  = funct(6, 0)
    cmd.raw_inst := Cat(f7, 0.U(5.W), 0.U(5.W), 3.U(3.W), 0.U(5.W), custom3Opcode)
    cmd.pc       := 0.U(xLen.W)
    cmd.funct    := funct
    cmd.funct3   := 3.U
    cmd.rs2      := 0.U
    cmd.rs1      := 0.U
    cmd.xd       := true.B
    cmd.xs1      := true.B
    cmd.xs2      := true.B
    cmd.rd       := 0.U
    cmd.opcode   := custom3Opcode
    cmd.rs1Data  := xs1
    cmd.rs2Data  := xs2
    cmd
  }

}

/** Full toy-config accelerator (Frontend + 9 balls + MemDomain + GpDomain) for Spike–Verilator cosim. */
class BebopBuckyballSubsystemCosim(implicit p: Parameters) extends LazyModule {

  val b: GlobalConfig = GlobalConfig()

  val beatBytes = b.memDomain.dma_buswidth / 8
  val ramMask   = (BigInt(1) << b.memDomain.memAddrLen) - 1
  require((ramMask & (beatBytes - 1)) == beatBytes - 1, "TLTestRAM mask must align to dma beatBytes")

  val readerNode = TLClientNode(
    Seq(
      TLMasterPortParameters.v1(
        Seq(
          TLClientParameters(
            name = "bebop-cosim-reader",
            sourceId = IdRange(0, b.memDomain.dma_n_xacts)
          )
        )
      )
    )
  )

  val writerNode = TLClientNode(
    Seq(
      TLMasterPortParameters.v1(
        Seq(
          TLClientParameters(
            name = "bebop-cosim-writer",
            sourceId = IdRange(0, b.memDomain.dma_n_xacts)
          )
        )
      )
    )
  )

  val xbar = TLXbar()
  val ram  = LazyModule(new TLTestRAM(AddressSet(0x0, ramMask), beatBytes = beatBytes))

  ram.node := xbar
  xbar     := TLBuffer() := readerNode
  xbar     := TLBuffer() := writerNode

  lazy val module = new LazyModuleImp(this) {
    override def desiredName: String = "BebopBuckyballSubsystemCosim"

    val start = IO(Input(Bool()))
    val funct = IO(Input(UInt(7.W)))
    val xs1   = IO(Input(UInt(64.W)))
    val xs2   = IO(Input(UInt(64.W)))

    val done   = IO(Output(Bool()))
    val result = IO(Output(UInt(64.W)))

    val (tlReader, edge) = readerNode.out(0)
    val (tlWriter, _)    = writerNode.out(0)

    val acc = Module(new BuckyballAccelerator(b)(edge))
    acc.io.tl_reader <> tlReader
    acc.io.tl_writer <> tlWriter

    acc.io.sfence := false.B
    acc.io.hartid := 0.U(b.core.xLen.W)

    acc.io.ptw(0).req.ready  := true.B
    acc.io.ptw(0).resp.valid := false.B
    acc.io.ptw(0).resp.bits  := 0.U.asTypeOf(acc.io.ptw(0).resp.bits)
    acc.io.ptw(0).ptbr.mode  := 0.U
    acc.io.ptw(0).ptbr.asid  := 0.U
    acc.io.ptw(0).ptbr.ppn   := 0.U
    acc.io.ptw(0).hgatp.mode := 0.U
    acc.io.ptw(0).hgatp.asid := 0.U
    acc.io.ptw(0).hgatp.ppn  := 0.U
    acc.io.ptw(0).vsatp.mode := 0.U
    acc.io.ptw(0).vsatp.asid := 0.U
    acc.io.ptw(0).vsatp.ppn  := 0.U
    acc.io.ptw(0).status     := 0.U.asTypeOf(new MStatus())
    acc.io.ptw(0).hstatus    := 0.U.asTypeOf(new HStatus())
    acc.io.ptw(0).gstatus    := 0.U.asTypeOf(new MStatus())
    for (i <- 0 until b.core.nPMPs) {
      acc.io.ptw(0).pmp(i).cfg.l   := false.B
      acc.io.ptw(0).pmp(i).cfg.res := 0.U
      acc.io.ptw(0).pmp(i).cfg.a   := 0.U
      acc.io.ptw(0).pmp(i).cfg.x   := false.B
      acc.io.ptw(0).pmp(i).cfg.w   := false.B
      acc.io.ptw(0).pmp(i).cfg.r   := false.B
      acc.io.ptw(0).pmp(i).addr    := 0.U
      acc.io.ptw(0).pmp(i).mask    := 0.U
    }

    val shared = Module(new SharedMemBackend(b))
    for (ch <- 0 until b.memDomain.bankChannel) {
      shared.io.mem_req(ch) <> acc.io.shared_mem_req(ch)
    }
    shared.io.config <> acc.io.shared_config
    shared.io.query_vbank_id        := acc.io.shared_query_vbank_id
    acc.io.shared_query_group_count := shared.io.query_group_count

    val barrier = Module(new BarrierUnit(1))
    barrier.io.arrive(0)   := acc.io.barrier_arrive
    acc.io.barrier_release := barrier.io.release(0)

    acc.io.tlbExp(0).flush_skip  := false.B
    acc.io.tlbExp(0).flush_retry := false.B

    val sIdle :: sCmd :: sWait :: sDone :: Nil = Enum(4)
    val state                                  = RegInit(sIdle)

    val cmdReg = Reg(new RoCCCommandBB(b.core.xLen))
    val resReg = RegInit(0.U(64.W))

    done   := state === sDone
    result := resReg

    acc.io.cmd.valid  := state === sCmd
    acc.io.cmd.bits   := cmdReg
    acc.io.resp.ready := state === sWait

    val waitCycles = RegInit(0.U(24.W))
    when(state === sWait) {
      waitCycles := waitCycles + 1.U
    }.otherwise {
      waitCycles := 0.U
    }
    assert(waitCycles < (1 << 23).U, "BebopBuckyballSubsystemCosim: RoCC wait timeout")

    switch(state) {
      is(sIdle) {
        when(start) {
          cmdReg := BebopBuckyballSubsystemCosim.roccCmd(funct, xs1, xs2, b.core.xLen)
          state  := sCmd
        }
      }
      is(sCmd) {
        when(acc.io.cmd.fire) {
          state := sWait
        }
      }
      is(sWait) {
        when(acc.io.resp.fire) {
          resReg := acc.io.resp.bits.data.asUInt
          state  := sDone
        }.elsewhen(!acc.io.busy) {
          resReg := Cat(0.U(57.W), cmdReg.funct)
          state  := sDone
        }
      }
      is(sDone) {
        when(!start) {
          state := sIdle
        }
      }
    }
  }

}
