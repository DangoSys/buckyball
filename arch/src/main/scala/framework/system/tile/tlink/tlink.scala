package framework.system.tile.tlink

import chisel3._
import chisel3.util._
import hier.tile.memory.CoreInterrupts
import framework.memdomain.frontend.mem.dma.DmaStatus
import memcore.bus.{axi4, chi}
import memcore.bus.chi.RequesterPort
import memcore.bus.chi.snf.{LineRequest, LineResponse}
import memcore.memory.cpu.{CpuMemParams, UncachedRequest, UncachedResponse}

class SharedPort(p: axi4.Params) extends Bundle {
  val tx = new axi4.Port(p)
  val rx = Flipped(new axi4.Port(p))
}

/** Tile boundary: coherent CPU control traffic and one independent DDR data port. */
class TLinkIO(
  main:          Boolean,
  cpus:          Int,
  executions:    Int,
  controls:      Int,
  bus:           chi.Params,
  axi:           axi4.Params,
  sharedStorage: Boolean)
    extends Bundle {
  val cp               = CpuMemParams(bus, tagBits = 6)
  val tileId           = Input(UInt(32.W))
  val t2t              = Option.when(sharedStorage)(new SharedPort(axi))
  val hartIds          = Input(Vec(cpus, UInt(64.W)))
  val executionIds     = Input(Vec(executions, UInt(64.W)))
  val resetVector      = Input(Vec(cpus, UInt(64.W)))
  val time             = Input(UInt(64.W))
  val interrupts       = Input(Vec(cpus, new CoreInterrupts))
  val uncachedRequest  = Vec(cpus, Decoupled(new UncachedRequest(cp)))
  val uncachedResponse = Vec(cpus, Flipped(Decoupled(new UncachedResponse(cp))))
  val control          = Vec(if (main) 0 else 2 * cpus, new RequesterPort(bus))
  val remoteControl    = Vec(if (main) 2 * (controls - cpus) else 0, Flipped(new RequesterPort(bus)))
  val backingRequest   = Vec(if (main) 1 else 0, Decoupled(new LineRequest(bus)))
  val backingResponse  = Vec(if (main) 1 else 0, Flipped(Decoupled(new LineResponse(bus))))
  val mem              = new axi4.Port(axi)
  val failure          = Output(Vec(cpus, Valid(new DmaStatus)))
  val workDrained      = Output(Vec(cpus, Bool()))
  val retired          = Output(Vec(cpus, Bool()))
  val retiredPc        = Output(Vec(cpus, UInt(64.W)))
  val trapped          = Output(Vec(cpus, Bool()))
  val trapCause        = Output(Vec(cpus, UInt(64.W)))
  val trapValue        = Output(Vec(cpus, UInt(64.W)))
  val trapPc           = Output(Vec(cpus, UInt(64.W)))
}

trait HasTLink {
  def tlink: TLinkIO
}
