package framework.memdomain.backend

import chisel3._
import chisel3.util._
import framework.memdomain.backend.banks.{SramReadIO, SramWriteIO}
import framework.top.GlobalConfig

class MemRequestIO(b: GlobalConfig) extends Bundle {
  val write    = Flipped(new SramWriteIO(b)) // midend sends write req into backend
  val read     = Flipped(new SramReadIO(b))  // midend sends read req into backend
  val bank_id  = Output(UInt(log2Up(b.memDomain.bankNum).W))
  val group_id = Output(UInt(3.W))
}
