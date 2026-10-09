package framework.rvv

import chisel3._

class ProgramWrite extends Bundle {
  val buffer  = Bool()
  val first   = Bool()
  val address = UInt(32.W)
  val data    = UInt(32.W)
}

class KernelLaunch extends Bundle {
  val iBuffer = Bool()
  val entry   = UInt(32.W)
  val end     = UInt(32.W)
  val stack   = UInt(64.W)
  val args    = Vec(8, UInt(64.W))
}

class KernelCompletion extends Bundle {
  val fault       = Bool()
  val pc          = UInt(32.W)
  val instruction = UInt(32.W)
  val cycles      = UInt(64.W)
  val cause       = UInt(32.W)
  val tval        = UInt(64.W)
  val fflags      = UInt(5.W)
  val vxsat       = Bool()
}

class VectorMemoryRequest extends Bundle {
  val address = UInt(64.W)
  val write   = Bool()
  val data    = UInt(64.W)
  val mask    = UInt(8.W)
  val size    = UInt(2.W)
}

class VectorMemoryResponse extends Bundle {
  val data  = UInt(64.W)
  val error = Bool()
}

class VectorIssue extends Bundle {
  val instruction = UInt(32.W)
  val scalar1     = UInt(64.W)
  val scalar2     = UInt(64.W)
  val floating    = UInt(64.W)
}

class VectorResult extends Bundle {
  val fault       = Bool()
  val cause       = UInt(32.W)
  val tval        = UInt(64.W)
  val scalarWrite = Bool()
  val scalarData  = UInt(64.W)
  val floatWrite  = Bool()
  val floatData   = UInt(64.W)
  val flags       = UInt(5.W)
  val saturated   = Bool()
}

class ImageLoad extends Bundle {
  val buffer = Bool()
  val bytes  = UInt(32.W)
}

class ImageResult extends Bundle {
  val fault      = Bool()
  val cause      = UInt(32.W)
  val tval       = UInt(32.W)
  val entry      = UInt(32.W)
  val textBytes  = UInt(32.W)
  val constBytes = UInt(32.W)
}

class RvvBallCommand extends Bundle {
  val funct7 = UInt(7.W)
  val rs1    = UInt(64.W)
  val rs2    = UInt(64.W)
}

class BankLayout(b: framework.top.GlobalConfig) extends Bundle {
  val readBank    = UInt(b.memDomain.vbankIdWidth.W)
  val writeBank   = UInt(b.memDomain.vbankIdWidth.W)
  val readGroups  = UInt(b.memDomain.groupCountWidth.W)
  val writeGroups = UInt(b.memDomain.groupCountWidth.W)
}
