package framework.rvv

import chisel3._

class ProgramWrite extends Bundle {
  val buffer  = Bool()
  val first   = Bool()
  val address = UInt(32.W)
  val data    = UInt(32.W)
}

class DataWrite extends Bundle {
  val buffer  = Bool()
  val address = UInt(32.W)
  val data    = UInt(32.W)
  val mask    = UInt(4.W)
}

class KernelLaunch extends Bundle {
  val iBuffer = Bool()
  val dBuffer = Bool()
  val entry   = UInt(32.W)
  val end     = UInt(32.W)
  val args    = Vec(8, UInt(32.W))
}

class KernelCompletion extends Bundle {
  val fault       = Bool()
  val pc          = UInt(32.W)
  val instruction = UInt(32.W)
  val cycles      = UInt(64.W)
  val cause       = UInt(32.W)
  val tval        = UInt(32.W)
}

class VectorMemoryRequest extends Bundle {
  val address = UInt(32.W)
  val write   = Bool()
  val data    = UInt(32.W)
}

class VectorMemoryResponse extends Bundle {
  val data  = UInt(32.W)
  val error = Bool()
}

class VectorIssue extends Bundle {
  val instruction = UInt(32.W)
  val scalar1     = UInt(32.W)
  val scalar2     = UInt(32.W)
}

class VectorResult extends Bundle {
  val fault       = Bool()
  val cause       = UInt(32.W)
  val tval        = UInt(32.W)
  val scalarWrite = Bool()
  val scalarData  = UInt(32.W)
}
