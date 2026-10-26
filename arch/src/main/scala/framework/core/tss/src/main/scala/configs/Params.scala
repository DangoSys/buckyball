package framework.core.tss.configs

import framework.ip.memory.spm.{Params => MemoryParams}
import upickle.default.{macroRW, ReadWriter}

case class Params(memory: MemoryParams, ports: Int) {
  require(ports > 0)
}

object Params {
  implicit val memoryRW: ReadWriter[MemoryParams] = macroRW
  implicit val rw:       ReadWriter[Params]       = macroRW
}
