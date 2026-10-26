package framework.acc.gpdomain.memory.configs

case class Params(
  memoryPorts: Int,
  wordBits:    Int) {
  require(memoryPorts > 0)
  require(wordBits == 32 || wordBits == 64)

  val constBytes: Int = 4096
  val stackBytes: Int = 4096
}
