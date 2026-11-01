package framework.device.peripherals.trace.configs

import buckyball.config.TraceHardware

case class TraceParams(base: BigInt, bytes: BigInt, stride: Int) {
  require(stride >= 96 && (stride & (stride - 1)) == 0)
  require(bytes > 0 && bytes % stride == 0)
}

object TraceParams {
  def fromProto(p: TraceHardware): TraceParams =
    TraceParams(BigInt(java.lang.Long.toUnsignedString(p.getBase)), BigInt(p.getBytes), p.getStride)
}
