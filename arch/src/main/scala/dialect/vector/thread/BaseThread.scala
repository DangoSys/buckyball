//===- BaseThread.scala - Level 1: Thread ---===//
package dialect.vector.thread

import chisel3._
import org.chipsalliance.cde.config._

// 参数定义
case class ThreadParam(lane: Int, attr: String, threadName: String, Op: OpParam)
case class OpParam(OpType: String, bondType: BondParam)
case class BondParam(bondType: String, inputWidth: Int = 8, outputWidth: Int = 32)

case object ThreadKey extends Field[Option[ThreadParam]](None)
case object ThreadOpKey extends Field[Option[OpParam]](None)
case object ThreadBondKey extends Field[Option[BondParam]](None)
case object ThreadMapKey extends Field[Map[String, ThreadParam]](Map.empty)

//===----------------------------------------------------------------------===//
// BaseThread 基类
//===----------------------------------------------------------------------===//
class BaseThread(implicit p: Parameters) extends Module {
  val io = IO(new Bundle {})
  val params = p
  val threadMap = p(ThreadMapKey)
  val threadParam = threadMap.getOrElse(
    p(ThreadKey).get.threadName,
    throw new Exception(s"ThreadParam not found for threadName: ${p(ThreadKey).get.threadName}")
  )
  val opParam = p(ThreadOpKey).get
  val bondParam = p(ThreadBondKey).get
  println(s"[Thread_${threadParam.threadName}] Op: ${opParam.OpType}, bond: ${bondParam.bondType}, Lanes: ${threadParam.lane}")
}


