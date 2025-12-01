package examples.toy.balldomain.bbus

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import examples.BuckyballConfigs.CustomBuckyballConfig
import framework.bbus.BBus

/**
 * BBusModule - Ball bus module that directly extends BBus
 */
class BBusModule(implicit b: CustomBuckyballConfig, p: Parameters) extends BBus (
  // Define Ball device generators to register
  Seq(
    () => new prototype.vector.VecBall(0),
    () => new prototype.matrix.MatrixBall(1),
    () => new prototype.im2col.Im2colBall(2),
    () => new prototype.transpose.TransposeBall(3),
    () => new prototype.relu.ReluBall(4),
    () => new examples.toy.balldomain.emptyball.EmptyBall(5),
    () => new prototype.transfer.TransferBall(6)
  )
) {
  override lazy val desiredName = "BBusModule"
}
