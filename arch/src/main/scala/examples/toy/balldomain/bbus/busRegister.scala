package examples.toy.balldomain.bbus

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import examples.BuckyBallConfigs.CustomBuckyBallConfig
import framework.bbus.BBus

/**
 * BBusModule - 直接继承BBus的Ball总线模块
 */
class BBusModule(implicit b: CustomBuckyBallConfig, p: Parameters) extends BBus (
  // 定义要注册的Ball设备生成器
  Seq(
    () => new prototype.vector.VecBall(0),
    () => new prototype.matrix.MatrixBall(1),
    () => new prototype.im2col.Im2colBall(2),
    () => new prototype.transpose.TransposeBall(3),
    () => new prototype.nagisa.gelu.GeluBall(4),
    () => new prototype.nagisa.layernorm.LayerNormBall(5),
    () => new prototype.nagisa.softmax.SoftmaxBall(6),
    () => new prototype.relu.ReluBall(7)
  )
) {
  override lazy val desiredName = "BBusModule"
}
