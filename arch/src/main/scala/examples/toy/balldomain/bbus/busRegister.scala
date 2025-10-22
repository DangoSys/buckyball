package examples.toy.balldomain.bbus

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import examples.BuckyBallConfigs.CustomBuckyBallConfig
import framework.bbus.BBus
import examples.toy.balldomain.im2colball.Im2colBall

/**
 * BBusModule - 直接继承BBus的Ball总线模块
 */
class BBusModule(implicit b: CustomBuckyBallConfig, p: Parameters) extends BBus (
  // 定义要注册的Ball设备生成器
  Seq(
    () => new examples.toy.balldomain.vecball.VecBall(0),
    () => new examples.toy.balldomain.matrixball.MatrixBall(1),
    () => new examples.toy.balldomain.im2colball.Im2colBall(2),
    () => new examples.toy.balldomain.transposeball.TransposeBall(3),
    () => new examples.toy.balldomain.geluball.GeluBall(4),
    () => new examples.toy.balldomain.layernormball.LayerNormBall(5),
    () => new examples.toy.balldomain.softmaxball.SoftmaxBall(6),
    () => new examples.toy.balldomain.reluball.ReluBall(7)
  )
) {
  override lazy val desiredName = "BBusModule"
}
