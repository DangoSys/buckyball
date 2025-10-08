package examples.toy.balldomain.bbus

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import examples.BuckyBallConfigs.CustomBuckyBallConfig
import framework.bbus.BBus
import examples.toy.balldomain.im2colball.Im2colBall

/** BBusModule - 直接继承BBus的Ball总线模块
  */
class BBusModule(implicit b: CustomBuckyBallConfig, p: Parameters)
    extends BBus(
      // 定义要注册的Ball设备生成器
      Seq(
        () => new examples.toy.balldomain.vecball.VecBall(0),
        () => new examples.toy.balldomain.matrixball.MatrixBall(1),
        () => new examples.toy.balldomain.im2colball.Im2colBall(2),
        () => new examples.toy.balldomain.transposeball.TransposeBall(3),
        () =>
          new examples.toy.balldomain.reluball.ReluBall(4) // Ball ID 4 - 新添加
      )
    ) {
  override lazy val desiredName = "BBusModule"
}
