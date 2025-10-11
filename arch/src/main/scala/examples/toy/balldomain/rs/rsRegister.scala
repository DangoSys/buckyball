package examples.toy.balldomain.rs

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import examples.BuckyBallConfigs.CustomBuckyBallConfig
import framework.builtin.frontend.rs.{BallReservationStation, BallRsRegist}

/**
 * Ball RS模块 - 参考BBus机制，管理Ball设备的注册和连接
 */
class BallRSModule(implicit b: CustomBuckyBallConfig, p: Parameters) extends BallReservationStation(
  // 定义要注册的Ball设备信息
  Seq(
    BallRsRegist(ballId = 0, ballName = "VecBall"),
    BallRsRegist(ballId = 1, ballName = "MatrixBall"),
    BallRsRegist(ballId = 2, ballName = "Im2colBall"),
    BallRsRegist(ballId = 3, ballName = "TransposeBall"),
    BallRsRegist(ballId = 4, ballName = "GeluBall"),
    BallRsRegist(ballId = 5, ballName = "LayerNormBall"),
    BallRsRegist(ballId = 6, ballName = "SoftmaxBall"),
  )
) {
  override lazy val desiredName = "BallRSModule"
}
