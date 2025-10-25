package sims.verify

import chisel3._
import _root_.circt.stage.ChiselStage
import org.chipsalliance.cde.config.{Config, Parameters, Field}
import examples.BuckyBallConfigs.CustomBuckyBallConfig
import framework.blink.Blink
import prototype.vector.VecBall
import prototype.matrix.MatrixBall
import prototype.transpose.TransposeBall
import prototype.im2col.Im2colBall
import prototype.relu.ReluBall

// Ball类型定义
sealed trait BallType
case object VecBallType extends BallType
case object MatrixBallType extends BallType
case object TransposeBallType extends BallType
case object Im2colBallType extends BallType
case object ReluBallType extends BallType

// Config Key
case object TargetBallKey extends Field[BallType](VecBallType)

// TargetBall - 直接实例化已封装好的Ball
class TargetBall(implicit b: CustomBuckyBallConfig, p: Parameters) extends Module {
  val io = IO(new Blink)

  p(TargetBallKey) match {
    case VecBallType =>
      val ball = Module(new VecBall(0))
      io <> ball.io
    case MatrixBallType =>
      val ball = Module(new MatrixBall(0))
      io <> ball.io
    case Im2colBallType =>
      val ball = Module(new Im2colBall(0))
      io <> ball.io
    case TransposeBallType =>
      val ball = Module(new TransposeBall(0))
      io <> ball.io
    case ReluBallType =>
      val ball = Module(new ReluBall(0))
      io <> ball.io
    case _ => throw new scala.MatchError("TargetBall does not handle this ball type")
  }
  override lazy val desiredName = "TargetBall"
}

// WithBlink配置 - 空配置，用于组合其他配置
// class WithBlink extends Config((site, here, up) => {
//   case _ => up(site)
// })

// ============================================================================
// Config 组合使用示例：
// new Config(new WithVecBall ++ new WithBlink)
// new Config(new WithMatrixBall ++ new WithBlink)
// new Config(new WithTransposeBall ++ new WithBlink)
// ============================================================================

class WithTargetBall(ballType: BallType) extends Config((site, here, up) => {
  case TargetBallKey => ballType
})

class CustomBallTopConfig(ballType: BallType) extends Config(
  // new WithBlink ++
  new WithTargetBall(ballType)
)

object BallTopMain extends App {
  // 从命令行参数选择 Ball 类型
  val ballType = if (args.isEmpty) {
    println("Usage: BallTopMain <ball-type> [firtool-opts...]")
    println("Available ball types: vecball, matrixball, transposeball, im2colball, reluball")
    println("Using default: vecball")
    VecBallType
  } else {
    args(0).toLowerCase match {
      case "vecball" => VecBallType
      case "matrixball" => MatrixBallType
      case "transpose" => TransposeBallType
      case "im2colball" => Im2colBallType
      case "reluball" => ReluBallType
      case other =>
        println(s"Unknown ball type: $other, using vecball")
        VecBallType
    }
  }

  implicit val config: CustomBuckyBallConfig = examples.CustomBuckyBallConfig()
  implicit val params: Parameters = new Config(new CustomBallTopConfig(ballType))

  ChiselStage.emitSystemVerilogFile(
    new TargetBall(),
    firtoolOpts = args.drop(1), // 剩余参数传给 firtool
    args = Array.empty
  )
}
