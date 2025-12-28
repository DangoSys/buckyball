package sims.verify

import chisel3._
import _root_.circt.stage.ChiselStage
import org.chipsalliance.cde.config.{Config, Field, Parameters}
import framework.top.GlobalConfig
import framework.balldomain.blink.BlinkIO
import framework.balldomain.prototype.vector.VecBall
// import framework.balldomain.prototype.matrix.MatrixBall
// import framework.balldomain.prototype.matrix.configs.MatrixConfig
// import framework.balldomain.prototype.transpose.TransposeBall
// import framework.balldomain.prototype.transpose.configs.TransposeConfig
// import framework.balldomain.prototype.im2col.Im2colBall
// import framework.balldomain.prototype.im2col.configs.Im2colConfig
// import framework.balldomain.prototype.relu.ReluBall
// import framework.balldomain.prototype.nnlut.NNLutBall
// import framework.balldomain.prototype.nnlut.configs.NNLutConfig

// Ball type definitions
sealed trait BallType
case object VecBallType       extends BallType
case object MatrixBallType    extends BallType
case object TransposeBallType extends BallType
case object Im2colBallType    extends BallType
case object ReluBallType      extends BallType
case object NNLutBallType     extends BallType

// Config Key
case object TargetBallKey extends Field[BallType](VecBallType)

// TargetBall - directly instantiate pre-packaged Ball
class TargetBall(implicit b: GlobalConfig, p: Parameters) extends Module {

  // Create BlinkIO with parameter
  val io = IO(new BlinkIO(b))

  p(TargetBallKey) match {
    case VecBallType       =>
      val ball = Module(new VecBall(b, 0)) // Use id=0 for VecBall
      io <> ball.io
    case MatrixBallType    =>
    // val ball = Module(new MatrixBall(MatrixConfig.fromBallDomain(ballParam), 0))
    // io <> ball.io
    case Im2colBallType    =>
    // val ball = Module(new Im2colBall(Im2colConfig.fromBallDomain(ballParam), 0))
    // io <> ball.io
    case TransposeBallType =>
    // val ball = Module(new TransposeBall(TransposeConfig.fromBallDomain(ballParam), 0))
    // io <> ball.io
    case ReluBallType      =>
    // val ball = Module(new ReluBall(ReluConfig.fromBallDomain(ballParam), 0))
    // io <> ball.io
    case NNLutBallType     =>
    // val ball = Module(new NNLutBall(NNLutConfig.fromBallDomain(ballParam), 0))
    // io <> ball.io
    case _                 => throw new scala.MatchError("TargetBall does not handle this ball type")
  }
}

class WithTargetBall(ballType: BallType)
    extends Config((site, here, up) => {
      case TargetBallKey => ballType
    })

class CustomBallTopConfig(ballType: BallType)
    extends Config(
      // new WithBlink ++
      new WithTargetBall(ballType)
    )

object BallTopMain extends App {

  // Select Ball type from command line arguments
  val ballType = if (args.isEmpty) {
    println("Usage: BallTopMain <ball-type> [firtool-opts...]")
    println("Available ball types: vecball, matrixball, transposeball, im2colball, reluball, nnlutball")
    println("Using default: vecball")
    VecBallType
  } else {
    args(0).toLowerCase match {
      case "vecball"       => VecBallType
      case "matrixball"    => MatrixBallType
      case "transposeball" => TransposeBallType
      case "im2colball"    => Im2colBallType
      case "reluball"      => ReluBallType
      case "nnlutball"     => NNLutBallType
      case other           =>
        println(s"Unknown ball type: $other, using vecball")
        VecBallType
    }
  }

  implicit val b:      GlobalConfig = GlobalConfig()
  implicit val params: Parameters   = new Config(new CustomBallTopConfig(ballType))

  ChiselStage.emitSystemVerilogFile(
    new TargetBall()(b, params),
    // Remaining parameters passed to firtool
    firtoolOpts = args.drop(1),
    args = Array.empty
  )
}
