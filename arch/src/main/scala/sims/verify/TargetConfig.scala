package sims.verify

import chisel3._
import _root_.circt.stage.ChiselStage
import org.chipsalliance.cde.config.{Config, Field, Parameters}
import examples.BuckyballConfigs.CustomBuckyballConfig
import examples.toy.balldomain.BallDomainParam
import framework.balldomain.blink.Blink
import prototype.vector.VecBall
import prototype.vector.configs.VecConfig
import prototype.matrix.MatrixBall
import prototype.matrix.configs.MatrixConfig
import prototype.transpose.TransposeBall
import prototype.transpose.configs.TransposeConfig
import prototype.im2col.Im2colBall
import prototype.im2col.configs.Im2colConfig
import prototype.relu.ReluBall
import prototype.relu.configs.ReluConfig
import prototype.nnlut.NNLutBall
import prototype.nnlut.configs.NNLutConfig

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
class TargetBall(implicit b: CustomBuckyballConfig, p: Parameters) extends Module {
  // Create BallDomainParam from global config
  val ballParam = BallDomainParam.fromGlobal(b)

  // Create Blink IO with parameter
  val io = IO(new Blink(ballParam, ballParam.bankEntries, ballParam.bankWidth, ballParam.bankMaskLen))

  p(TargetBallKey) match {
    case VecBallType       =>
      val ball = Module(new VecBall(VecConfig.fromBallDomain(ballParam), 0))
      io <> ball.io
    case MatrixBallType    =>
      val ball = Module(new MatrixBall(MatrixConfig.fromBallDomain(ballParam), 0))
      io <> ball.io
    case Im2colBallType    =>
      val ball = Module(new Im2colBall(Im2colConfig.fromBallDomain(ballParam), 0))
      io <> ball.io
    case TransposeBallType =>
      val ball = Module(new TransposeBall(TransposeConfig.fromBallDomain(ballParam), 0))
      io <> ball.io
    case ReluBallType      =>
      val ball = Module(new ReluBall(ReluConfig.fromBallDomain(ballParam), 0))
      io <> ball.io
    case NNLutBallType     =>
      val ball = Module(new NNLutBall(NNLutConfig.fromBallDomain(ballParam), 0))
      io <> ball.io
    case _                 => throw new scala.MatchError("TargetBall does not handle this ball type")
  }
  override lazy val desiredName = "TargetBall"
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

  implicit val config: CustomBuckyballConfig = examples.CustomBuckyballConfig()
  implicit val params: Parameters            = new Config(new CustomBallTopConfig(ballType))

  ChiselStage.emitSystemVerilogFile(
    new TargetBall(),
    // Remaining parameters passed to firtool
    firtoolOpts = args.drop(1),
    args = Array.empty
  )
}
