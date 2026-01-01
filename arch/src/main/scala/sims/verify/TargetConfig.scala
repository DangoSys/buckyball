package sims.verify

import chisel3._
import _root_.circt.stage.ChiselStage
import framework.top.GlobalConfig
import framework.balldomain.blink.{BallRegist, BlinkIO}
import framework.balldomain.prototype.vector.VecBall
import framework.balldomain.prototype.relu.ReluBall
// import framework.balldomain.prototype.matrix.MatrixBall
// import framework.balldomain.prototype.transpose.TransposeBall
// import framework.balldomain.prototype.im2col.Im2colBall
// import framework.balldomain.prototype.nnlut.NNLutBall

sealed trait BallType
case object VecBallType       extends BallType
case object MatrixBallType    extends BallType
case object TransposeBallType extends BallType
case object Im2colBallType    extends BallType
case object ReluBallType      extends BallType
case object NNLutBallType     extends BallType

class TargetBall(ballType: BallType, b: GlobalConfig) extends Module {

  // Determine bandwidth from config first
  val (inBW, outBW) = ballType match {
    case VecBallType  =>
      val mapping = b.ballDomain.ballIdMappings.find(_.ballName == "VecBall")
        .getOrElse(throw new IllegalArgumentException("VecBall not found in config"))
      (mapping.inBW, mapping.outBW)
    case ReluBallType =>
      val mapping = b.ballDomain.ballIdMappings.find(_.ballName == "ReluBall")
        .getOrElse(throw new IllegalArgumentException("ReluBall not found in config"))
      (mapping.inBW, mapping.outBW)
    case _            => (2, 4) // Default bandwidth
  }

  val io = IO(new BlinkIO(b, inBW, outBW))

  ballType match {
    case VecBallType       =>
      val ball = Module(new VecBall(b))
      io <> ball.io
    case ReluBallType      =>
      val ball = Module(new ReluBall(b))
      io <> ball.io
    case MatrixBallType    => throw new IllegalArgumentException("MatrixBall not implemented")
    case Im2colBallType    => throw new IllegalArgumentException("Im2colBall not implemented")
    case TransposeBallType => throw new IllegalArgumentException("TransposeBall not implemented")
    case NNLutBallType     => throw new IllegalArgumentException("NNLutBall not implemented")
    case _                 => throw new scala.MatchError("TargetBall does not handle this ball type")
  }
}

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

  val b: GlobalConfig = GlobalConfig()

  ChiselStage.emitSystemVerilogFile(
    new TargetBall(ballType, b),
    firtoolOpts = args.drop(1),
    args = Array.empty
  )
}
