package prototype.matrix.configs

import chisel3._
import chisel3.experimental.SerializableModuleParameter
import examples.toy.balldomain.BallDomainParam

object MatrixConfig {
  implicit def rw: upickle.default.ReadWriter[MatrixConfig] = upickle.default.macroRW

  def fromBallDomain(ballParam: BallDomainParam): MatrixConfig = {
    MatrixConfig(
      ballParam = ballParam
    )
  }

  /**
   * Load from JSON file
   */
  def fromJson(path: String): MatrixConfig = {
    val jsonStr = scala.io.Source.fromFile(path).mkString
    upickle.default.read[MatrixConfig](jsonStr)
  }

  /**
   * Save to JSON file
   */
  def toJson(config: MatrixConfig, path: String): Unit = {
    val jsonStr = upickle.default.write(config, indent = 2)
    val writer  = new java.io.FileWriter(path)
    try {
      writer.write(jsonStr)
    } finally {
      writer.close()
    }
  }

}

case class MatrixConfig(
  ballParam: BallDomainParam)
    extends SerializableModuleParameter {
  // Derived parameters
  val bankNum     = ballParam.numBanks
  val bankEntries = ballParam.bankEntries
  val bankWidth   = ballParam.bankWidth
  val bankMaskLen = ballParam.bankMaskLen
  val rob_entries = ballParam.rob_entries
  // InputNum and inputWidth are Ball-specific, not in BallDomainParam
  val InputNum    = 16 // Default value
  val inputWidth  = 8  // Default value

  override def toString: String =
    s"""MatrixConfig
       |  Bank num: $bankNum
       |  Bank entries: $bankEntries
       |  Bank width: $bankWidth
       |  Bank mask length: $bankMaskLen
       |  ROB entries: $rob_entries
       |  Input num: $InputNum
       |  Input width: $inputWidth
       |""".stripMargin
}
