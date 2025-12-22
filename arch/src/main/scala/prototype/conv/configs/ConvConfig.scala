package prototype.conv.configs

import chisel3._
import chisel3.experimental.SerializableModuleParameter
import examples.toy.balldomain.BallDomainParam

object ConvConfig {
  implicit def rw: upickle.default.ReadWriter[ConvConfig] = upickle.default.macroRW

  def fromBallDomain(ballParam: BallDomainParam): ConvConfig = {
    ConvConfig(
      ballParam = ballParam
    )
  }

  /**
   * Load from JSON file
   */
  def fromJson(path: String): ConvConfig = {
    val jsonStr = scala.io.Source.fromFile(path).mkString
    upickle.default.read[ConvConfig](jsonStr)
  }

  /**
   * Save to JSON file
   */
  def toJson(config: ConvConfig, path: String): Unit = {
    val jsonStr = upickle.default.write(config, indent = 2)
    val writer  = new java.io.FileWriter(path)
    try {
      writer.write(jsonStr)
    } finally {
      writer.close()
    }
  }

}

case class ConvConfig(
  ballParam: BallDomainParam)
    extends SerializableModuleParameter {
  val bankNum     = ballParam.numBanks
  val bankEntries = ballParam.bankEntries
  val bankWidth   = ballParam.bankWidth
  val bankMaskLen = ballParam.bankMaskLen
  val rob_entries = ballParam.rob_entries
  // InputNum and inputWidth are Ball-specific, not in BallDomainParam
  val InputNum    = 16 // Default value
  val inputWidth  = 8  // Default value

  override def toString: String =
    s"""ConvConfig
       |  Bank num: $bankNum
       |  Bank entries: $bankEntries
       |  Bank width: $bankWidth
       |  Bank mask length: $bankMaskLen
       |  ROB entries: $rob_entries
       |  Input num: $InputNum
       |  Input width: $inputWidth
       |""".stripMargin
}
