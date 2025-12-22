package examples

import chisel3._
import framework.builtin.BaseConfig
import examples.toy.BuckyballToyConfig
import scala.io.Source
import upickle.default._

object BuckyballConfigs {
  val defaultConfig = BaseConfig()
  val toyConfig     = BuckyballToyConfig.defaultConfig

  // Actually used configuration
  val customConfig = toyConfig

  type CustomBuckyballConfig = BaseConfig

  /**
   * Load global config from JSON file
   * Usage: BuckyballConfigs.fromJson("path/to/config.json")
   */
  def fromJson(path: String): BaseConfig = {
    val jsonStr = Source.fromFile(path).mkString
    read[BaseConfig](jsonStr)
  }

  /**
   * Load global config from JSON string
   */
  def fromJsonString(json: String): BaseConfig =
    read[BaseConfig](json)

  /**
   * Save global config to JSON file
   */
  def toJsonFile(config: BaseConfig, path: String): Unit = {
    val jsonStr = write(config, indent = 2)
    java.nio.file.Files.write(
      java.nio.file.Paths.get(path),
      jsonStr.getBytes
    )
  }

}

// Get currently selected configuration
object CustomBuckyballConfig {
  import BuckyballConfigs._
  def apply(): CustomBuckyballConfig = BuckyballConfigs.customConfig

  /**
   * Create config from JSON file
   * Usage: CustomBuckyballConfig.fromJson("configs/my_config.json")
   */
  def fromJson(path: String): CustomBuckyballConfig = BuckyballConfigs.fromJson(path)
}
