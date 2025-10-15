package prototype.format

import chisel3._
import chisel3.util._

// 数据格式定义
abstract class DataFormat {
  def width: Int
  def dataType: Data
  def name: String
}

// INT8 格式
class INT8Format extends DataFormat {
  override def width: Int = 8
  override def dataType: Data = UInt(8.W)
  override def name: String = "INT8"
}

// FP16 格式
class FP16Format extends DataFormat {
  override def width: Int = 16
  override def dataType: Data = UInt(16.W) // 暂时用UInt表示，后续可扩展为Float类型
  override def name: String = "FP16"
}

// FP32 格式
class FP32Format extends DataFormat {
  override def width: Int = 32
  override def dataType: Data = UInt(32.W) // 暂时用UInt表示，后续可扩展为Float类型
  override def name: String = "FP32"
}



// 数据格式工厂
object DataFormatFactory {
  def create(formatType: String): DataFormat = formatType.toUpperCase match {
    case "INT8" => new INT8Format
    case "FP16" => new FP16Format
    case "FP32" => new FP32Format
    case _ => throw new IllegalArgumentException(s"Unsupported data format: $formatType")
  }
}

// 泛型数据格式参数
case class DataFormatParams(
  formatType: String = "INT8"
) {
  def format: DataFormat = DataFormatFactory.create(formatType)
  def width: Int = format.width
  def dataType: Data = format.dataType
}
