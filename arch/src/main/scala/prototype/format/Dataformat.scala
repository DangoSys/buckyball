package prototype.format

import chisel3._
import chisel3.util._

// Data format definition
abstract class DataFormat {
  def width:    Int
  def dataType: Data
  def name:     String
}

// INT8 format
class INT8Format extends DataFormat {
  override def width:    Int    = 8
  override def dataType: Data   = UInt(8.W)
  override def name:     String = "INT8"
}

// FP16 format
class FP16Format extends DataFormat {
  override def width:    Int    = 16
  // Temporarily use UInt representation, can be extended to Float type later
  override def dataType: Data   = UInt(16.W)
  override def name:     String = "FP16"
}

// FP32 format
class FP32Format extends DataFormat {
  override def width:    Int    = 32
  // Temporarily use UInt representation, can be extended to Float type later
  override def dataType: Data   = UInt(32.W)
  override def name:     String = "FP32"
}

// Data format factory
object DataFormatFactory {

  def create(formatType: String): DataFormat = formatType.toUpperCase match {
    case "INT8" => new INT8Format
    case "FP16" => new FP16Format
    case "FP32" => new FP32Format
    case _      => throw new IllegalArgumentException(s"Unsupported data format: $formatType")
  }

}

// Generic data format parameters
case class DataFormatParams(
  formatType: String = "INT8") {
  def format:   DataFormat = DataFormatFactory.create(formatType)
  def width:    Int        = format.width
  def dataType: Data       = format.dataType
}
