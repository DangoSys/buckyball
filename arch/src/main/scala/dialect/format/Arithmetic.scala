package dialect.format

import chisel3._
import chisel3.util._

// 算术类型类
abstract class Arithmetic[T <: Data] {
  def add(x: T, y: T): T
  def sub(x: T, y: T): T  
  def mul(x: T, y: T): T
  def div(x: T, y: T): T
  def gt(x: T, y: T): Bool
}

// UInt 算术实现
class UIntArithmetic extends Arithmetic[UInt] {
  override def add(x: UInt, y: UInt): UInt = x + y
  override def sub(x: UInt, y: UInt): UInt = x - y
  override def mul(x: UInt, y: UInt): UInt = x * y
  override def div(x: UInt, y: UInt): UInt = Mux(y =/= 0.U, x / y, 0.U)
  override def gt(x: UInt, y: UInt): Bool = x > y
}

// 工厂
object ArithmeticFactory {
  def createArithmetic[T <: Data](dataType: T): Arithmetic[T] = {
    dataType match {
      case _: UInt => new UIntArithmetic().asInstanceOf[Arithmetic[T]]
      case _ => throw new IllegalArgumentException(s"Unsupported data type: ${dataType.getClass}")
    }
  }
} 