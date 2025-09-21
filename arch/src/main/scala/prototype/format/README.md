# 数据格式处理模块

## 概述

该目录实现了 BuckyBall 中的数据格式定义和算术运算抽象，提供统一的数据类型处理接口。位于 `arch/src/main/scala/prototype/format` 下，作为数据格式层，为其他原型加速器提供类型安全的数据格式支持。

实现的核心组件：
- **Dataformat.scala**: 数据格式定义和工厂类
- **Arithmetic.scala**: 算术运算类型类实现

## 代码结构

```
format/
├── Dataformat.scala  - 数据格式定义
└── Arithmetic.scala  - 算术运算抽象
```

### 文件依赖关系

**Dataformat.scala** (格式定义层)
- 定义 DataFormat 抽象类和具体格式实现
- 提供 DataFormatFactory 工厂类
- 实现 DataFormatParams 参数类

**Arithmetic.scala** (运算抽象层)
- 定义 Arithmetic 类型类接口
- 实现 UIntArithmetic 具体运算
- 提供 ArithmeticFactory 工厂类

## 模块说明

### Dataformat.scala

**主要功能**: 定义支持的数据格式类型

**格式定义**:
```scala
abstract class DataFormat {
  def width: Int
  def dataType: Data
  def name: String
}
```

**支持的格式**:
```scala
class INT8Format extends DataFormat {
  override def width: Int = 8
  override def dataType: Data = UInt(8.W)
  override def name: String = "INT8"
}

class FP16Format extends DataFormat {
  override def width: Int = 16
  override def dataType: Data = UInt(16.W)
  override def name: String = "FP16"
}

class FP32Format extends DataFormat {
  override def width: Int = 32
  override def dataType: Data = UInt(32.W)
  override def name: String = "FP32"
}
```

**工厂类**:
```scala
object DataFormatFactory {
  def create(formatType: String): DataFormat = formatType.toUpperCase match {
    case "INT8" => new INT8Format
    case "FP16" => new FP16Format
    case "FP32" => new FP32Format
    case _ => throw new IllegalArgumentException(...)
  }
}
```

**参数类**:
```scala
case class DataFormatParams(formatType: String = "INT8") {
  def format: DataFormat = DataFormatFactory.create(formatType)
  def width: Int = format.width
  def dataType: Data = format.dataType
}
```

### Arithmetic.scala

**主要功能**: 提供类型安全的算术运算抽象

**类型类定义**:
```scala
abstract class Arithmetic[T <: Data] {
  def add(x: T, y: T): T
  def sub(x: T, y: T): T
  def mul(x: T, y: T): T
  def div(x: T, y: T): T
  def gt(x: T, y: T): Bool
}
```

**UInt 实现**:
```scala
class UIntArithmetic extends Arithmetic[UInt] {
  override def add(x: UInt, y: UInt): UInt = x + y
  override def sub(x: UInt, y: UInt): UInt = x - y
  override def mul(x: UInt, y: UInt): UInt = x * y
  override def div(x: UInt, y: UInt): UInt = Mux(y =/= 0.U, x / y, 0.U)
  override def gt(x: UInt, y: UInt): Bool = x > y
}
```

**工厂类**:
```scala
object ArithmeticFactory {
  def createArithmetic[T <: Data](dataType: T): Arithmetic[T] = {
    dataType match {
      case _: UInt => new UIntArithmetic().asInstanceOf[Arithmetic[T]]
      case _ => throw new IllegalArgumentException(...)
    }
  }
}
```

## 使用方法

### 注意事项

1. **浮点支持**: FP16 和 FP32 目前使用 UInt 表示，后续可扩展为真正的浮点类型
2. **除零保护**: UInt 除法运算包含除零检查，返回 0 作为默认值
3. **类型安全**: 使用 Scala 类型系统确保运算的类型安全性
4. **扩展性**: 工厂模式支持添加新的数据格式和算术实现
5. **参数化**: DataFormatParams 提供便捷的参数化配置接口
