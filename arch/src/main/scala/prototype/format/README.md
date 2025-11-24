# Data Format Processing Module

## Overview

This directory implements data format definitions and arithmetic operation abstractions in Buckyball, providing a unified data type processing interface. Located at `arch/src/main/scala/prototype/format`, it serves as the data format layer, providing type-safe data format support for other prototype accelerators.

Core components:
- **Dataformat.scala**: Data format definitions and factory classes
- **Arithmetic.scala**: Arithmetic operation type class implementations

## Code Structure

```
format/
├── Dataformat.scala  - Data format definitions
└── Arithmetic.scala  - Arithmetic operation abstractions
```

### File Dependencies

**Dataformat.scala** (Format definition layer)
- Defines DataFormat abstract class and concrete format implementations
- Provides DataFormatFactory factory class
- Implements DataFormatParams parameter class

**Arithmetic.scala** (Operation abstraction layer)
- Defines Arithmetic type class interface
- Implements UIntArithmetic concrete operations
- Provides ArithmeticFactory factory class

## Module Description

### Dataformat.scala

**Main functionality**: Defines supported data format types

**Format definition**:
```scala
abstract class DataFormat {
  def width: Int
  def dataType: Data
  def name: String
}
```

**Supported formats**:
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

**Factory class**:
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

**Parameter class**:
```scala
case class DataFormatParams(formatType: String = "INT8") {
  def format: DataFormat = DataFormatFactory.create(formatType)
  def width: Int = format.width
  def dataType: Data = format.dataType
}
```

### Arithmetic.scala

**Main functionality**: Provides type-safe arithmetic operation abstractions

**Type class definition**:
```scala
abstract class Arithmetic[T <: Data] {
  def add(x: T, y: T): T
  def sub(x: T, y: T): T
  def mul(x: T, y: T): T
  def div(x: T, y: T): T
  def gt(x: T, y: T): Bool
}
```

**UInt implementation**:
```scala
class UIntArithmetic extends Arithmetic[UInt] {
  override def add(x: UInt, y: UInt): UInt = x + y
  override def sub(x: UInt, y: UInt): UInt = x - y
  override def mul(x: UInt, y: UInt): UInt = x * y
  override def div(x: UInt, y: UInt): UInt = Mux(y =/= 0.U, x / y, 0.U)
  override def gt(x: UInt, y: UInt): Bool = x > y
}
```

**Factory class**:
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

## Usage

### Notes

1. **Floating-point support**: FP16 and FP32 currently use UInt representation, can be extended to true floating-point types later
2. **Division by zero protection**: UInt division operation includes division-by-zero check, returns 0 as default value
3. **Type safety**: Uses Scala type system to ensure operation type safety
4. **Extensibility**: Factory pattern supports adding new data formats and arithmetic implementations
5. **Parameterization**: DataFormatParams provides convenient parameterized configuration interface
