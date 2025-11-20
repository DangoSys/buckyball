# BuckyBall Prototype Accelerators

This directory contains prototype implementations of various domain-specific computation accelerators in the BuckyBall framework, covering hardware accelerator designs for machine learning, numerical computation, and data processing domains.

## Directory Structure

```
prototype/
├── format/      - Data format conversion accelerators
├── im2col/      - Image-to-column transformation accelerator
├── matrix/      - Matrix computation accelerators
├── relu/        - ReLU activation accelerator
├── transpose/   - Matrix transpose accelerator
└── vector/      - Vector processing unit
```

## Accelerator Components

### format/ - Data Format Processing
Implements hardware acceleration for various data format conversions and arithmetic operations:
- **Arithmetic.scala**: Custom arithmetic operation units
- **Dataformat.scala**: Data format conversion and encoding

**Key Features**:
- Support for multiple data formats (INT8, FP16, FP32, BBFP)
- Abstract arithmetic interface for extensibility
- Concrete implementations for different data types

**Use Cases**:
- Floating-point format conversion
- Fixed-point arithmetic optimization
- Data compression and decompression
- Mixed-precision computation

### im2col/ - Image Processing Acceleration
Specialized accelerator for im2col operations in convolutional neural networks:
- **im2col.scala**: Hardware implementation of image-to-column matrix transformation

**Key Features**:
- Configurable kernel size and stride
- Efficient data reorganization for convolution
- Pipeline-based processing for high throughput
- Support for different input dimensions

**Use Cases**:
- CNN convolution layer acceleration
- Image preprocessing pipeline
- Feature extraction optimization
- Memory-efficient convolution implementation

### matrix/ - Matrix Computation Engine
Matrix computation accelerator implementation with multiple modules:

**Core Components**:
- **bbfpIns_decode.scala**: Instruction decoder for matrix operations
- **bbfp_load.scala**: Data loading unit for matrix operands
- **bbfp_ex.scala**: Execution unit for matrix multiplication
- **bbfp_pe.scala**: Processing Element (PE) array implementation
- **bbfp_control.scala**: Control logic for matrix operations

**PE Array Architecture**:
- **BBFP_PE**: Individual processing element with weight stationary mode
- **BBFP_PE_Array2x2**: 2×2 PE array building block
- **BBFP_PE_Array16x16**: 16×16 PE array for high-performance computing
- Systolic array dataflow for efficient matrix multiplication

**Supported Formats**:
- INT8 integer arithmetic
- FP16 half-precision floating-point
- FP32 single-precision floating-point
- BBFP (Brain Floating Point) custom format

**Use Cases**:
- Deep learning training and inference
- Scientific computing acceleration
- Linear algebra operations
- High-performance GEMM operations

### relu/ - ReLU Activation
Efficient hardware implementation of ReLU (Rectified Linear Unit) activation:
- **Relu.scala**: Pipelined ReLU accelerator

**Key Features**:
- Element-wise ReLU computation
- Configurable tile size
- Pipeline-based processing
- Integrated with scratchpad memory

**Use Cases**:
- Neural network activation layers
- Non-linear transformation
- Post-convolution activation

### transpose/ - Matrix Transpose
Efficient hardware implementation for matrix transpose operations:
- **Transpose.scala**: Matrix transpose accelerator

**Key Features**:
- Tile-based transpose for large matrices
- Optimized memory access patterns
- Configurable tile size
- Pipeline-based implementation

**Use Cases**:
- Matrix operation preprocessing
- Data reorganization and transformation
- Memory access pattern optimization
- Transpose in GEMM operations

### vector/ - Vector Processing Unit
Vector processing architecture supporting SIMD and multi-threading:

**Core Components**:
- **VecUnit.scala**: Vector processor top-level module
- **VecCtrlUnit.scala**: Vector control unit for instruction dispatch
- **VecLoadUnit.scala**: Vector load unit for data fetching
- **VecEXUnit.scala**: Vector execution unit with multiple functional units
- **VecStoreUnit.scala**: Vector store unit for result write-back

**Submodules**:
- **bond/**: Binding and synchronization mechanisms
  - Various bond types (VSSBond, VVVBond, VSVBond, VVSBond, VVBond)
  - Operand routing and data distribution

- **op/**: Vector operation implementations
  - AddOp, MulOp, CascadeOp, SelectOp, etc.
  - Arithmetic and logical operations

- **thread/**: Multi-threading support
  - Thread-level parallelism
  - Warp-based execution model

- **warp/**: Thread bundle management (MeshWarp)
  - 16×16 PE mesh for vector operations
  - Parallel execution of vector instructions

**Architecture Highlights**:
- Configurable number of PEs and threads
- Support for various vector operations (add, mul, cascade, select)
- Flexible data routing through bond mechanisms
- High parallelism with warp-level execution

**Use Cases**:
- Parallel numerical computation
- Signal processing acceleration
- High-performance computing applications
- SIMD-style data processing

## Design Features

### Modular Design
Each accelerator adopts modular design for:
- Independent development and testing
- Flexible composition and configuration
- Performance tuning and extension
- Easy integration with BuckyBall framework

### Pipeline Architecture
Most accelerators use deep pipeline design:
- Improved throughput and frequency
- Support for continuous data stream processing
- Optimized resource utilization
- Latency hiding through pipelining

### Configurable Parameters
Support rich configuration parameters:
- Data width and precision
- Parallelism and pipeline depth
- Cache size and organization
- Interface protocol and timing

## Integration Method

### Blink Protocol Interface
All Ball accelerators implement the Blink protocol interface:
```scala
class CustomBall(implicit b: CustomBuckyBallConfig, p: Parameters)
  extends Module with BallRegist {
  val io = IO(new BlinkIO)
  def ballId = <unique_id>.U
  def Blink = // Implement Blink protocol
}
```

**Blink Interface Components**:
- **cmdReq**: Command request interface with rob_id tracking
- **cmdResp**: Command response interface for completion signaling
- **status**: Status signals (ready, valid, idle, complete)
- **sramRead/Write**: SRAM interfaces for scratchpad and accumulator access

### Memory Interface
Support multiple memory access patterns:
- DMA bulk transfer through MemDomain
- Scratchpad direct access for low-latency operations
- Accumulator access for result accumulation
- Bank-aware memory access (op1 and op2 must access different banks)

### Configuration Integration
Parameterized through BuckyBall configuration system:
```scala
case class BaseConfig(
  veclane: Int = 16,        // Vector lane width
  numVecPE: Int = 16,       // Number of vector PEs
  numVecThread: Int = 16,   // Number of vector threads
  // ... more parameters
)
```

## Performance Optimization

### Data Locality
- Optimize data access patterns for spatial and temporal locality
- Reduce memory bandwidth requirements through data reuse
- Improve cache hit rate with tile-based processing
- Scratchpad memory for frequently accessed data

### Parallel Processing
- Multi-level parallelism design
  - Instruction-level parallelism (ILP) through pipelining
  - Data-level parallelism (DLP) through vector operations
  - Thread-level parallelism (TLP) through multiple warps
- Pipeline parallelism for continuous data flow
- Data parallelism through PE arrays

### Resource Sharing
- Arithmetic unit reuse across different operations
- Storage resource sharing between modules
- Control logic optimization for area efficiency
- Flexible routing for resource utilization

## Verification and Testing

Each accelerator comes with corresponding test cases:
- Functional correctness verification
- Performance benchmark testing
- Boundary condition checking
- Random test generation
- Integration testing with complete system

## Development Guidelines

### Adding New Accelerators

**Steps**:
1. Implement Ball device with BallRegist trait
2. Define Blink protocol interfaces
3. Implement computation logic
4. Add SRAM access logic (respect bank constraints)
5. Register in BBus and Ball RS

**Example Template**:
```scala
class NewBall(implicit b: CustomBuckyBallConfig, p: Parameters)
  extends Module with BallRegist {
  val io = IO(new BlinkIO)

  def ballId = <unique_id>.U
  def Blink = io

  // State machine
  val sIdle :: sCompute :: sComplete :: Nil = Enum(3)
  val state = RegInit(sIdle)

  // Computation logic
  switch(state) {
    is(sIdle) {
      when(io.cmdReq.fire) {
        state := sCompute
      }
    }
    is(sCompute) {
      // Perform computation
      when(done) {
        state := sComplete
      }
    }
    is(sComplete) {
      io.cmdResp.valid := true.B
      state := sIdle
    }
  }
}
```

### Performance Optimization Tips

1. **Memory Access**:
   - Group memory accesses to same bank
   - Use streaming access patterns
   - Minimize random access

2. **Pipeline Design**:
   - Balance pipeline stages
   - Add registers for timing closure
   - Use buffering for throughput

3. **Resource Utilization**:
   - Share expensive resources (multipliers, dividers)
   - Use LUTs for simple operations
   - Optimize control logic

### Common Pitfalls

1. **Bank Conflict**: op1 and op2 accessing same bank - violates design constraint
2. **ROB ID Tracking**: Must forward rob_id from request to response
3. **Ready/Valid Protocol**: Carefully implement handshake to avoid deadlock
4. **Iteration Count**: Properly handle iteration for multi-row operations

## Related Documentation

- [Format Conversion](format/README.md) - Data format details
- [Im2col Implementation](im2col/README.md) - Im2col accelerator
- [Matrix Operations](matrix/README.md) - Matrix computation
- [ReLU Activation](relu/README.md) - ReLU implementation
- [Transpose Operations](transpose/README.md) - Matrix transpose
- [Vector Processing](vector/README.md) - Vector unit architecture
- [Blink Protocol](../framework/blink/README.md) - Ball protocol specification

## Future Enhancements

Potential areas for extension:
- Support for additional data formats (INT4, BF16)
- Advanced matrix operations (SVD, QR decomposition)
- Fused operations (Conv+ReLU, GEMM+BiasAdd)
- Dynamic reconfiguration for different workloads
- Power management and clock gating
- Advanced synchronization mechanisms
