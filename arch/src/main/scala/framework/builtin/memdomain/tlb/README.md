# TLB 模块 (Translation Lookaside Buffer)

## 概述

TLB 模块实现了虚拟地址到物理地址的转换缓存功能，位于 `framework/builtin/memdomain/tlb` 路径下。该模块基于 Rocket-chip 的 TLB 实现，提供了 BuckyBall 特定的 TLB 封装和集群管理功能。

## 文件结构

```
tlb/
├── BBTLB.scala           - BuckyBall TLB 实现
├── TLBCluster.scala      - TLB 集群管理器
├── spec-BBTLB.md         - BBTLB 规范文档
└── spec-BBTLBCluster.md  - TLB 集群规范文档
```

## 核心组件

### BBTLB - BuckyBall TLB

BBTLB 是对 Rocket-chip TLB 的封装，提供 BuckyBall 特定的接口和异常处理：

```scala
class BBTLB(entries: Int, maxSize: Int)(implicit edge: TLEdgeOut, p: Parameters)
  extends CoreModule {

  val lgMaxSize = log2Ceil(maxSize)
  val io = IO(new Bundle {
    val req = Flipped(Valid(new BBTLBReq(lgMaxSize)))
    val resp = new TLBResp
    val ptw = new TLBPTWIO
    val exp = new BBTLBExceptionIO
  })
}
```

#### 请求接口

```scala
class BBTLBReq(val lgMaxSize: Int)(implicit p: Parameters) extends CoreBundle {
  val tlb_req = new TLBReq(lgMaxSize)    // TLB 请求
  val status = new MStatus               // 处理器状态
}
```

#### 异常处理接口

```scala
class BBTLBExceptionIO extends Bundle {
  val interrupt = Output(Bool())         // 中断输出
  val flush_retry = Input(Bool())        // 重试刷新
  val flush_skip = Input(Bool())         // 跳过刷新

  def flush(dummy: Int = 0): Bool = flush_retry || flush_skip
}
```

#### 核心实现

BBTLB 内部实例化 Rocket-chip 的 TLB：

```scala
val tlb = Module(new TLB(false, lgMaxSize, TLBConfig(nSets=1, nWays=entries)))
tlb.io.req.valid := io.req.valid
tlb.io.req.bits := io.req.bits.tlb_req
io.resp := tlb.io.resp
tlb.io.kill := false.B
```

#### 异常检测

```scala
val exception = io.req.valid && Mux(
  io.req.bits.tlb_req.cmd === M_XRD,
  tlb.io.resp.pf.ld || tlb.io.resp.ae.ld,
  tlb.io.resp.pf.st || tlb.io.resp.ae.st
)
when (exception) { interrupt := true.B }
```

### BBTLBCluster - TLB 集群

BBTLBCluster 管理多个 TLB 实例，为多个客户端提供地址转换服务：

```scala
class BBTLBCluster(nClients: Int, entries: Int, maxSize: Int)
                 (implicit edge: TLEdgeOut, p: Parameters) extends CoreModule {

  val io = IO(new Bundle {
    val clients = Flipped(Vec(nClients, new BBTLBIO))
    val ptw = Vec(nClients, new TLBPTWIO)
    val exp = Vec(nClients, new BBTLBExceptionIO)
  })
}
```

#### 客户端接口

```scala
class BBTLBIO(implicit p: Parameters) extends CoreBundle {
  val lgMaxSize = log2Ceil(coreDataBytes)
  val req = Valid(new BBTLBReq(lgMaxSize))    // TLB 请求
  val resp = Flipped(new TLBResp)             // TLB 响应
}
```

#### L0 TLB 缓存

每个客户端都有一个 L0 TLB 缓存，用于缓存最近的地址转换：

```scala
val last_translated_valid = RegInit(false.B)
val last_translated_vpn = RegInit(0.U(vaddrBits.W))
val last_translated_ppn = RegInit(0.U(paddrBits.W))

val l0_tlb_hit = last_translated_valid &&
  ((client.req.bits.tlb_req.vaddr >> pgIdxBits).asUInt ===
   (last_translated_vpn >> pgIdxBits).asUInt)
```

#### 地址转换流程

1. **L0 缓存检查**：首先检查 L0 TLB 缓存是否命中
2. **L1 TLB 查询**：L0 缓存未命中时查询 L1 TLB
3. **页表遍历**：TLB 未命中时通过 PTW 进行页表遍历
4. **结果缓存**：成功转换后更新 L0 缓存

```scala
when (tlbReqFire && !tlb.io.resp.miss) {
  last_translated_valid := true.B
  last_translated_vpn := tlbReq.tlb_req.vaddr
  last_translated_ppn := tlb.io.resp.paddr
}
```

## 配置参数

### TLB 配置

- `entries`: TLB 条目数量
- `maxSize`: 最大传输大小
- `nClients`: 客户端数量

### TLB 配置示例

```scala
val tlbConfig = TLBConfig(
  nSets = 1,           // TLB 组数
  nWays = 32           // 每组的路数
)
```

## 使用方法

### 创建单个 TLB

```scala
val bbtlb = Module(new BBTLB(entries = 32, maxSize = 64))

// 连接请求
bbtlb.io.req.valid := tlbReqValid
bbtlb.io.req.bits.tlb_req := tlbRequest
bbtlb.io.req.bits.status := processorStatus

// 获取响应
val tlbResp = bbtlb.io.resp
val physicalAddr = tlbResp.paddr
val tlbMiss = tlbResp.miss

// 连接页表遍历器
ptw <> bbtlb.io.ptw

// 处理异常
when(bbtlb.io.exp.interrupt) {
  // 处理 TLB 异常
}
```

### 创建 TLB 集群

```scala
val tlbCluster = Module(new BBTLBCluster(
  nClients = 4,
  entries = 32,
  maxSize = 64
))

// 连接客户端
for (i <- 0 until nClients) {
  tlbCluster.io.clients(i).req.valid := clientReqValid(i)
  tlbCluster.io.clients(i).req.bits := clientReq(i)
  clientResp(i) := tlbCluster.io.clients(i).resp
}

// 连接页表遍历器
for (i <- 0 until nClients) {
  ptw(i) <> tlbCluster.io.ptw(i)
}
```

## 地址转换过程

### 虚拟地址格式

```
Virtual Address (64-bit):
[63:39] [38:30] [29:21] [20:12] [11:0]
  VPN3    VPN2    VPN1    VPN0   Offset
```

### 物理地址格式

```
Physical Address:
[PPN][Offset]
```

### 转换流程

1. **地址解析**：解析虚拟地址的 VPN 和偏移量
2. **TLB 查找**：在 TLB 中查找 VPN 对应的 PPN
3. **页表遍历**：TLB 未命中时通过 PTW 遍历页表
4. **权限检查**：检查访问权限和保护位
5. **地址合成**：将 PPN 和偏移量合成物理地址

## 异常处理

### TLB 异常类型

- **页错误 (Page Fault)**：访问无效页面
- **访问异常 (Access Exception)**：权限不足
- **TLB 未命中 (TLB Miss)**：需要页表遍历

### 异常处理流程

```scala
val exception = io.req.valid && Mux(
  io.req.bits.tlb_req.cmd === M_XRD,
  tlb.io.resp.pf.ld || tlb.io.resp.ae.ld,    // 读异常
  tlb.io.resp.pf.st || tlb.io.resp.ae.st     // 写异常
)
```

### TLB 刷新

```scala
tlb.io.sfence.valid := io.exp.flush()
tlb.io.sfence.bits.rs1 := false.B
tlb.io.sfence.bits.rs2 := false.B
```

## 性能优化

### L0 TLB 缓存

- 减少 L1 TLB 访问延迟
- 提高地址转换吞吐量
- 降低功耗消耗

### 并行处理

- 多个客户端并行访问
- 独立的页表遍历器
- 分离的异常处理

## 调试和监控

### 性能计数器

- TLB 命中率统计
- 页表遍历次数
- 异常发生频率

### 调试接口

- TLB 条目状态查看
- 地址转换跟踪
- 异常信息记录

## 相关模块

- [内存域概览](../README.md) - 上层内存管理
- [DMA 引擎](../dma/README.md) - DMA 地址转换
- [内存控制器](../mem/README.md) - 内存访问接口
