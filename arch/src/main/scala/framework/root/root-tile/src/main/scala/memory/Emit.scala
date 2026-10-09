package hier.tile.memory

import java.nio.file.{Files, Paths}
import scala.sys.process._
import framework.system.core.rocket.CpuParams
import freechips.rocketchip.rocket.{DCacheParams, ICacheParams, RocketCoreParams}
import memcore.bus.chi.rnf.RnfParams
import memcore.bus.chi._
import memcore.bus.chi.snf.{LineRequest, LineResponse}
import memcore.memory.cpu.PhysicalRegion
import memcore.memory.cache.configs.CacheParams
import memcore.memory.coherence.configs.CoherenceParams

object Emit extends App {
  val root        = Paths.get("src/main/scala/framework/root/root-tile").toAbsolutePath
  val output      = root.resolve("build")
  Files.createDirectories(output)
  // Frozen verification profile: two identical 2-bank/8-line RN-Fs, one Home64.
  val chi         = Params()
  val p           = CoherenceParams(chi, CacheParams(chi.addressBits, 64, 4, 2, 8, 4, 2), 2, 4, 64)
  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new CacheSystem(p),
    firtoolOpts = args,
    args = Array("--target-dir", output.resolve("CacheSystem").toString, "--split-verilog")
  )
  val definitions = scala.collection.mutable.ArrayBuffer[String]()
  val ports       = scala.collection.mutable.ArrayBuffer[String]()
  def fields(flit: Flit): Seq[(String, Int)] =
    flit.fieldsLSB.map(field => flit.elements.find(_._2 eq field).get._1 -> field.getWidth)

  def layout(
    namespace:   String,
    definitions: scala.collection.mutable.Buffer[String],
    kind:        String,
    fs:          Seq[(String, Int)]
  ): Unit = {
    var offset = 0
    for ((name, width) <- fs if width > 0) {
      definitions += s"`define ${namespace}_${kind}_${name.toUpperCase}_OFFSET $offset"
      definitions += s"`define ${namespace}_${kind}_${name.toUpperCase}_WIDTH $width"
      offset += width
    }
    definitions += s"`define ${namespace}_${kind}_WIDTH $offset"
  }

  def bind(
    ports:  scala.collection.mutable.Buffer[String],
    prefix: String,
    signal: String,
    fs:     Seq[(String, Int)]
  ): Unit = {
    var offset = 0
    for ((name, width) <- fs if width > 0) {
      ports += s".$prefix$name($signal[$offset +: $width])"
      offset += width
    }
  }

  val req      = fields(new RequestFlit(chi))
  val rsp      = fields(new ResponseFlit(chi))
  val dat      = fields(new DataFlit(chi))
  val snp      = fields(new SnoopFlit(chi))
  val lineReq  = new LineRequest(chi)
  val lineResp = new LineResponse(chi)
  val memReq   = Seq("id", "addr", "write", "data", "mask").map(name => name -> lineReq.elements(name).getWidth)
  val memResp  = Seq("id", "data", "error").map(name => name -> lineResp.elements(name).getWidth)
  for (
    (kind, fs) <- Seq("REQ" -> req, "RSP" -> rsp, "DAT" -> dat, "SNP" -> snp, "MEMREQ" -> memReq, "MEMRESP" -> memResp)
  ) { layout("COH", definitions, kind, fs) }
  val snpWidth = snp.map(_._2).sum
  definitions += s"`define COH_SNP_TARGET_OFFSET $snpWidth"
  definitions += s"`define COH_SNP_TARGET_WIDTH ${chi.nodeIdBits}"
  definitions += s"`define COH_SNP_CHANNEL_WIDTH ${snpWidth + chi.nodeIdBits}"
  for (
    (name, value) <- Seq(
                       "AGENTS"           -> p.agents,
                       "MSHRS"            -> p.mshrEntries,
                       "HOME"             -> p.homeId,
                       "SETS"             -> p.cache.sets,
                       "WAYS"             -> p.cache.ways,
                       "DATA_BITS"        -> chi.dataBits,
                       "OUTSTANDING_BITS" -> chisel3.util.log2Ceil(p.mshrEntries + 1),
                       "READ_NSD"         -> Opcode.ReadNotSharedDirty,
                       "READ_UNIQUE"      -> Opcode.ReadUnique,
                       "WRITEBACK"        -> Opcode.WriteBackFull,
                       "COMP"             -> Opcode.Comp,
                       "COMP_DBID"        -> Opcode.CompDBIDResp,
                       "COMP_ACK"         -> Opcode.CompAck,
                       "COMP_DATA"        -> Opcode.CompData,
                       "SNP_RESP"         -> Opcode.SnpResp,
                       "SNP_DATA"         -> Opcode.SnpRespData,
                       "COPYBACK_DATA"    -> Opcode.CopyBackWriteData
                     )
  ) {
    definitions += s"`define COH_$name $value"
  }

  val observedChannels = Seq(
    ("observedReq", "req", req),
    ("observedRsp", "rsp", rsp),
    ("observedDat", "dat", dat),
    ("observedRxRsp", "rx_rsp", rsp),
    ("observedRxDat", "rx_dat", dat)
  )

  for (core               <- 0 until p.agents) {
    for (field <- Seq("addr", "write", "data", "mask", "atomic", "atomicWord")) {
      ports += s".io_access_${core}_bits_$field(control.access_$field[$core])"
    }
    ports += s".io_access_${core}_valid(control.access_valid[$core])"
    ports += s".io_access_${core}_ready(control.access_ready[$core])"
    ports += s".io_result_${core}_valid(control.result_valid[$core])"
    ports += s".io_result_${core}_ready(control.result_ready[$core])"
    ports += s".io_result_${core}_bits_data(control.result_data[$core])"
    ports += s".io_result_${core}_bits_error(control.result_error[$core])"
  }
  for ((port, signal, fs) <- observedChannels) {
    bind(ports, s"io_${port}_bits_", s"control.${signal}_bits", fs)
    ports += s".io_${port}_valid(control.${signal}_valid)"
  }
  bind(ports, "io_observedSnp_bits_flit_", "control.snp_bits", snp)
  ports += s".io_observedSnp_bits_targetNode(control.snp_bits[$snpWidth +: ${chi.nodeIdBits}])"
  ports += ".io_observedSnp_valid(control.snp_valid)"
  bind(ports, "io_memoryReq_bits_", "mem_req.bits", memReq)
  bind(ports, "io_memoryResp_bits_", "mem_resp.bits", memResp)
  ports ++= Seq(
    ".io_memoryReq_valid(mem_req.valid)",
    ".io_memoryReq_ready(mem_req.ready)",
    ".io_memoryResp_valid(mem_resp.valid)",
    ".io_memoryResp_ready(mem_resp.ready)"
  )
  Files.writeString(
    output.resolve("cache_system_config.svh"),
    "`ifndef CACHE_SYSTEM_CONFIG_SVH\n`define CACHE_SYSTEM_CONFIG_SVH\n" + definitions.mkString("\n") + "\n`endif\n"
  )
  Files.writeString(output.resolve("cache_system_ports.svh"), ports.mkString(",\n") + "\n")

  // The short uncached window makes aligned cross-region accesses testable.
  val regions = Seq(
    PhysicalRegion(BigInt("80000000", 16), 0x1000000, true, true, true, true, true, true),
    PhysicalRegion(BigInt("10000000", 16), 0x1000, false, false, true, true, false, false),
    PhysicalRegion(BigInt("10002000", 16), 5, false, false, true, true, false, false),
    PhysicalRegion(BigInt("90000000", 16), 64, true, false, true, false, false, true),
    PhysicalRegion(BigInt("90000040", 16), 64, true, false, false, true, false, true),
    PhysicalRegion(BigInt("a0000000", 16), 0x100000, false, true, true, true, true, true)
  )

  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new VirtualCacheSystem(p, regions),
    firtoolOpts = args,
    args = Array("--target-dir", output.resolve("VirtualCacheSystem").toString, "--split-verilog")
  )
  val vd = scala.collection.mutable.ArrayBuffer[String]()
  val vp = scala.collection.mutable.ArrayBuffer[String]()

  def virtualFields(
    kind:   String,
    prefix: String,
    signal: String,
    fs:     Seq[(String, Int)]
  ): Unit = {
    layout("VM", vd, kind, fs)
    bind(vp, prefix, signal, fs)
  }

  virtualFields(
    "REQ",
    "io_request_bits_",
    "source_if.bits",
    Seq(
      "vaddr"     -> 64,
      "tag"       -> 6,
      "size"      -> 3,
      "write"     -> 1,
      "execute"   -> 1,
      "signed"    -> 1,
      "data"      -> 64,
      "atomic"    -> 4,
      "privilege" -> 2,
      "sum"       -> 1,
      "mxr"       -> 1,
      "satpMode"  -> 4,
      "rootPpn"   -> 44
    )
  )
  virtualFields(
    "AUTH",
    "io_authorizationRequest_bits_",
    "auth_req.bits",
    Seq("paddr" -> 64, "size" -> 3, "read" -> 1, "write" -> 1, "execute" -> 1, "isPte" -> 1, "privilege" -> 2)
  )
  vp += ".io_authorizationResponse_bits(auth_resp.bits)"
  virtualFields(
    "RESP",
    "io_response_bits_",
    "sink_if.bits",
    Seq("tag" -> 6, "data" -> 64, "misaligned" -> 1, "pageFault" -> 1, "accessFault" -> 1)
  )
  virtualFields("MEMREQ", "io_memoryReq_bits_", "mem_req.bits", memReq)
  virtualFields("MEMRESP", "io_memoryResp_bits_", "mem_resp.bits", memResp)
  virtualFields(
    "UNCACHED",
    "io_uncachedRequest_bits_",
    "uncached_req.bits",
    Seq("addr" -> 64, "tag" -> 6, "size" -> 3, "write" -> 1, "data" -> 64, "atomic" -> 4, "normal" -> 1)
  )
  virtualFields("URESP", "io_uncachedResponse_bits_", "uncached_resp.bits", Seq("tag" -> 6, "data" -> 64, "error" -> 1))
  virtualFields(
    "TRANSLATION",
    "io_observedTranslation_bits_",
    "control.translation_bits",
    Seq(
      "paddr"       -> chi.addressBits,
      "pageFault"   -> 1,
      "accessFault" -> 1,
      "level"       -> 2
    )
  )
  virtualFields(
    "PHYSICAL",
    "io_observedPhysical_bits_",
    "control.physical_bits",
    Seq(
      "addr"      -> 64,
      "tag"       -> 6,
      "size"      -> 3,
      "write"     -> 1,
      "signed"    -> 1,
      "data"      -> 64,
      "atomic"    -> 4,
      "cacheable" -> 1,
      "normal"    -> 1
    )
  )
  virtualFields(
    "CACHE",
    "io_observedCache_bits_access_",
    "control.cache_bits",
    Seq("addr" -> chi.addressBits, "write" -> 1, "data" -> 64, "mask" -> 8, "atomic" -> 4, "atomicWord" -> 1)
  )
  vp ++= Seq(
    ".io_observedTranslation_valid(control.translation_valid)",
    ".io_observedPhysical_valid(control.physical_valid)",
    ".io_observedCache_valid(control.cache_valid)",
    ".io_observedCache_bits_pte(control.cache_pte)",
    ".io_observedEviction_valid(control.eviction_valid)",
    ".io_observedEviction_bits(control.eviction_addr)"
  )
  for (
    (port, signal) <- Seq(
                        "request"               -> "source_if",
                        "response"              -> "sink_if",
                        "memoryReq"             -> "mem_req",
                        "memoryResp"            -> "mem_resp",
                        "authorizationRequest"  -> "auth_req",
                        "authorizationResponse" -> "auth_resp",
                        "uncachedRequest"       -> "uncached_req",
                        "uncachedResponse"      -> "uncached_resp"
                      )
  ) {
    vp += s".io_${port}_valid($signal.valid)"
    vp += s".io_${port}_ready($signal.ready)"
  }
  for (
    (name, region) <- Seq(
                        "DDR"        -> regions(0),
                        "MMIO"       -> regions(1),
                        "SHORT_MMIO" -> regions(2),
                        "READ_ONLY"  -> regions(3),
                        "WRITE_ONLY" -> regions(4),
                        "NORMAL_RAM" -> regions(5)
                      )
  ) {
    vd += s"`define VM_${name}_BASE 64'h${region.base.toString(16)}"
    vd += s"`define VM_${name}_BYTES 64'h${region.bytes.toString(16)}"
  }
  vd += s"`define VM_OUTSTANDING_BITS ${chisel3.util.log2Ceil(p.mshrEntries + 1)}"
  Files.writeString(
    output.resolve("virtual_cache_config.svh"),
    "`ifndef VIRTUAL_CACHE_CONFIG_SVH\n`define VIRTUAL_CACHE_CONFIG_SVH\n" + vd.mkString("\n") + "\n`endif\n"
  )
  Files.writeString(output.resolve("virtual_cache_ports.svh"), vp.mkString(",\n") + "\n")

  implicit val cpu: CpuParams = CpuParams(
    core = new RocketCoreParams(
      useVM = false,
      useUser = false,
      useSupervisor = false,
      useHypervisor = false,
      useDebug = false,
      fpu = None,
      nPMPs = 4,
      clockGate = false,
      haveCFlush = false,
      haveCease = false,
      haveSimTimeout = false
    ) { override val pmpGranularity = 8 },
    dcache = Some(DCacheParams(nSets = 64, nWays = 2)),
    icache = Some(ICacheParams()),
    btb = None,
    physicalAddressBits = 44,
    hartIdBits = 1,
    beatBytes = 8,
    blockBytes = 64
  )

  val builder = Paths.get("src/main/scala/framework/root/root-core/src/main/resources/build_fixture.py").toAbsolutePath
  require(
    Process(Seq("python", builder.toString, output.toString, "multicore")).! == 0,
    "Two-Core fixture generation failed"
  )
  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new Tile(p.copy(agents = 2 * p.agents), RnfParams(chi, cacheLines = 8, banks = 2), regions),
    firtoolOpts = args,
    args = Array("--target-dir", output.resolve("Tile").toString, "--split-verilog")
  )

  val td = scala.collection.mutable.ArrayBuffer[String](
    s"""`define TILE_IMAGE "${output.resolve("multicore.hex")}""""
  )

  val tp = scala.collection.mutable.ArrayBuffer[String]()

  bind(tp, "io_memoryReq_bits_", "mem_req.bits", memReq)
  bind(tp, "io_memoryResp_bits_", "mem_resp.bits", memResp)
  tp ++= Seq(
    ".io_memoryReq_valid(mem_req.valid)",
    ".io_memoryReq_ready(mem_req.ready)",
    ".io_memoryResp_valid(mem_resp.valid)",
    ".io_memoryResp_ready(mem_resp.ready)"
  )
  val uncached     = Seq("addr" -> 64, "tag" -> 6, "size" -> 3, "write" -> 1, "data" -> 64)
  val uncachedResp = Seq("tag" -> 6, "data" -> 64, "error" -> 1)
  for ((kind, fs)         <- Seq("MEMREQ" -> memReq, "MEMRESP" -> memResp, "UNCACHED" -> uncached, "URESP" -> uncachedResp)) {
    layout("TILE", td, kind, fs)
  }
  for (i                  <- 0 until p.agents) {
    bind(tp, s"io_uncachedRequest_${i}_bits_", s"uncached_req_$i.bits", uncached)
    bind(tp, s"io_uncachedResponse_${i}_bits_", s"uncached_resp_$i.bits", uncachedResp)
    tp ++= Seq(
      s".io_uncachedRequest_${i}_valid(uncached_req_$i.valid)",
      s".io_uncachedRequest_${i}_ready(uncached_req_$i.ready)",
      s".io_uncachedResponse_${i}_valid(uncached_resp_$i.valid)",
      s".io_uncachedResponse_${i}_ready(uncached_resp_$i.ready)"
    )
    for (field <- Seq("retired", "retiredPc", "trapped", "trapCause", "trapValue", "trapPc"))
      tp += s".io_${field}_$i(control.$field[$i])"
  }
  for ((port, signal, fs) <- observedChannels) {
    bind(tp, s"io_${port}_bits_", s"control.${signal}_bits", fs)
    tp += s".io_${port}_valid(control.${signal}_valid)"
  }
  bind(tp, "io_observedSnp_bits_flit_", "control.snp_bits", snp)
  tp += s".io_observedSnp_bits_targetNode(control.snp_bits[$snpWidth +: ${chi.nodeIdBits}])"
  tp += ".io_observedSnp_valid(control.snp_valid)"
  Files.writeString(output.resolve("tile_config.svh"), td.mkString("\n") + "\n")
  Files.writeString(output.resolve("tile_ports.svh"), tp.mkString(",\n") + "\n")

  val tracking          = memcore.memory.interlock.Params()
  val consistencyMemory = p.copy(cache = p.cache.copy(sets = 16))
  val consistencyL1     = RnfParams(chi, cacheLines = 8, banks = 2)
  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new Consistency(consistencyMemory, consistencyL1, tracking),
    firtoolOpts = args,
    args = Array("--target-dir", output.resolve("Consistency").toString, "--split-verilog")
  )

  val cd = scala.collection.mutable.ArrayBuffer[String](
    s"`define CONS_AGENTS ${p.agents}",
    s"`define CONS_HOME ${p.homeId}",
    s"`define CONS_MSHRS ${p.mshrEntries}",
    s"`define CONS_OUTSTANDING_BITS ${chisel3.util.log2Ceil(p.mshrEntries + 1)}",
    s"`define CONS_PA_BITS ${tracking.addressBits}",
    s"`define CONS_LINE_BYTES ${tracking.lineBytes}",
    s"`define CONS_L1_LINES ${consistencyL1.cacheLines}",
    s"`define CONS_L2_SETS ${consistencyMemory.cache.sets}",
    s"`define CONS_SLOTS ${tracking.entries}"
  )

  val cp = scala.collection.mutable.ArrayBuffer[String]()
  def members(bundle: chisel3.Bundle): Seq[(String, Int)] =
    bundle.elements.toSeq.collect { case (name, field) if field.getWidth > 0 => name -> field.getWidth }
  val accessFields = members(new memcore.bus.chi.rnf.CacheAccess(chi))
  val resultFields = members(new memcore.bus.chi.rnf.CacheResult)
  layout("CONS", cd, "CPU", accessFields)
  layout("CONS", cd, "RESULT", resultFields)

  def connect(port: String, signal: String, fs: Seq[(String, Int)]): Unit = {
    bind(cp, s"io_${port}_bits_", s"$signal.bits", fs)
    cp += s".io_${port}_valid($signal.valid)"
    cp += s".io_${port}_ready($signal.ready)"
  }

  for (i                     <- 0 until p.agents) {
    connect(s"access_$i", s"access_$i", accessFields)
    connect(s"result_$i", s"result_$i", resultFields)
  }
  for (
    (kind, port, signal, fs) <- Seq(
                                  (
                                    "DISPATCH",
                                    "dispatch",
                                    "dispatch",
                                    members(new memcore.memory.interlock.Dispatch(tracking))
                                  ),
                                  (
                                    "INFO",
                                    "accessInfo",
                                    "access_info",
                                    members(new memcore.memory.interlock.AccessInfo(tracking))
                                  ),
                                  ("GRANT", "grant", "grant", members(new memcore.memory.interlock.Tag(tracking))),
                                  ("DONE", "done", "done", members(new memcore.memory.interlock.Acknowledgement(tracking))),
                                  ("COMPLETE", "complete", "complete", members(new memcore.memory.interlock.Tag(tracking))),
                                  ("MEMREQ", "memoryReq", "mem_req", memReq),
                                  ("MEMRESP", "memoryResp", "mem_resp", memResp)
                                )
  ) {
    layout("CONS", cd, kind, fs)
    connect(port, signal, fs)
  }
  layout("CONS", cd, "REQ", req)
  layout("CONS", cd, "SNP", snp)
  cd += s"`define CONS_SNP_TARGET_OFFSET $snpWidth"
  cd += s"`define CONS_SNP_TARGET_WIDTH ${chi.nodeIdBits}"
  cd += s"`define CONS_SNP_CHANNEL_WIDTH ${snpWidth + chi.nodeIdBits}"
  bind(cp, "io_observedReq_bits_", "control.req_bits", req)
  bind(cp, "io_observedSnp_bits_flit_", "control.snp_bits", snp)
  cp ++= Seq(
    ".io_observedReq_valid(control.req_valid)",
    ".io_observedSnp_valid(control.snp_valid)",
    s".io_observedSnp_bits_targetNode(control.snp_bits[$snpWidth +: ${chi.nodeIdBits}])",
    ".io_olderDispatchPending(control.older_dispatch_pending)",
    ".io_olderRequestsDrained(control.older_requests_drained)",
    ".io_blockRequesterRsp(control.block_requester_rsp)",
    ".io_blockRequesterData(control.block_requester_data)",
    ".io_cpuAllow(control.cpu_allow)",
    ".io_outstanding(control.outstanding)"
  )
  Files.writeString(output.resolve("consistency_config.svh"), cd.mkString("\n") + "\n")
  Files.writeString(output.resolve("consistency_ports.svh"), cp.mkString(",\n") + "\n")
  // Five-core Goban shape elaboration only: explicit placement and opaque test signatures.
  // This does not load a PB profile; the production Tile is assembled in framework.system.tile.
  val compositionCpu    = cpu.copy(hartIdBits = 3)
  val compositionMemory = p.copy(agents = 10)

  val compositionPlacements = Seq(
    CorePlacement(RnfParams(chi, nodeId = 1, cacheLines = 8, banks = 2), compositionCpu, CoreRole.Controller),
    CorePlacement(RnfParams(chi, nodeId = 2, cacheLines = 8, banks = 2), compositionCpu, CoreRole.Compute),
    CorePlacement(RnfParams(chi, nodeId = 3, cacheLines = 8, banks = 2), compositionCpu, CoreRole.Compute),
    CorePlacement(RnfParams(chi, nodeId = 4, cacheLines = 8, banks = 2), compositionCpu, CoreRole.Compute),
    CorePlacement(RnfParams(chi, nodeId = 5, cacheLines = 8, banks = 2), compositionCpu, CoreRole.Compute)
  )

  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new Composition(
      compositionMemory,
      compositionPlacements,
      regions,
      memcore.memory.interlock.Params(addressBits = chi.addressBits),
      workerCoreIds = Seq(1, 2, 3, 4),
      signatures = Seq(BigInt(1), BigInt(2), BigInt(3), BigInt(4))
    ),
    firtoolOpts = args,
    args = Array("--target-dir", output.resolve("Composition").toString, "--split-verilog")
  )

}
