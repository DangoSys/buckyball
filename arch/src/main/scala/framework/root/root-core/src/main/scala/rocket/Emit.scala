package hier.core.rocket

import java.nio.file.{Files, Paths}
import scala.sys.process._
import framework.system.core.rocket.{CpuParams, RocketParameters}
import freechips.rocketchip.tile.FPUParams
import freechips.rocketchip.rocket.{
  ASIdBits,
  DCacheParams,
  HellaCacheReq,
  HellaCacheResp,
  ICacheParams,
  RocketCoreParams
}
import memcore.memory.cpu.{CpuMemParams, PhysicalRegion, VirtualMemoryRequest, VirtualMemoryResponse}
import chisel3._
import memcore.memory.cache.configs.CacheParams
import memcore.memory.coherence.configs.CoherenceParams
import memcore.bus.chi.{Params => ChiParams}

object Emit extends App {
  val root   = Paths.get("src/main/scala/framework/root/root-core").toAbsolutePath
  val output = root.resolve("build")
  Files.createDirectories(output)
  require(
    Process(Seq("python", root.resolve("src/main/resources/build_fixture.py").toString, output.toString)).! == 0,
    "Core fixture generation failed"
  )

  implicit val cpuParams: CpuParams = CpuParams(
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

  implicit val upstream = RocketParameters(cpuParams)

  val memory = CoherenceParams(ChiParams(), CacheParams(44, 64, 4, 2, 8, 4, 2), 2, 4, 64)

  val floatingParameters = cpuParams.copy(core = cpuParams.core.copy(fpu = Some(FPUParams(minFLen = 16, fLen = 64))))

  val regions = Seq(
    PhysicalRegion(BigInt("80000000", 16), BigInt(16) << 20, true, true, true, true, true, true),
    PhysicalRegion(BigInt("10000000", 16), BigInt(4096), false, false, true, true, false, false)
  )

  require(
    Process(Seq("python", root.resolve("src/main/resources/build_fixture.py").toString, output.toString, "fpu")).! == 0,
    "FPU fixture generation failed"
  )
  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new Verification(memory, regions, "Fpu")(floatingParameters),
    firtoolOpts = args,
    args = Array("--target-dir", output.resolve("Fpu").toString, "--split-verilog")
  )

  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new Verification(memory, regions, "Core"),
    firtoolOpts = args,
    args = Array("--target-dir", output.resolve("Core").toString, "--split-verilog")
  )

  val supervisorParameters = cpuParams.copy(
    core = new RocketCoreParams(
      useVM = true,
      useUser = true,
      useSupervisor = true,
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
    physicalAddressBits = 36,
    asIdBits = 16
  )

  require(
    Process(
      Seq("python", root.resolve("src/main/resources/build_fixture.py").toString, output.toString, "supervisor")
    ).! == 0,
    "Supervisor fixture generation failed"
  )
  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new Verification(memory, regions, "Supervisor")(supervisorParameters),
    firtoolOpts = args,
    args = Array("--target-dir", output.resolve("Supervisor").toString, "--split-verilog")
  )

  require(
    Process(Seq("python", root.resolve("src/main/resources/build_fixture.py").toString, output.toString, "irq")).! == 0,
    "IRQ fixture generation failed"
  )
  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new Verification(memory, regions, "Irq"),
    firtoolOpts = args,
    args = Array("--target-dir", output.resolve("Irq").toString, "--split-verilog")
  )

  val definitions = scala.collection.mutable.ArrayBuffer[String](
    s"""`define IRQ_IMAGE "${output.resolve("irq.hex")}"""",
    s"""`define FPU_IMAGE "${output.resolve("fpu.hex")}"""",
    s"""`define SUPERVISOR_IMAGE "${output.resolve("supervisor.hex")}"""",
    s"""`define CORE_IMAGE "${output.resolve("core.hex")}""""
  )

  val ports = scala.collection.mutable.ArrayBuffer[String]()

  def channel(
    kind:     String,
    prefix:   String,
    instance: String,
    fields:   Seq[(String, Int)]
  ): Unit = {
    var offset = 0
    for ((name, width) <- fields) {
      definitions += s"`define CORE_${kind}_${name.toUpperCase}_OFFSET $offset"
      definitions += s"`define CORE_${kind}_${name.toUpperCase}_WIDTH $width"
      ports += s".$prefix$name($instance.bits[$offset +: $width])"
      offset += width
    }
    definitions += s"`define CORE_${kind}_WIDTH $offset"
  }

  channel(
    "MEMREQ",
    "io_memoryRequest_bits_",
    "mem_req",
    Seq("id" -> 12, "addr" -> 44, "write" -> 1, "data" -> 512, "mask" -> 64)
  )
  channel("MEMRESP", "io_memoryResponse_bits_", "mem_resp", Seq("id" -> 12, "data" -> 512, "error" -> 1))
  channel(
    "UNCACHED",
    "io_uncachedRequest_bits_",
    "uncached_req",
    Seq("addr" -> 64, "tag" -> 6, "size" -> 3, "write" -> 1, "data" -> 64)
  )
  channel("URESP", "io_uncachedResponse_bits_", "uncached_resp", Seq("tag" -> 6, "data" -> 64, "error" -> 1))
  for (
    (prefix, instance) <- Seq(
                            "memoryRequest"    -> "mem_req",
                            "memoryResponse"   -> "mem_resp",
                            "uncachedRequest"  -> "uncached_req",
                            "uncachedResponse" -> "uncached_resp"
                          )
  ) {
    ports += s".io_${prefix}_valid($instance.valid)"
    ports += s".io_${prefix}_ready($instance.ready)"
  }
  Files.write(output.resolve("core_config.svh"), (definitions.mkString("\n") + "\n").getBytes)
  Files.write(output.resolve("core_ports.svh"), ports.mkString(",\n").getBytes)
  val cp = CpuMemParams(memory.chi, tagBits = 6)
  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new Lsu(cp),
    firtoolOpts = args,
    args = Array("--target-dir", output.resolve("Lsu").toString, "--split-verilog")
  )
  definitions.clear()
  ports.clear()

  def elements(bundle: Bundle): Seq[(String, Int)] = bundle.elements.toSeq.collect {
    case (name, field) if field.getWidth > 0 => name -> field.getWidth
  }

  channel("CPU", "io_cpu_req_bits_", "source_if", elements(new HellaCacheReq))
  channel("RETURN", "io_cpu_resp_bits_", "sink_if", elements(new HellaCacheResp))
  channel("VIRTUAL", "io_request_bits_", "virtual_req", elements(new VirtualMemoryRequest(cp)))
  channel("RESULT", "io_response_bits_", "virtual_resp", elements(new VirtualMemoryResponse(cp)))
  for ((prefix, instance) <- Seq("cpu_req" -> "source_if", "request" -> "virtual_req", "response" -> "virtual_resp")) {
    ports += s".io_${prefix}_valid($instance.valid)"
    ports += s".io_${prefix}_ready($instance.ready)"
  }
  ports += ".io_cpu_resp_valid(sink_if.valid)"
  for (i                  <- 0 until 4) {
    for (name <- Seq("l", "res", "a", "x", "w", "r")) ports += s".io_pmp_${i}_cfg_$name('0)"
    ports += s".io_pmp_${i}_addr('0)"
    ports += s".io_pmp_${i}_mask('0)"
  }
  ports += ".io_context_privilege(2'd3)"
  ports += ".io_context_satp(64'd0)"
  ports += ".io_context_sum(1'b0)"
  ports += ".io_context_mxr(1'b0)"
  ports += ".io_maintenance_ready(1'b0)"
  ports += ".io_maintained_valid(1'b0)"
  ports += ".io_maintained_bits(1'b0)"
  Files.write(output.resolve("lsu_config.svh"), (definitions.mkString("\n") + "\n").getBytes)
  Files.write(output.resolve("lsu_ports.svh"), ports.mkString(",\n").getBytes)

  val tracking = memcore.memory.interlock.Params()
  val rnf      = memcore.bus.chi.rnf.RnfParams(memory.chi, homeCount = 2)
  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new Maintenance(rnf, tracking),
    firtoolOpts = args,
    args = Array("--target-dir", output.resolve("Maintenance").toString, "--split-verilog")
  )
  definitions.clear()
  ports.clear()
  definitions += s"`define MAINTENANCE_HOME_BASE ${rnf.homeId}"
  definitions += s"`define MAINTENANCE_HOME_COUNT ${rnf.homeCount}"
  definitions += s"`define MAINTENANCE_NODE ${rnf.nodeId}"
  definitions += s"`define MAINTENANCE_TXN ${rnf.banks}"
  channel("RANGE", "io_request_bits_", "source_if", elements(new memcore.memory.interlock.Maintenance(tracking)))
  channel("ACK", "io_response_bits_", "sink_if", elements(new memcore.memory.interlock.Acknowledgement(tracking)))
  channel("REQ", "io_req_bits_", "chi_req", elements(new memcore.bus.chi.RequestFlit(memory.chi)))
  channel("RSP", "io_rsp_bits_", "chi_rsp", elements(new memcore.bus.chi.ResponseFlit(memory.chi)))
  for (
    (prefix, instance) <- Seq("request" -> "source_if", "response" -> "sink_if", "req" -> "chi_req", "rsp" -> "chi_rsp")
  ) {
    ports += s".io_${prefix}_valid($instance.valid)"
    ports += s".io_${prefix}_ready($instance.ready)"
  }
  Files.write(output.resolve("maintenance_config.svh"), (definitions.mkString("\n") + "\n").getBytes)
  Files.write(output.resolve("maintenance_ports.svh"), ports.mkString(",\n").getBytes)

  require(
    Process(
      Seq("python", root.resolve("src/main/resources/build_fixture.py").toString, output.toString, "admission")
    ).! == 0,
    "Admission fixture generation failed"
  )
  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new Verification(
      memory,
      regions :+ PhysicalRegion(BigInt("a0000000", 16), BigInt(4096), false, false, true, true, true, true),
      "AdmissionCore",
      Some(Commands(compute = true, scheduler = true, tracking = tracking))
    )(supervisorParameters),
    firtoolOpts = args,
    args = Array("--target-dir", output.resolve("AdmissionCore").toString, "--split-verilog")
  )
  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new memcore.memory.interlock.Interlock(tracking),
    firtoolOpts = args,
    args = Array("--target-dir", output.resolve("AdmissionInterlock").toString, "--split-verilog")
  )

  val admissionDefinitions = scala.collection.mutable.ArrayBuffer[String](
    s"""`define ADMIT_IMAGE "${output.resolve("admission.hex")}"""",
    s"`define ADMIT_ENTRIES ${tracking.entries}",
    s"`define ADMIT_PMP_COUNT ${supervisorParameters.core.nPMPs}",
    s"`define ADMIT_OUTSTANDING_BITS ${chisel3.util.log2Ceil(tracking.entries + 1)}",
    s"`define ADMIT_MEMORY_OUTSTANDING_BITS ${chisel3.util.log2Ceil(memory.mshrEntries + 1)}"
  )

  val admissionPorts = scala.collection.mutable.ArrayBuffer[String]()

  def flatten(data: Data, prefix: String = ""): Seq[(String, Int)] = data match {
    case record: Record => record.elements.toSeq.flatMap { case (name, value) =>
        flatten(value, if (prefix.isEmpty) name else s"${prefix}_$name")
      }
    case vector: Vec[_] => vector.zipWithIndex.flatMap { case (value, i) => flatten(value, s"${prefix}_$i") }
    case field if field.getWidth > 0 => Seq(prefix -> field.getWidth)
    case _                           => Seq.empty
  }

  def admissionLayout(
    kind:   String,
    prefix: String,
    signal: String,
    bundle: Data
  ): Unit = {
    var offset = 0
    for ((name, width) <- flatten(bundle)) {
      admissionDefinitions += s"`define ADMIT_${kind}_${name.toUpperCase}_OFFSET $offset"
      admissionDefinitions += s"`define ADMIT_${kind}_${name.toUpperCase}_WIDTH $width"
      admissionPorts += s".$prefix$name($signal[$offset +: $width])"
      offset += width
    }
    admissionDefinitions += s"`define ADMIT_${kind}_WIDTH $offset"
  }

  for (
    (kind, port, signal, bundle) <- Seq(
                                      (
                                        "RESERVE",
                                        "admission_reserve",
                                        "reserve",
                                        new memcore.memory.interlock.Dispatch(tracking)
                                      ),
                                      (
                                        "COMMAND",
                                        "admission_command",
                                        "command",
                                        new CommandSnapshot(tracking, supervisorParameters.core.nPMPs)(
                                          supervisorParameters
                                        )
                                      ),
                                      (
                                        "COMPLETE",
                                        "admission_complete",
                                        "complete",
                                        new memcore.memory.interlock.Tag(tracking)
                                      ),
                                      (
                                        "CANCELLED",
                                        "admission_cancelled",
                                        "cancelled",
                                        new memcore.memory.interlock.Tag(tracking)
                                      ),
                                      (
                                        "RESPONSE",
                                        "admission_response",
                                        "response",
                                        new framework.system.core.rocket.RoCCResponseBB
                                      ),
                                      (
                                        "MAINTENANCE",
                                        "admission_maintenance",
                                        "maintenance",
                                        new memcore.memory.interlock.Maintenance(tracking)
                                      ),
                                      (
                                        "MAINTAINED",
                                        "admission_maintained",
                                        "maintained",
                                        new memcore.memory.interlock.Acknowledgement(tracking)
                                      ),
                                      (
                                        "PTE",
                                        "admission_pteRequest",
                                        "pte_req",
                                        new memcore.bus.chi.rnf.CacheAccess(memory.chi)
                                      ),
                                      (
                                        "PTERESP",
                                        "admission_pteResponse",
                                        "pte_resp",
                                        new memcore.bus.chi.rnf.CacheResult
                                      ),
                                      (
                                        "MEMREQ",
                                        "memoryRequest",
                                        "mem_req",
                                        new memcore.bus.chi.snf.LineRequest(memory.chi)
                                      ),
                                      (
                                        "MEMRESP",
                                        "memoryResponse",
                                        "mem_resp",
                                        new memcore.bus.chi.snf.LineResponse(memory.chi)
                                      ),
                                      (
                                        "UNCACHED",
                                        "uncachedRequest",
                                        "uncached_req",
                                        new memcore.memory.cpu.UncachedRequest(cp)
                                      ),
                                      (
                                        "URESP",
                                        "uncachedResponse",
                                        "uncached_resp",
                                        new memcore.memory.cpu.UncachedResponse(cp)
                                      )
                                    )
  ) {
    admissionLayout(kind, s"io_${port}_bits_", s"$signal.bits", bundle)
    admissionPorts += s".io_${port}_valid($signal.valid)"
    admissionPorts += s".io_${port}_ready($signal.ready)"
  }
  admissionLayout("QUERY", "io_admission_cpuQuery_", "control.query", new memcore.memory.interlock.CpuQuery(tracking))
  admissionPorts ++= Seq(
    ".io_admission_cpuAllow(control.cpu_allow)",
    ".io_admission_cpuProbeAllow(control.cpu_allow)",
    ".io_admission_interrupt(control.accelerator_irq)",
    ".io_admission_outstanding(control.admission_outstanding)"
  )
  Files.write(output.resolve("admission_config.svh"), (admissionDefinitions.mkString("\n") + "\n").getBytes)
  Files.write(output.resolve("admission_ports.svh"), admissionPorts.mkString(",\n").getBytes)
  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new AdmissionBridge(16, tracking, supervisorParameters.core.nPMPs, Seq(1))(supervisorParameters),
    firtoolOpts = args,
    args = Array("--target-dir", output.resolve("AdmissionBridge").toString, "--split-verilog")
  )

  val bridgeDefinitions = scala.collection.mutable.ArrayBuffer[String](
    "`define AB_ROB_ENTRIES 16",
    "`define AB_ROB_BITS 4",
    s"`define AB_ENTRIES ${tracking.entries}"
  )

  val bridgePorts    = scala.collection.mutable.ArrayBuffer[String]()
  val bridgeSnapshot = new CommandSnapshot(tracking, supervisorParameters.core.nPMPs)(supervisorParameters)

  def bridgeLayout(
    kind:   String,
    prefix: String,
    signal: String,
    data:   Data,
    define: Boolean
  ): Unit = {
    var offset = 0
    for ((name, width) <- flatten(data)) {
      if (define) {
        bridgeDefinitions += s"`define AB_${kind}_${name.toUpperCase}_OFFSET $offset"
        bridgeDefinitions += s"`define AB_${kind}_${name.toUpperCase}_WIDTH $width"
      }
      bridgePorts += s".$prefix$name($signal[$offset +: $width])"
      offset += width
    }
    if (define) bridgeDefinitions += s"`define AB_${kind}_WIDTH $offset"
  }

  bridgeLayout("SNAP", "io_command_bits_", "command.bits", bridgeSnapshot, true)
  val bridgeRetirement =
    new AdmissionRetirement(tracking, supervisorParameters.core.nPMPs)(supervisorParameters)
  bridgeLayout("RET", "io_retirement_bits_", "retirement.bits", bridgeRetirement, true)
  bridgeLayout("FAULT", "io_fault_bits_", "control.fault_bits", new RobFault(16), true)
  bridgeLayout("FAULT", "io_unboundFault_bits_", "control.unbound_fault_bits", new RobFault(16), false)
  bridgePorts ++= Seq(".io_fault_valid(control.fault_valid)", ".io_unboundFault_valid(control.unbound_fault_valid)")
  val snapshotStart    =
    flatten(bridgeRetirement).takeWhile { case (name, _) => !name.startsWith("snapshot_") }.map(_._2).sum
  bridgeDefinitions += s"`define AB_RET_SNAPSHOT_OFFSET $snapshotStart"
  bridgeLayout("NPU", "io_npuCommand_bits_", "npu.bits", new framework.system.core.rocket.RoCCCommandBB, true)
  for ((port, signal) <- Seq("command" -> "command", "retirement" -> "retirement", "npuCommand" -> "npu")) {
    bridgePorts += s".io_${port}_valid($signal.valid)"
    bridgePorts += s".io_${port}_ready($signal.ready)"
  }
  for (i <- 0 until 3) {
    bridgeLayout("SNAP", s"io_lookup_${i}_snapshot_bits_", s"control.lookup_bits[$i]", bridgeSnapshot, false)
    bridgePorts += s".io_lookup_${i}_snapshot_valid(control.lookup_valid[$i])"
    bridgePorts += s".io_lookup_${i}_robId(control.lookup_id[$i])"
  }
  bridgePorts += ".io_lookupTag_tag(8'd0)"
  bridgePorts ++= Seq(
    ".io_allocation_valid(control.allocation_valid)",
    ".io_allocation_bits(control.allocation_id)",
    ".io_retired(control.retired)"
  )
  val instructionStart =
    flatten(bridgeSnapshot).takeWhile { case (name, _) => !name.startsWith("instruction_") }.map(_._2).sum
  bridgeDefinitions += s"`define AB_INSTRUCTION_OFFSET $instructionStart"
  Files.writeString(
    output.resolve("admission_bridge_config.svh"),
    "`ifndef ADMISSION_BRIDGE_CONFIG\n`define ADMISSION_BRIDGE_CONFIG\n" + bridgeDefinitions.mkString(
      "\n"
    ) + "\n`endif\n"
  )
  Files.writeString(output.resolve("admission_bridge_ports.svh"), bridgePorts.mkString(",\n") + "\n")

  val preparationConfig = memcore.memory.preflight.Params(bus = memory.chi, beatBytes = 64)

  val preparationRegions = Seq(
    PhysicalRegion(BigInt("80000000", 16), BigInt(16) << 20, true, true, true, true, true, true),
    PhysicalRegion(BigInt("90000000", 16), BigInt(4096), false, false, true, false, false, true)
  )

  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new PermissionVerification(preparationConfig, preparationRegions)(supervisorParameters),
    firtoolOpts = args,
    args = Array("--target-dir", output.resolve("PermissionVerification").toString, "--split-verilog")
  )
  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new Preparation(preparationConfig, preparationRegions)(supervisorParameters),
    firtoolOpts = args,
    args = Array("--target-dir", output.resolve("Preparation").toString, "--split-verilog")
  )

}
