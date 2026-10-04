package framework.memdomain.verification

import chisel3._
import chisel3.reflect.DataMirror
import chisel3.ActualDirection
import framework.system.core.rocket.CpuParams
import java.nio.file.{Files, Paths}
import framework.system.configloader.{ChipLoader, RocketTileCore}
import framework.memdomain.frontend.mem.{KernelDma, MemLoader, MemStorer}

object Emit extends App {
  require(
    args.length >= 4 && args(0) == "--chip" && args(2) == "--targets",
    "Usage: memdomain_ack.run --chip <chip> --targets <comma-separated targets> [firtool options]"
  )
  val chip    = args(1)
  val targets = args(3).split(",").toSet

  val known = Set(
    "loader_ack",
    "storer_ack",
    "kernel_dma",
    "admission_system",
    "controller_system",
    "memory_system",
    "tile_system_single",
    "tile_system_four"
  )

  require(targets.nonEmpty && targets.subsetOf(known), s"Unknown verification targets: ${targets -- known}")
  val firtoolOptions = args.drop(4)
  require(chip == "toy" || chip == "goban", s"MemDomain gates declare Toy or Goban profiles, got $chip")
  val repo           = Paths.get("..").toAbsolutePath.normalize
  val topology       = ChipLoader.load(repo.resolve(s"examples/chips/$chip/configs/generated/chip.pb").toString)
  val configs        = topology.tiles.flatMap(_.cores).collect { case RocketTileCore(_, Some(config)) => config }
  require(configs.nonEmpty, "MemDomain gate requires a configured accelerator")
  if (chip == "toy") require(configs.size == 1, "Toy ACK gate requires exactly one compute configuration")
  val b              = configs.head
  require(
    b.memDomain.bankWidth == 128 && b.memDomain.bankMaskLen == 16,
    "ACK gate profile requires 16-byte bank rows and byte masks"
  )
  val output         = repo.resolve("arch/src/main/scala/framework/memdomain/verification/build")
  Files.createDirectories(output)

  def leaves(d: Data, name: String): Seq[(String, Data)] = d match {
    case r: Record => r.elements.toSeq.flatMap { case (n, v) => leaves(v, s"${name}_$n") }
    case v: Vec[_] => v.zipWithIndex.flatMap { case (e, i) => leaves(e, s"${name}_$i") }.toSeq
    case e => if (e.getWidth > 0) Seq(name -> e) else Seq.empty
  }

  for (load <- Seq(true, false) if targets(if (load) "loader_ack" else "storer_ack")) {
    var io: Data = null
    val stem = if (load) "loader_ack" else "storer_ack"
    _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
      if (load) { val m = new MemLoader(b); io = m.io; m }
      else { val m = new MemStorer(b); io = m.io; m },
      args = Array("--target-dir", output.resolve(if (load) "MemLoader" else "MemStorer").toString, "--split-verilog"),
      firtoolOpts = firtoolOptions
    )
    val fs   = leaves(io, "io")
    Files.writeString(
      output.resolve(s"${stem}_signals.svh"),
      fs.map { case (n, d) => s"logic [${d.getWidth - 1}:0] $n;" }.mkString("\n") + "\n"
    )
    Files.writeString(
      output.resolve(s"${stem}_ports.svh"),
      (Seq(".clock(clock)", ".reset(reset)") ++ fs.map { case (n, _) => s".$n($n)" }).mkString(",\n") + "\n"
    )
    Files.writeString(
      output.resolve(s"${stem}_init.svh"),
      fs.collect { case (n, d) if DataMirror.directionOf(d) == ActualDirection.Input => s"$n = '0;" }.mkString(
        "\n"
      ) + "\n"
    )
  }

  if (targets("kernel_dma")) {
    var kernelIo: Data = null
    _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
      { val m = new KernelDma(b); kernelIo = m.io; m },
      args = Array("--target-dir", output.resolve("KernelDma").toString, "--split-verilog"),
      firtoolOpts = firtoolOptions
    )
    val kernelFields = leaves(kernelIo, "io")
    Files.writeString(
      output.resolve("kernel_dma_signals.svh"),
      kernelFields.map { case (n, d) => s"logic [${d.getWidth - 1}:0] $n;" }.mkString("\n") + "\n"
    )
    Files.writeString(
      output.resolve("kernel_dma_ports.svh"),
      (Seq(".clock(clock)", ".reset(reset)") ++ kernelFields.map { case (n, _) => s".$n($n)" }).mkString(",\n") + "\n"
    )
    Files.writeString(
      output.resolve("kernel_dma_init.svh"),
      kernelFields.collect {
        case (n, d) if DataMirror.directionOf(d) == ActualDirection.Input => s"$n = '0;"
      }.mkString("\n") + "\n"
    )
    Files.writeString(
      output.resolve("kernel_dma_clocking.svh"),
      "clocking sample @(posedge clock);\ndefault input #1step;\ninput reset;\n" +
        kernelFields.map { case (n, _) => s"input $n;" }.mkString("\n") + "\nendclocking\n"
    )
  }
  if (targets("admission_system") || targets("controller_system")) {
    // Task admission and the controller belong to compute tiles; the main tile has neither.
    val tile       = topology.computeTiles.headOption.getOrElse(
      throw new IllegalArgumentException(s"$chip: admission/controller gates need a compute tile")
    )
    val selected   = tile.cores.zipWithIndex.collectFirst { case (RocketTileCore(cpu, Some(config)), i) =>
      (cpu, config, i)
    }.get
    val cpu        = selected._1; val config = selected._2; val coreIndex = selected._3
    val signatures = tile.signatures.tail
    require(
      signatures.size == tile.cores.size - 1 && config.memDomain.sharedEnable,
      "AdmissionSystem requires the actual shared-bank/task-worker configuration"
    )
    implicit val coreParameters: CpuParams = CpuParams(
      core = framework.system.core.rocket.configs.RocketCpuParam.toRocketCoreParams(
        cpu,
        tile.param.xLen,
        tile.param.pgLevels
      ),
      dcache = Some(freechips.rocketchip.rocket.DCacheParams()),
      icache = Some(freechips.rocketchip.rocket.ICacheParams()),
      btb = None,
      physicalAddressBits = tile.param.paddrBits,
      hartIdBits = 6,
      beatBytes = 16,
      blockBytes = 64
    )
    val prepared = memcore.memory.preflight.Params(beatBytes = 16)
    val regions  =
      Seq(memcore.memory.cpu.PhysicalRegion(BigInt("80000000", 16), BigInt(16) << 20, true, true, true, true, true, true))
    if (targets("admission_system")) {
      var interface: Data = null
      _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
        { val m = new AdmissionSystem(config, prepared, coreIndex, signatures, regions); interface = m.io; m },
        args = Array("--target-dir", output.resolve("AdmissionSystem").toString, "--split-verilog"),
        firtoolOpts = firtoolOptions
      )
      val fields = leaves(interface, "io")
      Files.writeString(
        output.resolve("admission_system_signals.svh"),
        fields.map { case (n, d) => s"logic [${d.getWidth - 1}:0] $n;" }.mkString("\n") + "\n"
      )
      Files.writeString(
        output.resolve("admission_system_ports.svh"),
        (Seq(".clock(clock)", ".reset(reset)") ++ fields.map { case (n, _) => s".$n($n)" }).mkString(",\n") + "\n"
      )
      Files.writeString(
        output.resolve("admission_system_init.svh"),
        fields.collect { case (n, d) if DataMirror.directionOf(d) == ActualDirection.Input => s"$n = '0;" }.mkString(
          "\n"
        ) + "\n"
      )
      Files.writeString(
        output.resolve("admission_system_clocking.svh"),
        "clocking sample @(posedge clock);\ndefault input #1step;\ninput reset;\n" +
          fields.map { case (n, _) => s"input $n;" }.mkString("\n") + "\nendclocking\n"
      )
      Files.writeString(
        output.resolve("admission_system_config.svh"),
        s"`define ADMISSION_ROWS ${config.memDomain.bankEntries}\n`define ADMISSION_SIGNATURE 64'h${config.coreSignature.toString(16)}\n`define ADMISSION_CORE $coreIndex\n"
      )
    }
    if (targets("controller_system")) EmitController(output, config, signatures, firtoolOptions)
  }

  if (targets.exists(_.startsWith("tile_system_"))) {
    require(chip == "goban", "Tile verification profile requires Goban")
    EmitTile(topology, output, firtoolOptions)
  }
  if (targets("memory_system")) EmitMemory(output, firtoolOptions)
}
