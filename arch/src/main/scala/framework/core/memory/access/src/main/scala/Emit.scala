package framework.core.memory.access

import chisel3._
import chisel3.reflect.DataMirror
import java.nio.file.{Files, Paths}
import framework.core.memory.access.configs.Params
import framework.ip.bus.chi.rnf.{CacheAccess, CacheResult}

object Emit extends App {
  val Array("--config", config, "--output-dir", directory) = args
  val p                                                    = upickle.default.read[Params](Files.readString(Paths.get(config)))
  Files.createDirectories(Paths.get(directory))

  val defs = scala.collection.mutable.ArrayBuffer[String](
    s"`define CPU_PHYSICAL_BITS ${p.memory.chi.addressBits}",
    s"`define CPU_TAG_BITS ${p.memory.tagBits}",
    s"`define CPU_LINE_BITS ${p.lineBits}"
  )

  val ports = scala.collection.mutable.ArrayBuffer[String]()

  def channel(
    kind:   String,
    prefix: String,
    signal: String,
    bundle: Bundle
  ): Unit = {
    var offset = 0
    for ((name, field) <- bundle.elements.toSeq) {
      val width = field.getWidth
      if (width > 0) {
        defs += s"`define CPU_${kind}_${name.toUpperCase}_OFFSET $offset";
        defs += s"`define CPU_${kind}_${name.toUpperCase}_WIDTH $width";
        ports += s".${prefix}_bits_$name($signal.bits[$offset +: $width])"; offset += width
      }
    }
    defs += s"`define ACCESS_${kind}_WIDTH $offset"
    ports += s".${prefix}_valid($signal.valid)"; ports += s".${prefix}_ready($signal.ready)"
  }

  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new PhysicalAccess(p),
    args = Array("--target-dir", s"$directory/PhysicalAccess", "--split-verilog")
  )
  channel("REQ", "io_request", "request", new AccessRequest(p))
  channel("RESP", "io_response", "response", new AccessResponse(p))
  channel("CACHE", "io_cacheRequest", "cache_request", new CacheAccess(p.memory.chi))
  channel("CRESULT", "io_cacheResponse", "cache_response", new CacheResult(p.lineBits))
  channel("UNCACHED", "io_uncachedRequest", "uncached_request", new UncachedRequest(p.memory))
  channel("URESULT", "io_uncachedResponse", "uncached_response", new UncachedResponse(p.memory))
  Files.writeString(Paths.get(directory, "physical_parameters.svh"), defs.mkString("\n") + "\n")
  Files.writeString(Paths.get(directory, "physical_ports.svh"), ports.mkString(",\n") + "\n")

  val definitions = scala.collection.mutable.ArrayBuffer[String](
    s"`define VA_LINE_BITS ${p.lineBits}",
    s"`define VA_ADDRESS_BITS ${p.memory.chi.addressBits}"
  )

  val binding     = scala.collection.mutable.ArrayBuffer[String]()
  var inputOffset = 0; var outputOffset = 0

  def fields(data: Data, path: String): Seq[(String, Data)] = data match {
    case record: Record => record.elements.toSeq.flatMap { case (name, field) =>
        fields(field, if (path.isEmpty) name else s"${path}_$name")
      }
    case vector: Vec[_] => vector.toSeq.zipWithIndex.flatMap { case (field, index) => fields(field, s"${path}_$index") }
    case leaf => if (leaf.getWidth > 0) Seq(path -> leaf) else Seq.empty
  }

  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    {
      val dut = new VirtualAccess(p)
      for ((name, field) <- fields(dut.io, "")) {
        val input     = DataMirror.directionOf(field) == ActualDirection.Input
        val direction = if (input) "IN" else "OUT"; val signal              = if (input) "inputs" else "outputs";
        val offset    = if (input) inputOffset else outputOffset; val width = field.getWidth
        definitions += s"`define VA_${direction}_${name.toUpperCase}_OFFSET $offset";
        definitions += s"`define VA_${direction}_${name.toUpperCase}_WIDTH $width"
        binding += s".io_$name(bus.$signal[$offset +: $width])"
        if (input) inputOffset += width else outputOffset += width
      }; dut
    },
    args = Array("--target-dir", s"$directory/VirtualAccess", "--split-verilog")
  )
  definitions += s"`define VA_INPUT_BITS $inputOffset"; definitions += s"`define VA_OUTPUT_BITS $outputOffset"
  Files.writeString(Paths.get(directory, "virtual_parameters.svh"), definitions.mkString("\n") + "\n")
  Files.writeString(Paths.get(directory, "virtual_ports.svh"), binding.mkString(",\n") + "\n")
}
