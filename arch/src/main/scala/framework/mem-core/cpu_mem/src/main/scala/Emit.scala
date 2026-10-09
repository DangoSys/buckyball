package memcore.memory.cpu

import java.nio.file.{Files, Paths}

object Emit extends App {
  val p = CpuMemParams()
  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new CpuMem(p),
    firtoolOpts = args,
    args = Array("--target-dir", "build/CpuMem", "--split-verilog")
  )

  val definitions = scala.collection.mutable.ArrayBuffer[String](
    s"`define CPU_PHYSICAL_BITS ${p.chi.addressBits}",
    s"`define CPU_TAG_BITS ${p.tagBits}"
  )

  val ports = scala.collection.mutable.ArrayBuffer[String]()

  def channel(
    kind:     String,
    prefix:   String,
    instance: String,
    fields:   Seq[(String, Int)]
  ): Unit = {
    var offset = 0
    for ((field, width) <- fields) {
      definitions += s"`define CPU_${kind}_${field.toUpperCase}_OFFSET $offset"
      definitions += s"`define CPU_${kind}_${field.toUpperCase}_WIDTH $width"
      ports += s".$prefix$field($instance.bits[$offset +: $width])"
      offset += width
    }
    definitions += s"`define CPU_${kind}_WIDTH $offset"
  }

  channel(
    "REQ",
    "io_request_bits_",
    "request",
    Seq(
      "addr"      -> 64,
      "tag"       -> p.tagBits,
      "size"      -> 3,
      "write"     -> 1,
      "signed"    -> 1,
      "data"      -> 64,
      "atomic"    -> 4,
      "cacheable" -> 1,
      "normal"    -> 1
    )
  )
  channel(
    "RESP",
    "io_response_bits_",
    "response",
    Seq("tag" -> p.tagBits, "data" -> 64, "misaligned" -> 1, "accessFault" -> 1)
  )
  channel(
    "CACHE",
    "io_cacheRequest_bits_",
    "cache_request",
    Seq("addr" -> p.chi.addressBits, "write" -> 1, "data" -> 64, "mask" -> 8, "atomic" -> 4, "atomicWord" -> 1)
  )
  channel("CRESULT", "io_cacheResponse_bits_", "cache_response", Seq("data" -> 64, "error" -> 1))
  channel(
    "UNCACHED",
    "io_uncachedRequest_bits_",
    "uncached_request",
    Seq("addr" -> 64, "tag" -> p.tagBits, "size" -> 3, "write" -> 1, "data" -> 64)
  )
  channel(
    "URESULT",
    "io_uncachedResponse_bits_",
    "uncached_response",
    Seq("tag" -> p.tagBits, "data" -> 64, "error" -> 1)
  )
  for (
    (prefix, instance) <- Seq(
                            "request"          -> "request",
                            "response"         -> "response",
                            "cacheRequest"     -> "cache_request",
                            "cacheResponse"    -> "cache_response",
                            "uncachedRequest"  -> "uncached_request",
                            "uncachedResponse" -> "uncached_response"
                          )
  ) {
    ports += s".io_${prefix}_valid($instance.valid)"
    ports += s".io_${prefix}_ready($instance.ready)"
  }
  Files.write(Paths.get("build/cpu_mem_config.svh"), (definitions.mkString("\n") + "\n").getBytes)
  Files.write(Paths.get("build/cpu_mem_ports.svh"), ports.mkString(",\n").getBytes)
}
