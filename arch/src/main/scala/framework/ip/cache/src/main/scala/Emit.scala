package framework.ip.memory.cache

import java.nio.file.{Files, Paths}
import framework.ip.memory.cache.configs.CacheParams

object Emit extends App {
  val Array("--config", config, "--output-dir", directory) = args
  val values                                               = ujson.read(Files.readString(Paths.get(config)))
  val output                                               = Paths.get(directory)
  Files.createDirectories(output)

  val p = CacheParams(
    values("addressBits").num.toInt,
    values("lineBytes").num.toInt,
    values("sets").num.toInt,
    values("ways").num.toInt,
    values("idBits").num.toInt,
    values("metadataBits").num.toInt,
    values("responseDepth").num.toInt
  )

  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new Cache(p),
    args = Array("--target-dir", output.resolve("Cache").toString, "--split-verilog")
  )

  val constants = Seq(
    "ADDR_BITS"      -> p.addressBits,
    "LINE_BYTES"     -> p.lineBytes,
    "LINE_BITS"      -> p.lineBits,
    "SETS"           -> p.sets,
    "WAYS"           -> p.ways,
    "WAY_BITS"       -> p.wayBits,
    "ID_BITS"        -> p.idBits,
    "META_BITS"      -> p.metadataBits,
    "RESPONSE_DEPTH" -> p.responseDepth
  ).map { case (name, value) => s"`define CACHE_$name $value" }.mkString("\n")

  Files.writeString(
    output.resolve("cache_config.svh"),
    s"`ifndef CACHE_CONFIG_SVH\n`define CACHE_CONFIG_SVH\n$constants\n`endif\n"
  )
}
