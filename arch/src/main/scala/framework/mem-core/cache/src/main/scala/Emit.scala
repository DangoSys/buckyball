package memcore.memory.cache

import memcore.memory.cache.configs.CacheParams

object Emit extends App {
  val p = CacheParams.load("configs/default.toml")
  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new Cache(p),
    firtoolOpts = args,
    args = Array("--target-dir", "build/Cache", "--split-verilog")
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

  java.nio.file.Files.writeString(
    java.nio.file.Paths.get("build/cache_config.svh"),
    s"`ifndef CACHE_CONFIG_SVH\n`define CACHE_CONFIG_SVH\n$constants\n`endif\n"
  )
}
