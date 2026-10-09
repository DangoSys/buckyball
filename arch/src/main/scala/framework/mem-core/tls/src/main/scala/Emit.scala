package memcore.memory.tls

import java.nio.file.{Files, Paths}

object Emit extends App {
  val memory = memcore.memory.spm.Params(base = BigInt("10000", 16), bytes = 64)
  val output = Paths.get("src/main/scala/framework/mem-core/tls/build")
  Files.createDirectories(output)
  Files.writeString(
    output.resolve("config.svh"),
    s"`define LOCAL_BASE 64'h${memory.base.toString(16)}\n`define LOCAL_BYTES ${memory.bytes}\n"
  )
  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new Store(memory),
    firtoolOpts = args,
    args = Array("--target-dir", output.resolve("Store").toString, "--split-verilog")
  )
}
