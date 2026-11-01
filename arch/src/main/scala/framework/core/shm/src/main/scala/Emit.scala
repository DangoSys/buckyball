package framework.core.shm

object Emit extends App {
  val Array("--config", config, "--output-dir", output) = args
  val values                                            = ujson.read(os.read(os.Path(config)))

  val p = MeshSharedMemParams(
    rows = values("rows").num.toInt,
    cols = values("cols").num.toInt,
    entriesPerBank = values("entriesPerBank").num.toInt,
    dataBits = values("dataBits").num.toInt,
    tagBits = values("tagBits").num.toInt,
    threads = values("threads").arr.toSeq.map(t => MeshThreadAttachment(t("bank").num.toInt, t("ports").num.toInt))
  )

  require(p == MeshSharedMemParams.prototype, "The core SHM verification interface requires its declared geometry")
  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new MeshRouter(p, 1, 1),
    args = Array("--target-dir", s"$output/MeshRouter", "--split-verilog")
  )
  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new MeshSharedMem(p),
    args = Array("--target-dir", s"$output/MeshSharedMem", "--split-verilog")
  )
}
