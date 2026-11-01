package framework.device.shm

import framework.ip.bus.axi4

object Emit extends App {
  val Array("--config", config, "--output-dir", output) = args
  val v                                                 = ujson.read(os.read(os.Path(config)))

  val p = Params(
    BigInt(v("capacity").str),
    v("cores").num.toInt,
    axi4.Params(v("addressBits").num.toInt, v("dataBits").num.toInt, v("idBits").num.toInt)
  )

  val clients = v("clients").num.toInt
  require(
    clients == 2 && p.capacity == 8192 && p.axi.dataBits == 128,
    "The device storage verification interface requires two clients and 8 KiB"
  )
  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new Storage(p, clients),
    args = Array("--target-dir", s"$output/Storage", "--split-verilog")
  )
}
