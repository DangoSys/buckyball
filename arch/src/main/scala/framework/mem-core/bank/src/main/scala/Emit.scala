package memcore.memory.bank

object Emit extends App {
  val p = BankSetParams(dataBits = 32, banks = 4, entriesPerBank = 16)
  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new Bank(p),
    firtoolOpts = args,
    args = Array("--target-dir", "build")
  )
  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new BankSet(p),
    firtoolOpts = args,
    args = Array("--target-dir", "build")
  )
}
