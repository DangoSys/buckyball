package examples.balls.redmule

import chisel3.experimental.hierarchy.{instantiable, public}

import chisel3._
import chisel3.util.HasBlackBoxInline

@instantiable
class RedmuleNative extends BlackBox with HasBlackBoxInline {

  @public
  val io = IO(new Bundle {
    val clk_i        = Input(Clock())
    val rst_ni       = Input(Bool())
    val busy_o       = Output(Bool())
    val evt_o        = Output(Bool())
    val ctrl_req_i   = Input(Bool())
    val ctrl_wen_i   = Input(Bool())
    val ctrl_addr_i  = Input(UInt(32.W))
    val ctrl_data_i  = Input(UInt(32.W))
    val ctrl_gnt_o   = Output(Bool())
    val ctrl_valid_o = Output(Bool())
    val ctrl_data_o  = Output(UInt(32.W))
    val mem_req_o    = Output(Bool())
    val mem_wen_o    = Output(Bool())
    val mem_addr_o   = Output(UInt(32.W))
    val mem_be_o     = Output(UInt(32.W))
    val mem_data_o   = Output(UInt(256.W))
    val mem_id_o     = Output(UInt(8.W))
    val mem_gnt_i    = Input(Bool())
    val mem_valid_i  = Input(Bool())
    val mem_data_i   = Input(UInt(256.W))
    val mem_id_i     = Input(UInt(8.W))
    val mem_ready_o  = Output(Bool())
  })

  // Locate RTL from the source checkout, independent of the elaboration directory.
  val sourceDir  = os.Path(sourcecode.File()) / os.up
  val repoRoot   = sourceDir / os.up / os.up / os.up / os.up / os.up
  val thirdparty = repoRoot / "arch" / "thirdparty"
  val common     = thirdparty / "common_cells"
  val fp         = thirdparty / "cvfpu" / "src"
  val hci        = thirdparty / "hci" / "rtl"
  val stream     = thirdparty / "hwpe-stream" / "rtl"
  val ctrl       = thirdparty / "hwpe-ctrl" / "rtl"
  val native     = thirdparty / "redmule" / "rtl"
  for (name <- Seq("assertions.svh", "registers.svh")) {
    setInline(s"common_cells/$name", os.read(common / "include" / "common_cells" / name))
  }
  setInline("hci_helpers.svh", os.read(hci / "common" / "hci_helpers.svh"))

  val packages = Seq(
    common / "src" / "cf_math_pkg.sv",
    fp / "fpnew_pkg.sv",
    stream / "hwpe_stream_package.sv",
    ctrl / "hwpe_ctrl_package.sv",
    hci / "common" / "hci_package.sv",
    native / "redmule_pkg.sv",
    native / "ctrl" / "regif" / "redmule_regif_pkg.sv",
    stream / "hwpe_stream_interfaces.sv",
    ctrl / "hwpe_ctrl_interfaces.sv",
    hci / "common" / "hci_interfaces.sv"
  )

  val primitives = Seq(
    common / "src" / "lzc.sv",
    common / "src" / "edge_detect.sv",
    common / "src" / "sync.sv",
    common / "src" / "sync_wedge.sv",
    common / "src" / "fifo_v3.sv",
    thirdparty / "tech_cells_generic" / "src" / "rtl" / "tc_clk.sv",
    thirdparty / "tech_cells_generic" / "src" / "deprecated" / "cluster_clk_cells.sv",
    thirdparty / "tech_cells_generic" / "src" / "deprecated" / "pulp_clk_cells.sv"
  ) ++ Seq("fpnew_classifier.sv", "fpnew_rounding.sv", "fpnew_cast_multi.sv", "lfsr_sr.sv").map(fp / _)

  val streamSources  =
    Seq("basic", "fifo", "streamer", "tcdm").flatMap(dir => os.list(stream / dir).filter(_.ext == "sv"))
  val hciSources     =
    os.walk(hci / "core").filter(_.ext == "sv") ++ Seq("hci_ecc_enc.sv", "hci_ecc_dec.sv").map(hci / "ecc" / _)
  val controlSources = Seq("hwpe_ctrl_target.sv", "hwpe_ctrl_seq_mult.sv", "hwpe_ctrl_partial_mult.sv").map(ctrl / _)

  val nativeSources = os.walk(native).filter(p =>
    p.ext == "sv" && !packages.contains(p) &&
      !Seq("redmule_xif_wrap.sv", "redmule_mm_wrap.sv", "redmule_inst_decoder.sv").contains(p.last)
  )

  // redmule_pkg.sv imports its dependencies at file scope, before the package.
  // vvac gathers every file-scope item into one $unit placed at the first of
  // them, so the later `import redmule_pkg::*` would precede redmule_pkg.
  // Move these imports inside the package; everything else is copied verbatim.
  val redmulePkgHead =
    "import fpnew_pkg::*;\nimport hci_package::*;\nimport hwpe_stream_package::*;\n\npackage redmule_pkg;\n"
  val redmulePkg     = os.read(native / "redmule_pkg.sv")
  require(redmulePkg.contains(redmulePkgHead), "redmule_pkg.sv header changed")

  val redmulePkgFixed = redmulePkg.replace(
    redmulePkgHead,
    "package redmule_pkg;\n  import fpnew_pkg::*;\n  import hci_package::*;\n  import hwpe_stream_package::*;\n"
  )

  val wrapper = sourceDir / "RedmuleNative.sv"
  // Only the external wrapper is authored here.
  setInline(
    "RedmuleNative.sv",
    "/* verilator lint_off ENUMVALUE */\n" +
      (packages ++ primitives ++ streamSources ++ hciSources ++
        controlSources ++ nativeSources).map(p =>
        if (p == native / "redmule_pkg.sv") redmulePkgFixed else os.read(p)
      ).mkString("\n") +
      "\n/* verilator lint_on ENUMVALUE */\n" + os.read(wrapper)
  )
}
