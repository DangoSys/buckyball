+define+SUPERVISOR_PROFILE
// SATP PPN/ASID have no architectural reset value; select concrete bits for this 4-state run.
+define+RANDOMIZE_REG_INIT
+define+RANDOM=32'b0
+incdir+@VERIFY@/uvm/src/ip
+incdir+@RTL@
+incdir+@RESOURCES@
@VERIFY@/uvm/src/ip/stream_if.sv
@VERIFY@/uvm/src/ip/package.sv
@RESOURCES@/core_interface.sv
@RESOURCES@/core_package.sv
-F @RTL@/Supervisor/filelist.f
@RESOURCES@/core_tb.sv
