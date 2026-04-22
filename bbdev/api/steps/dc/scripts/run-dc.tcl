# DC synthesis script.
# Inputs (set via `dc_shell -x` before sourcing this file):
#   design_dir, top_module, work_dir, report_dir, tmp_dir
#   target_libs        : space-separated list of .db files
#   clock_name, clock_period, clock_uncertainty, clock_transition
#   input_delay, output_delay, input_transition, output_load

set search_path [list . $design_dir]
define_design_lib work -path $tmp_dir

set target_library $target_libs
set link_library   $target_libs

set file_list [glob -nocomplain -directory $design_dir *.sv]
analyze -format sverilog $file_list
elaborate $top_module
current_design $top_module
link

# Keep module boundaries for hierarchical reports.
set_ungroup [get_designs *] false
set_boundary_optimization [get_designs *] false

create_clock -name $clock_name -period $clock_period [get_ports $clock_name]
set_clock_uncertainty $clock_uncertainty [get_clocks $clock_name]
set_input_delay  $input_delay  -clock $clock_name [remove_from_collection [all_inputs] [get_ports $clock_name]]
set_output_delay $output_delay -clock $clock_name [all_outputs]
set_clock_equivalence $clock_name
set_load $output_load [all_outputs]
set_input_transition $input_transition [remove_from_collection [all_inputs] [get_ports $clock_name]]
set_clock_transition $clock_transition [get_clocks $clock_name]

compile_ultra -retime -scan -no_autoungroup
write -format ddc -hierarchy -output $report_dir/design_compiled.ddc

report_area -hierarchy -nosplit > $report_dir/area.rpt
report_hierarchy -noleaf        > $report_dir/hierarchy.rpt
report_timing                   > $report_dir/timing.rpt
report_power -hierarchy         > $report_dir/power.rpt

write -format verilog -output $report_dir/netlist.v
exit
