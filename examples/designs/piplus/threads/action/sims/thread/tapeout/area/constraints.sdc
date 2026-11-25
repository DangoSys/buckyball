set clock_port [get_ports {clock clk_i}]
if {[sizeof_collection $clock_port] != 1} { error "synthesis top must have one clock or clk_i input" }
create_clock -name core -period 10.0 $clock_port
set_clock_uncertainty -setup 1.0 [get_clocks core]
set_clock_transition 0.5 [get_clocks core]
set data_inputs [remove_from_collection [all_inputs] $clock_port]
set_input_delay 2.0 -clock core $data_inputs
set_output_delay 2.0 -clock core [all_outputs]
# A thread is a block inside the chip: its ports meet neighbouring logic,
# driven by and loading ordinary buffers, not pads.
set_driving_cell -lib_cell BUFUHDV2 -pin Z $data_inputs
set_load [expr {4 * [load_of [get_lib_pins */BUFUHDV2/I]]}] [all_outputs]
# Wire-load estimates make one driver of a 100+ pin net miss max_transition;
# a fanout limit makes compile build buffer trees instead.
set_max_fanout 16 [current_design]
