set clock_port [get_ports {clock clk_i}]
if {[sizeof_collection $clock_port] != 1} { error "synthesis top must have one clock or clk_i input" }
create_clock -name core -period 10.0 $clock_port
set_clock_uncertainty -setup 1.0 [get_clocks core]
set_clock_transition 0.5 [get_clocks core]
set data_inputs [remove_from_collection [all_inputs] $clock_port]
set_input_delay 2.0 -clock core $data_inputs
set_output_delay 2.0 -clock core [all_outputs]
