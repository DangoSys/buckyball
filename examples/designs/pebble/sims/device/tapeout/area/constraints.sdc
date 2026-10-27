# DeviceTop has one clock; every other port is synchronous to it.
set core_clock [get_ports clock]
create_clock -name core -period 10.0 $core_clock

set_clock_uncertainty -setup 3.0 [get_clocks core]
set_clock_transition 1.0 [get_clocks core]

set_input_delay 3.0 -clock core [remove_from_collection [all_inputs] $core_clock]
set_output_delay 3.0 -clock core [all_outputs]
