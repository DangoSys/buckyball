# Pebble DigitalTop timing. Edit this file for real tapeout constraints.
create_clock -name bb_clock -period 10.0 \
  [get_ports auto_chipyard_prcictrl_domain_reset_setter_clock_in_member_allClocks_uncore_clock]
set_clock_uncertainty 3.0 [get_clocks bb_clock]
set_clock_transition 1.0 [get_clocks bb_clock]
set_input_delay 7.0 -clock bb_clock [all_inputs]
set_output_delay 7.0 -clock bb_clock [all_outputs]
