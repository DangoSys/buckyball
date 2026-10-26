set core_clock [get_ports auto_chipyard_prcictrl_domain_reset_setter_clock_in_member_allClocks_uncore_clock]
set debug_clock [get_ports debug_clock]
set jtag_clock [get_ports debug_systemjtag_jtag_TCK]
set serial_tl_clock [get_ports serial_tl_0_clock_in]
set jtag_inputs [get_ports {debug_systemjtag_jtag_TMS debug_systemjtag_jtag_TDI}]
set jtag_outputs [get_ports debug_systemjtag_jtag_TDO_data]
set serial_tl_inputs [get_ports {serial_tl_0_in_valid serial_tl_0_in_bits_phit* serial_tl_0_out_ready}]
set serial_tl_outputs [get_ports {serial_tl_0_in_ready serial_tl_0_out_valid serial_tl_0_out_bits_phit*}]
set core_inputs [get_ports {
  debug_dmactiveAck
  custom_boot
  i2c_0_scl_in
  i2c_0_sda_in
  uart_0_rxd
  gpio_0_pins_0_i_ival
  gpio_0_pins_1_i_ival
  gpio_0_pins_2_i_ival
  gpio_0_pins_3_i_ival
  gpio_0_pins_4_i_ival
  gpio_0_pins_5_i_ival
  gpio_0_pins_6_i_ival
  gpio_0_pins_7_i_ival
  spi_0_dq_0_i
  spi_0_dq_1_i
  spi_0_dq_2_i
  spi_0_dq_3_i
}]
set core_outputs [get_ports {
  i2c_0_scl_oe
  i2c_0_sda_oe
  uart_0_txd
  gpio_0_pins_0_o_oval
  gpio_0_pins_0_o_oe
  gpio_0_pins_0_o_ie
  gpio_0_pins_1_o_oval
  gpio_0_pins_1_o_oe
  gpio_0_pins_1_o_ie
  gpio_0_pins_2_o_oval
  gpio_0_pins_2_o_oe
  gpio_0_pins_2_o_ie
  gpio_0_pins_3_o_oval
  gpio_0_pins_3_o_oe
  gpio_0_pins_3_o_ie
  gpio_0_pins_4_o_oval
  gpio_0_pins_4_o_oe
  gpio_0_pins_4_o_ie
  gpio_0_pins_5_o_oval
  gpio_0_pins_5_o_oe
  gpio_0_pins_5_o_ie
  gpio_0_pins_6_o_oval
  gpio_0_pins_6_o_oe
  gpio_0_pins_6_o_ie
  gpio_0_pins_7_o_oval
  gpio_0_pins_7_o_oe
  gpio_0_pins_7_o_ie
  spi_0_sck
  spi_0_dq_0_o
  spi_0_dq_0_ie
  spi_0_dq_0_oe
  spi_0_dq_1_o
  spi_0_dq_1_ie
  spi_0_dq_1_oe
  spi_0_dq_2_o
  spi_0_dq_2_ie
  spi_0_dq_2_oe
  spi_0_dq_3_o
  spi_0_dq_3_ie
  spi_0_dq_3_oe
  spi_0_cs_0
}]

create_clock -name core -period 10.0 $core_clock
create_clock -name debug -period 10.0 $debug_clock
create_clock -name jtag -period 100.0 $jtag_clock
create_clock -name serial_tl -period 10.0 $serial_tl_clock
set_clock_groups -asynchronous \
  -group [get_clocks core] \
  -group [get_clocks debug] \
  -group [get_clocks jtag] \
  -group [get_clocks serial_tl]

set_clock_uncertainty -setup 3.0 [get_clocks {core debug serial_tl}]
set_clock_uncertainty -setup 10.0 [get_clocks jtag]
set_clock_transition 1.0 [get_clocks {core debug serial_tl}]
set_clock_transition 10.0 [get_clocks jtag]

set_input_delay 3.0 -clock core $core_inputs
set_output_delay 3.0 -clock core $core_outputs

set_input_delay 20.0 -clock_fall -clock jtag $jtag_inputs
set_output_delay 20.0 -clock jtag $jtag_outputs
set_input_delay 3.0 -clock serial_tl $serial_tl_inputs
set_output_delay 3.0 -clock serial_tl $serial_tl_outputs
