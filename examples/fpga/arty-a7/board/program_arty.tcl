# Volatile JTAG programming only. Usage:
# vivado -mode batch -source board/program_arty.tcl -tclargs /path/to/pebble_a7.bit
if {$argc != 1} { error "Supply the Arty A7-100T .bit file" }
set bitfile [file normalize [lindex $argv 0]]
if {![file isfile $bitfile]} { error "Bitstream not found: $bitfile" }
open_hw_manager
connect_hw_server -allow_non_jtag
set targets [get_hw_targets -quiet]
if {[llength $targets] != 1} {
  error "Expected one connected JTAG target. Select the Arty manually in Hardware Manager if multiple boards are attached."
}
open_hw_target [lindex $targets 0]
set devices [get_hw_devices -quiet -filter {PART == xc7a100t}]
if {[llength $devices] != 1} { error "Expected one XC7A100T; inspect the connected board" }
set device [lindex $devices 0]
current_hw_device $device
set_property PROGRAM.FILE $bitfile $device
program_hw_devices $device
refresh_hw_device $device
puts "ARTY_JTAG_PROGRAMMED: $bitfile"
close_hw_target
disconnect_hw_server
close_hw_manager
exit
