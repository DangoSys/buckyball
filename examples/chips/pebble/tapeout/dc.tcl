# Pebble-owned DC entry point. Keep chip timing/configuration in config.toml;
# source bbdev's shared DC implementation.
set bbdev_dc_script [file normalize [file join [file dirname [info script]] .. .. .. .. bbdev api steps dc scripts dc.tcl]]
source $bbdev_dc_script
