# Toy-owned PTPX entry point. Delegates standard-cell dynamic power to the
# shared bbdev implementation.
set bbdev_ptpx_script [file normalize [file join [file dirname [info script]] .. .. .. .. .. bbdev api steps dc scripts ptpx.tcl]]
source $bbdev_ptpx_script
