// verilog_syntax: parse-as-statements
@(negedge clock);
access_info.valid = 1;
`ILF(access_info.bits, ACCESS_INFO, ID) = 7;
