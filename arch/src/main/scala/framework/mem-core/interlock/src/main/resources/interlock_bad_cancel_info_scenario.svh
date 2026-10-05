// verilog_syntax: parse-as-statements
reserve(7);
@(negedge clock);
cancel.valid = 1;
`ILF(cancel.bits, CANCEL, ID) = 7;
access_info.valid = 1;
`ILF(access_info.bits, ACCESS_INFO, ID) = 7;
`ILF(access_info.bits, ACCESS_INFO, LAST) = 1;
