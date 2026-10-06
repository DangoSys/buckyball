// verilog_syntax: parse-as-statements
reserve(7);
@(negedge clock);
dispatch.valid = 1;
`ILF(dispatch.bits, DISPATCH, ID) = 7;
