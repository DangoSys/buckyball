// verilog_syntax: parse-as-statements
reserve(7);
info(7, 1, 'h4000, 64, 0, 1);
@(negedge clock);
cancel.valid = 1;
`ILF(cancel.bits, CANCEL, ID) = 7;
