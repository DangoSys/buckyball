// verilog_syntax: parse-as-statements
@(negedge clock);
cancel.valid = 1;
`ILF(cancel.bits, CANCEL, ID) = 7;
