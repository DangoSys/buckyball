// verilog_syntax: parse-as-statements
reserve(7);
info(7, 1, 'h4000, 64, 1);
@(negedge clock);
done.valid = 1;
`ILF(done.bits, DONE, TAG) = 7;
`ILF(done.bits, DONE, OK) = 1;
