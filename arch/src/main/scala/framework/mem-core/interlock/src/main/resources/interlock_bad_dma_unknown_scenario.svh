// verilog_syntax: parse-as-statements
@(negedge clock);
done.valid = 1;
`ILF(done.bits, DONE, TAG) = 7;
`ILF(done.bits, DONE, OK) = 1;
