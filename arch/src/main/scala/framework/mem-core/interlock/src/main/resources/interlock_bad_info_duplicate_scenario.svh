// verilog_syntax: parse-as-statements
reserve(7);
info(7, 1, 'h4000, 64, 0);
@(negedge clock);
access_info.valid = 1;
