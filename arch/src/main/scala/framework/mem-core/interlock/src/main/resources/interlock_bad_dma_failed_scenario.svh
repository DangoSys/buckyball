// verilog_syntax: parse-as-statements
reserve(7);
info(7, 1, 'h4000, 64, 1);
maintenance_accept();
maintenance_ack(7, 1);
do @(posedge clock); while (!maintained.ready);
@(negedge clock);
maintained.valid = 0;
grant.ready = 1;
do @(posedge clock); while (!grant.valid);
@(negedge clock);
grant.ready = 0;
done.valid = 1;
`ILF(done.bits, DONE, TAG) = 7;
`ILF(done.bits, DONE, OK) = 0;
