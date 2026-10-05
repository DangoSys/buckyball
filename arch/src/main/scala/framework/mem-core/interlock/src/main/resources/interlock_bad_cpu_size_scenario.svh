// verilog_syntax: parse-as-statements
@(negedge clock);
`ILF(cpu_query.bits, CPU_QUERY, VALID) = 1;
`ILF(cpu_query.bits, CPU_QUERY, SIZELOG2) = 4;
