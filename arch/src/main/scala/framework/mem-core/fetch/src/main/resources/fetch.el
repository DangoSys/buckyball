// Format Version: 2
// 4-byte groups, 16-bit halfword/RVC alignment, one 64-bit word request.
CHECKSUM: "1717534490 3493169012"
INSTANCE: fetch_tb.dut
ANNOTATION: "Assertion-failure diagnostic only; assertion remains enabled: Fetch privilege must be U, S or M"
Block 2 "3303546747" "if (1)"
Block 3 "784000845" "$error(\"Assertion failed: Fetch privilege must be U, S or M\n    at Fetch.scala:73 assert(io.context.privilege =/= 2.U, \\"Fetch privilege must be U, S or M\\")\n\");"
Block 5 "3303546747" "if (1)"
Block 6 "2616558672" "$fatal;"
ANNOTATION: "Assertion-failure diagnostic only; assertion remains enabled: Fetch reset vector must be instruction aligned"
Block 10 "1637879550" "if (1)"
Block 11 "1142822639" "$error(\"Assertion failed: Fetch reset vector must be instruction aligned\n    at Fetch.scala:74 assert(pc(log2Ceil(p.instructionBytes) - 1, 0) === 0.U, \\"Fetch reset vector must be instruction aligned\\")\n\");"
Block 13 "1637879550" "if (1)"
Block 14 "2917930836" "$fatal;"
ANNOTATION: "Assertion-failure diagnostic only; assertion remains enabled: Fetch maintenance response must acknowledge completion"
Block 18 "2033796801" "if (1)"
Block 19 "2717912749" "$error(\"Assertion failed: Fetch maintenance response must acknowledge completion\n    at Fetch.scala:105 assert(io.maintained.bits, \\"Fetch maintenance response must acknowledge completion\\")\n\");"
Block 21 "2033796801" "if (1)"
Block 22 "2015233156" "$fatal;"
ANNOTATION: "Assertion-failure diagnostic only; assertion remains enabled: Fetch redirect must be instruction aligned"
Block 26 "2247434341" "if (1)"
Block 27 "1751189408" "$error(\"Assertion failed: Fetch redirect must be instruction aligned\n    at Fetch.scala:109 assert(io.redirect.bits(log2Ceil(p.instructionBytes) - 1, 0) === 0.U, \\"Fetch redirect must be instruction aligned\\")\n\");"
Block 29 "2247434341" "if (1)"
Block 30 "1248359523" "$fatal;"
ANNOTATION: "Assertion-failure diagnostic only; assertion remains enabled: Fetch flush requires a restart redirect"
Block 34 "1762317919" "if (1)"
Block 35 "4112421248" "$error(\"Assertion failed: Fetch flush requires a restart redirect\n    at Fetch.scala:115 assert(io.redirect.valid, \\"Fetch flush requires a restart redirect\\")\n\");"
Block 37 "1762317919" "if (1)"
Block 38 "564188382" "$fatal;"

CHECKSUM: "1717534490 1587966699"
INSTANCE: fetch_tb.dut
ANNOTATION: "Only full enabled contract-failure vectors: privilege 2, odd start/redirect PC, incomplete maintenance, or flush without restart. Each is independently tested by an exact negative target; all idle/reset/pure comparator vectors remain measured."
Condition 2 "1830893280" "(_GEN_1 & (io_context_privilege == 2'h2)) 1 -1" (3 "11")
Condition 4 "2086071732" "(_GEN_1 & pc[0]) 1 -1" (3 "11")
Condition 5 "1305965348" "(_GEN_0 & ((~reset)) & ((~io_maintained_bits))) 1 -1" (4 "111")
Condition 6 "1743659560" "(io_redirect_valid & ((~reset)) & io_redirect_bits[0]) 1 -1" (4 "111")
Condition 7 "2120892050" "(io_flush & ((~reset)) & ((~io_redirect_valid))) 1 -1" (4 "111")

CHECKSUM: "1717534490 202402385"
INSTANCE: fetch_tb.dut
ANNOTATION: "Word addresses are assigned pc & ~7; execute and maintenance request bits are literal true. Every deliverable packet comes from an accepted halfword-aligned fetch, so PC bit 0 is zero; its full 4-byte group mask is 11 or 10, so mask bit 1 is one. Reset-only PC/readPc/input values remain measured, and no instruction data is waived."
Toggle io_packet_bits_pc [0] "net io_packet_bits_pc[63:0]"
Toggle io_packet_bits_mask [1] "net io_packet_bits_mask[1:0]"
Toggle io_request_bits_addr [2:0] "net io_request_bits_addr[63:0]"
Toggle io_request_bits_execute "net io_request_bits_execute"
Toggle io_maintenance_bits "net io_maintenance_bits"
Toggle command_addr [2:0] "reg command_addr[63:0]"
Toggle answer_pc [0] "reg answer_pc[63:0]"
Toggle answer_mask [1] "reg answer_mask[1:0]"
