// Format Version: 2
// Reviewed for Emit: dataBits=32, banks=4, entriesPerBank=16, tagBits=8.

CHECKSUM: "1481319037 3871254824"
INSTANCE: bankset_tb.dut
ANNOTATION: "Nonempty-command assertion failure diagnostics; legal commands have beats > 0."
Block 2 "555555334" "if (1)"
ANNOTATION: "Nonempty-command assertion failure diagnostics; legal commands have beats > 0."
Block 3 "3006158740" "$error(\"Assertion failed: BankSet command must be nonempty\n    at BankSet.scala:61 assert(io.command.bits.beats =/= 0.U, \\"BankSet command must be nonempty\\")\n\");"
ANNOTATION: "Nonempty-command assertion failure diagnostics; legal commands have beats > 0."
Block 5 "555555334" "if (1)"
ANNOTATION: "Nonempty-command assertion failure diagnostics; legal commands have beats > 0."
Block 6 "1956737205" "$fatal;"
ANNOTATION: "Alignment assertion failure diagnostics; commands are 4-byte aligned."
Block 10 "530868893" "if (1)"
ANNOTATION: "Alignment assertion failure diagnostics; commands are 4-byte aligned."
Block 11 "181752484" "$error(\"Assertion failed: BankSet command must be beat aligned\n    at BankSet.scala:62 assert(io.command.bits.addr %%%% p.bytes.U === 0.U, \\"BankSet command must be beat aligned\\")\n\");"
ANNOTATION: "Alignment assertion failure diagnostics; commands are 4-byte aligned."
Block 13 "530868893" "if (1)"
ANNOTATION: "Alignment assertion failure diagnostics; commands are 4-byte aligned."
Block 14 "3298909189" "$fatal;"
ANNOTATION: "Capacity assertion failure diagnostics; command end is at most 256 bytes."
Block 18 "2870240055" "if (1)"
ANNOTATION: "Capacity assertion failure diagnostics; command end is at most 256 bytes."
Block 19 "3417018178" "$error(\"Assertion failed: BankSet command exceeds capacity\n    at BankSet.scala:63 assert(\n\");"
ANNOTATION: "Capacity assertion failure diagnostics; command end is at most 256 bytes."
Block 21 "2870240055" "if (1)"
ANNOTATION: "Capacity assertion failure diagnostics; command end is at most 256 bytes."
Block 22 "2272232831" "$fatal;"
ANNOTATION: "TLAST assertion failure diagnostics; last is asserted on exactly the final write beat."
Block 26 "1487852324" "if (1)"
ANNOTATION: "TLAST assertion failure diagnostics; last is asserted on exactly the final write beat."
Block 27 "1783058220" "$error(\"Assertion failed: BankSet write TLAST does not match command length\n    at BankSet.scala:87 assert(io.write.bits.last === last, \\"BankSet write TLAST does not match command length\\")\n\");"
ANNOTATION: "TLAST assertion failure diagnostics; last is asserted on exactly the final write beat."
Block 29 "1487852324" "if (1)"
ANNOTATION: "TLAST assertion failure diagnostics; last is asserted on exactly the final write beat."
Block 30 "605788389" "$fatal;"

CHECKSUM: "1481319037 1542845018"
INSTANCE: bankset_tb.dut
ANNOTATION: "Assertion enable is disabled during reset; no command is issued in reset."
Condition 1 "3864235482" "(_GEN & ((~reset))) 1 -1" (2 "10")
ANNOTATION: "A valid command cannot have zero beats."
Condition 2 "4106865980" "(_GEN_10 & (io_command_bits_beats == 32'b0)) 1 -1" (3 "11")
ANNOTATION: "All driven addresses are 4-byte aligned."
Condition 4 "3972072533" "(_GEN_10 & ((|_GEN_11[2:0]))) 1 -1" (1 "01")
ANNOTATION: "All driven addresses are 4-byte aligned."
Condition 4 "3972072533" "(_GEN_10 & ((|_GEN_11[2:0]))) 1 -1" (3 "11")
ANNOTATION: "All driven commands end within the 256-byte capacity."
Condition 5 "3649435992" "(_GEN_10 & (({28'b0, io_command_bits_addr} + {2'b0, io_command_bits_beats, 2'b0}) > 36'h000000100)) 1 -1" (1 "01")
ANNOTATION: "All driven commands end within the 256-byte capacity."
Condition 5 "3649435992" "(_GEN_10 & (({28'b0, io_command_bits_addr} + {2'b0, io_command_bits_beats, 2'b0}) > 36'h000000100)) 1 -1" (3 "11")
ANNOTATION: "No write handshake in reset; accepted TLAST matches the command length."
Condition 6 "3432548480" "(_GEN_9 & ((~reset)) & (io_write_bits_tlast != last)) 1 -1" (2 "101")
ANNOTATION: "No write handshake in reset; accepted TLAST matches the command length."
Condition 6 "3432548480" "(_GEN_9 & ((~reset)) & (io_write_bits_tlast != last)) 1 -1" (4 "111")
ANNOTATION: "writeResponse follows an accepted Bank write; that Bank holds response.valid until ready."
Condition 9 "618862985" "(_GEN_3 & _GEN_13) 1 -1" (2 "10")
ANNOTATION: "Only one request is outstanding; a read is issued only after the preceding response retires."
Condition 13 "3211095629" "(_GEN_1 & _GEN_8) 1 -1" (2 "10")
ANNOTATION: "Only one request is outstanding; writing selects an idle Bank after the preceding response retires."
Condition 36 "1292920334" "(_io_write_ready_T & _GEN_8) 1 -1" (2 "10")

CHECKSUM: "1481319037 745494789"
INSTANCE: bankset_tb.dut
ANNOTATION: "Four-byte aligned addresses have low two bits zero."
Toggle io_command_bits_addr [1:0] "net io_command_bits_addr[7:0]"
ANNOTATION: "Four-byte aligned addresses have low two bits zero."
Toggle command_addr [1:0] "reg command_addr[7:0]"
ANNOTATION: "Capacity is 64 words; legal lengths and completed-beat count cannot exceed 64."
Toggle io_command_bits_beats [31:7] "net io_command_bits_beats[31:0]"
ANNOTATION: "Capacity is 64 words; legal lengths and completed-beat count cannot exceed 64."
Toggle command_beats [31:7] "reg command_beats[31:0]"
ANNOTATION: "Capacity is 64 words; legal lengths and completed-beat count cannot exceed 64."
Toggle beat [31:7] "reg beat[31:0]"
ANNOTATION: "Read responses contain a full word; TKEEP is always 4'b1111."
Toggle io_read_bits_tkeep "net io_read_bits_tkeep[3:0]"
ANNOTATION: "Completion error is tied to zero; invalid commands are assertions, not error responses."
Toggle io_done_bits "net io_done_bits"
