// Format Version: 2
// Reviewed for configs/default.toml: 44-bit addresses, 64-byte lines, 4 sets, 2 ways, responseDepth=2.

CHECKSUM: "1581756980 3097157900"
INSTANCE: cache_tb.dut
ANNOTATION: "Capacity assertion diagnostics: admission reserves count plus pending, so an outstanding pipeline response always has queue space."
Block 2 "3078567325" "if (1)"
ANNOTATION: "Capacity assertion diagnostics: admission reserves count plus pending, so an outstanding pipeline response always has queue space."
Block 3 "1895567494" "$error(\"Assertion failed: Cache response capacity reservation failed\n    at Cache.scala:51 assert(!pending || responses.io.enq.ready, \\"Cache response capacity reservation failed\\")\n\");"
ANNOTATION: "Capacity assertion diagnostics: admission reserves count plus pending, so an outstanding pipeline response always has queue space."
Block 5 "3078567325" "if (1)"
ANNOTATION: "Capacity assertion diagnostics: admission reserves count plus pending, so an outstanding pipeline response always has queue space."
Block 6 "2299949409" "$fatal;"
ANNOTATION: "Opcode assertion diagnostics: the legal interface operations are Lookup, Read, Write, Fill, and Invalidate."
Block 10 "422554957" "if (1)"
ANNOTATION: "Opcode assertion diagnostics: the legal interface operations are Lookup, Read, Write, Fill, and Invalidate."
Block 11 "3963963684" "$error(\"Assertion failed: Unknown Cache operation\n    at Cache.scala:67 assert(req.op <= CacheOp.Invalidate.U, \\"Unknown Cache operation\\")\n\");"
ANNOTATION: "Opcode assertion diagnostics: the legal interface operations are Lookup, Read, Write, Fill, and Invalidate."
Block 13 "422554957" "if (1)"
ANNOTATION: "Opcode assertion diagnostics: the legal interface operations are Lookup, Read, Write, Fill, and Invalidate."
Block 14 "2766194732" "$fatal;"
ANNOTATION: "Alignment assertion diagnostics: requests use cache-line-aligned addresses."
Block 18 "874237168" "if (1)"
ANNOTATION: "Alignment assertion diagnostics: requests use cache-line-aligned addresses."
Block 19 "1020309457" "$error(\"Assertion failed: Cache operations require a line-aligned address\n    at Cache.scala:68 assert(req.addr(p.offsetBits - 1, 0) === 0.U, \\"Cache operations require a line-aligned address\\")\n\");"
ANNOTATION: "Alignment assertion diagnostics: requests use cache-line-aligned addresses."
Block 21 "874237168" "if (1)"
ANNOTATION: "Alignment assertion diagnostics: requests use cache-line-aligned addresses."
Block 22 "1069368929" "$fatal;"
ANNOTATION: "Duplicate-tag assertion diagnostics: reset clears valid, Fill requires an absent tag or its existing way, and Invalidate only clears valid."
Block 26 "1759021865" "if (1)"
ANNOTATION: "Duplicate-tag assertion diagnostics: reset clears valid, Fill requires an absent tag or its existing way, and Invalidate only clears valid."
Block 27 "155016256" "$error(\"Assertion failed: Duplicate Cache tag\n    at Cache.scala:70 assert(PopCount(hits) <= 1.U, \\"Duplicate Cache tag\\")\n\");"
ANNOTATION: "Duplicate-tag assertion diagnostics: reset clears valid, Fill requires an absent tag or its existing way, and Invalidate only clears valid."
Block 29 "1759021865" "if (1)"
ANNOTATION: "Duplicate-tag assertion diagnostics: reset clears valid, Fill requires an absent tag or its existing way, and Invalidate only clears valid."
Block 30 "3067504005" "$fatal;"
ANNOTATION: "Fill assertion diagnostics: the coherence controller must not install the same line in two ways of a set."
Block 34 "1166789632" "if (1)"
ANNOTATION: "Fill assertion diagnostics: the coherence controller must not install the same line in two ways of a set."
Block 35 "1252711441" "$error(\"Assertion failed: Cache fill would create a duplicate tag\n    at Cache.scala:83 assert(!hits.orR || hits(req.way), \\"Cache fill would create a duplicate tag\\")\n\");"
ANNOTATION: "Fill assertion diagnostics: the coherence controller must not install the same line in two ways of a set."
Block 37 "1166789632" "if (1)"
ANNOTATION: "Fill assertion diagnostics: the coherence controller must not install the same line in two ways of a set."
Block 38 "3120397658" "$fatal;"
ANNOTATION: "Write assertion diagnostics: masked writes require a matching valid resident line in the selected way."
Block 42 "2426821494" "if (1)"
ANNOTATION: "Write assertion diagnostics: masked writes require a matching valid resident line in the selected way."
Block 43 "1714232348" "$error(\"Assertion failed: Cache write requires the matching resident line\n    at Cache.scala:93 assert(entryValid && hits(req.way), \\"Cache write requires the matching resident line\\")\n\");"
ANNOTATION: "Write assertion diagnostics: masked writes require a matching valid resident line in the selected way."
Block 45 "2426821494" "if (1)"
ANNOTATION: "Write assertion diagnostics: masked writes require a matching valid resident line in the selected way."
Block 46 "3284370139" "$fatal;"

CHECKSUM: "1581756980 4074018439"
INSTANCE: cache_tb.dut
ANNOTATION: "Requests are inactive during reset; this condition only gates assertion diagnostics."
Condition 1 "2427994462" "(_readData_1_T & ((~reset))) 1 -1" (2 "10")
ANNOTATION: "Reserved capacity invariant: count plus pending never exceeds responseDepth, so pending implies enq.ready."
Condition 2 "1313259708" "(((~reset)) & ( ~ (((~pending)) | _responses_io_enq_ready) )) 1 -1" (1 "01")
ANNOTATION: "Reserved capacity invariant: count plus pending never exceeds responseDepth, so pending implies enq.ready."
Condition 2 "1313259708" "(((~reset)) & ( ~ (((~pending)) | _responses_io_enq_ready) )) 1 -1" (3 "11")
ANNOTATION: "Reserved capacity invariant: count plus pending never exceeds responseDepth, so pending implies enq.ready."
Condition 4 "1917659308" "(((~pending)) | _responses_io_enq_ready) 1 -1" (1 "00")
ANNOTATION: "Only the five defined operation encodings are driven; invalid-opcode diagnostics are outside legal-operation coverage."
Condition 5 "521367662" "(_GEN_10 & (io_request_bits_op > 3'h4)) 1 -1" (1 "01")
ANNOTATION: "Only the five defined operation encodings are driven; invalid-opcode diagnostics are outside legal-operation coverage."
Condition 5 "521367662" "(_GEN_10 & (io_request_bits_op > 3'h4)) 1 -1" (3 "11")
ANNOTATION: "Addresses are line aligned; misalignment diagnostics are outside legal-operation coverage."
Condition 6 "3741468578" "(_GEN_10 & ((|io_request_bits_addr[5:0]))) 1 -1" (1 "01")
ANNOTATION: "Addresses are line aligned; misalignment diagnostics are outside legal-operation coverage."
Condition 6 "3741468578" "(_GEN_10 & ((|io_request_bits_addr[5:0]))) 1 -1" (3 "11")
ANNOTATION: "Valid tags remain unique: reset clears validity and legal fills cannot duplicate another valid way."
Condition 7 "3021235683" "(_GEN_10 & _GEN_12[1]) 1 -1" (1 "01")
ANNOTATION: "Valid tags remain unique: reset clears validity and legal fills cannot duplicate another valid way."
Condition 7 "3021235683" "(_GEN_10 & _GEN_12[1]) 1 -1" (3 "11")
ANNOTATION: "Legal fills target an absent line or its existing way; failed and reset-gated fill diagnostics are excluded."
Condition 8 "2233729839" "(_readData_1_T & _mask_T_2 & ((~reset)) & ( ~ (((~(|hits))) | _GEN_11[0]) )) 1 -1" (1 "0111")
ANNOTATION: "Legal fills target an absent line or its existing way; failed and reset-gated fill diagnostics are excluded."
Condition 8 "2233729839" "(_readData_1_T & _mask_T_2 & ((~reset)) & ( ~ (((~(|hits))) | _GEN_11[0]) )) 1 -1" (3 "1101")
ANNOTATION: "Legal fills target an absent line or its existing way; failed and reset-gated fill diagnostics are excluded."
Condition 8 "2233729839" "(_readData_1_T & _mask_T_2 & ((~reset)) & ( ~ (((~(|hits))) | _GEN_11[0]) )) 1 -1" (5 "1111")
ANNOTATION: "Legal writes match a valid line; failed and reset-gated write diagnostics are excluded."
Condition 11 "3664488518" "(_readData_1_T & _GEN_8 & ((~reset)) & ( ~ (entryValid & _GEN_11[0]) )) 1 -1" (1 "0111")
ANNOTATION: "Legal writes match a valid line; failed and reset-gated write diagnostics are excluded."
Condition 11 "3664488518" "(_readData_1_T & _GEN_8 & ((~reset)) & ( ~ (entryValid & _GEN_11[0]) )) 1 -1" (3 "1101")
ANNOTATION: "Legal writes match a valid line; failed and reset-gated write diagnostics are excluded."
Condition 11 "3664488518" "(_readData_1_T & _GEN_8 & ((~reset)) & ( ~ (entryValid & _GEN_11[0]) )) 1 -1" (5 "1111")
ANNOTATION: "A tag hit includes its valid bit. Selecting a hit cannot produce entryValid=0."
Condition 13 "2913363430" "(entryValid & _GEN_11[0]) 1 -1" (1 "01")

CHECKSUM: "784137509 320702925"
INSTANCE: cache_tb.dut.responses
ANNOTATION: "The Cache reserves queue capacity before accepting requests; enq.valid cannot occur with enq.ready=0."
Condition 6 "2766317690" "(io_enq_ready_0 & io_enq_valid) 1 -1" (1 "01")

CHECKSUM: "1581756980 2991418269"
INSTANCE: cache_tb.dut
ANNOTATION: "Line addresses are 64-byte aligned; low six address bits are zero."
Toggle io_request_bits_addr [5:0] "net io_request_bits_addr[43:0]"
ANNOTATION: "Line addresses are 64-byte aligned; low six address bits are zero."
Toggle io_response_bits_addr [5:0] "net io_response_bits_addr[43:0]"
ANNOTATION: "Line addresses are 64-byte aligned; low six address bits are zero."
Toggle result_addr [5:0] "reg result_addr[43:0]"

CHECKSUM: "784137509 896502883"
INSTANCE: cache_tb.dut.responses
ANNOTATION: "Line addresses are 64-byte aligned; low six address bits are zero."
Toggle io_enq_bits_addr [5:0] "net io_enq_bits_addr[43:0]"
ANNOTATION: "Line addresses are 64-byte aligned; low six address bits are zero."
Toggle io_deq_bits_addr [5:0] "net io_deq_bits_addr[43:0]"
ANNOTATION: "Line addresses are 64-byte aligned; low six address bits are zero. Queue packing maps address[5:0] to RAM output[17:12]."
Toggle _ram_ext_R0_data [17:12] "net _ram_ext_R0_data[571:0]"
