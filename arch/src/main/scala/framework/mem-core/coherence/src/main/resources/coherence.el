// Format Version: 2
// Reviewed for 2 CPUs, 4 MSHRs, 4 sets, 2 ways, 64-byte lines, 256-bit CHI data.

CHECKSUM: "2192584405 2792015601"
INSTANCE: coherence_tb.dut.cache
ANNOTATION: "Assertion-failure diagnostic only; assertion remains enabled: Unknown Cache operation"
Block 2 "422554957" "if (1)"
Block 3 "3963963684" "$error(\"Assertion failed: Unknown Cache operation\n    at Cache.scala:67 assert(req.op <= CacheOp.Invalidate.U, \\"Unknown Cache operation\\")\n\");"
Block 5 "422554957" "if (1)"
Block 6 "2766194732" "$fatal;"
ANNOTATION: "Assertion-failure diagnostic only; assertion remains enabled: Cache operations require a line-aligned address"
Block 10 "874237168" "if (1)"
Block 11 "1020309457" "$error(\"Assertion failed: Cache operations require a line-aligned address\n    at Cache.scala:68 assert(req.addr(p.offsetBits - 1, 0) === 0.U, \\"Cache operations require a line-aligned address\\")\n\");"
Block 13 "874237168" "if (1)"
Block 14 "1069368929" "$fatal;"
ANNOTATION: "Assertion-failure diagnostic only; assertion remains enabled: Duplicate Cache tag"
Block 18 "1759021865" "if (1)"
Block 19 "155016256" "$error(\"Assertion failed: Duplicate Cache tag\n    at Cache.scala:70 assert(PopCount(hits) <= 1.U, \\"Duplicate Cache tag\\")\n\");"
Block 21 "1759021865" "if (1)"
Block 22 "3067504005" "$fatal;"
ANNOTATION: "Assertion-failure diagnostic only; assertion remains enabled: Cache fill would create a duplicate tag"
Block 26 "1166789632" "if (1)"
Block 27 "1252711441" "$error(\"Assertion failed: Cache fill would create a duplicate tag\n    at Cache.scala:83 assert(!hits.orR || hits(req.way), \\"Cache fill would create a duplicate tag\\")\n\");"
Block 29 "1166789632" "if (1)"
Block 30 "3120397658" "$fatal;"
ANNOTATION: "Assertion-failure diagnostic only; assertion remains enabled: Cache write requires the matching resident line"
Block 34 "2426821494" "if (1)"
Block 35 "1714232348" "$error(\"Assertion failed: Cache write requires the matching resident line\n    at Cache.scala:93 assert(entryValid && hits(req.way), \\"Cache write requires the matching resident line\\")\n\");"
Block 37 "2426821494" "if (1)"
Block 38 "3284370139" "$fatal;"

CHECKSUM: "3544733189 4187106336"
INSTANCE: coherence_tb.dut.directory
ANNOTATION: "Assertion-failure diagnostic only; assertion remains enabled: Multiple transactions update one directory entry"
Block 2 "2664146107" "if (1)"
Block 3 "970576364" "$error(\"Assertion failed: Multiple transactions update one directory entry\n    at Directory.scala:42 assert(PopCount(writers) <= 1.U, \\"Multiple transactions update one directory entry\\")\n\");"
Block 5 "2664146107" "if (1)"
Block 6 "3472375403" "$fatal;"
Block 18 "1723217575" "if (1)"
Block 19 "63705195" "$error(\"Assertion failed: Multiple transactions update one directory entry\n    at Directory.scala:42 assert(PopCount(writers) <= 1.U, \\"Multiple transactions update one directory entry\\")\n\");"
Block 21 "1723217575" "if (1)"
Block 22 "4109238252" "$fatal;"
Block 34 "3919299312" "if (1)"
Block 35 "1635698152" "$error(\"Assertion failed: Multiple transactions update one directory entry\n    at Directory.scala:42 assert(PopCount(writers) <= 1.U, \\"Multiple transactions update one directory entry\\")\n\");"
Block 37 "3919299312" "if (1)"
Block 38 "2522819183" "$fatal;"
Block 50 "4085285969" "if (1)"
Block 51 "2054067127" "$error(\"Assertion failed: Multiple transactions update one directory entry\n    at Directory.scala:42 assert(PopCount(writers) <= 1.U, \\"Multiple transactions update one directory entry\\")\n\");"
Block 53 "4085285969" "if (1)"
Block 54 "2370796592" "$fatal;"
Block 66 "2091741190" "if (1)"
Block 67 "417089076" "$error(\"Assertion failed: Multiple transactions update one directory entry\n    at Directory.scala:42 assert(PopCount(writers) <= 1.U, \\"Multiple transactions update one directory entry\\")\n\");"
Block 69 "2091741190" "if (1)"
Block 70 "4026378675" "$fatal;"
Block 82 "2228281882" "if (1)"
Block 83 "583632819" "$error(\"Assertion failed: Multiple transactions update one directory entry\n    at Directory.scala:42 assert(PopCount(writers) <= 1.U, \\"Multiple transactions update one directory entry\\")\n\");"
Block 85 "2228281882" "if (1)"
Block 86 "3588778036" "$fatal;"
Block 98 "201135693" "if (1)"
Block 99 "1081823792" "$error(\"Assertion failed: Multiple transactions update one directory entry\n    at Directory.scala:42 assert(PopCount(writers) <= 1.U, \\"Multiple transactions update one directory entry\\")\n\");"
Block 101 "201135693" "if (1)"
Block 102 "3076177335" "$fatal;"
Block 114 "3662072233" "if (1)"
Block 115 "141097524" "$error(\"Assertion failed: Multiple transactions update one directory entry\n    at Directory.scala:42 assert(PopCount(writers) <= 1.U, \\"Multiple transactions update one directory entry\\")\n\");"
Block 117 "3662072233" "if (1)"
Block 118 "4282979763" "$fatal;"
ANNOTATION: "Assertion-failure diagnostic only; assertion remains enabled: Unique ownership requires exactly one CPU"
Block 10 "2881929055" "if (1)"
Block 11 "1532442423" "$error(\"Assertion failed: Unique ownership requires exactly one CPU\n    at Directory.scala:45 assert(!next.unique || PopCount(next.sharers) === 1.U, \\"Unique ownership requires exactly one CPU\\")\n\");"
Block 13 "2881929055" "if (1)"
Block 14 "1126102520" "$fatal;"
Block 26 "3405939620" "if (1)"
Block 27 "1221036824" "$error(\"Assertion failed: Unique ownership requires exactly one CPU\n    at Directory.scala:45 assert(!next.unique || PopCount(next.sharers) === 1.U, \\"Unique ownership requires exactly one CPU\\")\n\");"
Block 29 "3405939620" "if (1)"
Block 30 "1351502295" "$fatal;"
Block 42 "539207809" "if (1)"
Block 43 "3838492220" "$error(\"Assertion failed: Unique ownership requires exactly one CPU\n    at Directory.scala:45 assert(!next.unique || PopCount(next.sharers) === 1.U, \\"Unique ownership requires exactly one CPU\\")\n\");"
Block 45 "539207809" "if (1)"
Block 46 "4236475635" "$fatal;"
Block 58 "3010328178" "if (1)"
Block 59 "1376219673" "$error(\"Assertion failed: Unique ownership requires exactly one CPU\n    at Directory.scala:45 assert(!next.unique || PopCount(next.sharers) === 1.U, \\"Unique ownership requires exactly one CPU\\")\n\");"
Block 61 "3010328178" "if (1)"
Block 62 "1246671062" "$fatal;"
Block 74 "2992832868" "if (1)"
Block 75 "1267078095" "$error(\"Assertion failed: Unique ownership requires exactly one CPU\n    at Directory.scala:45 assert(!next.unique || PopCount(next.sharers) === 1.U, \\"Unique ownership requires exactly one CPU\\")\n\");"
Block 77 "2992832868" "if (1)"
Block 78 "1406128384" "$fatal;"
Block 90 "3031124909" "if (1)"
Block 91 "2669400392" "$error(\"Assertion failed: Unique ownership requires exactly one CPU\n    at Directory.scala:45 assert(!next.unique || PopCount(next.sharers) === 1.U, \\"Unique ownership requires exactly one CPU\\")\n\");"
Block 93 "3031124909" "if (1)"
Block 94 "2270303111" "$fatal;"
Block 106 "3770691673" "if (1)"
Block 107 "1999768136" "$error(\"Assertion failed: Unique ownership requires exactly one CPU\n    at Directory.scala:45 assert(!next.unique || PopCount(next.sharers) === 1.U, \\"Unique ownership requires exactly one CPU\\")\n\");"
Block 109 "3770691673" "if (1)"
Block 110 "1870383239" "$fatal;"
Block 122 "1829712518" "if (1)"
Block 123 "2412862675" "$error(\"Assertion failed: Unique ownership requires exactly one CPU\n    at Directory.scala:45 assert(!next.unique || PopCount(next.sharers) === 1.U, \\"Unique ownership requires exactly one CPU\\")\n\");"
Block 125 "1829712518" "if (1)"
Block 126 "2543361564" "$fatal;"

CHECKSUM: "2192584405 3677989228"
INSTANCE: coherence_tb.dut.cache
ANNOTATION: "RTL line 487: this condition only evaluates synthesis-excluded assertion diagnostics; it does not control functional state or data. Assertions remain enabled."
Condition 1 "2427994462" "(_readData_1_T & ((~reset))) 1 -1" (2 "10")
ANNOTATION: "RTL line 490: this condition only evaluates synthesis-excluded assertion diagnostics; it does not control functional state or data. Assertions remain enabled."
Condition 2 "521367662" "(_GEN_10 & (io_request_bits_op > 3'h4)) 1 -1" (1 "01")
Condition 2 "521367662" "(_GEN_10 & (io_request_bits_op > 3'h4)) 1 -1" (3 "11")
ANNOTATION: "RTL line 496: this condition only evaluates synthesis-excluded assertion diagnostics; it does not control functional state or data. Assertions remain enabled."
Condition 3 "3741468578" "(_GEN_10 & ((|io_request_bits_addr[5:0]))) 1 -1" (1 "01")
Condition 3 "3741468578" "(_GEN_10 & ((|io_request_bits_addr[5:0]))) 1 -1" (3 "11")
ANNOTATION: "RTL line 502: this condition only evaluates synthesis-excluded assertion diagnostics; it does not control functional state or data. Assertions remain enabled."
Condition 4 "3021235683" "(_GEN_10 & _GEN_12[1]) 1 -1" (1 "01")
Condition 4 "3021235683" "(_GEN_10 & _GEN_12[1]) 1 -1" (3 "11")
ANNOTATION: "RTL line 508: this condition only evaluates synthesis-excluded assertion diagnostics; it does not control functional state or data. Assertions remain enabled."
Condition 5 "2233729839" "(_readData_1_T & _mask_T_2 & ((~reset)) & ( ~ (((~(|hits))) | _GEN_11[0]) )) 1 -1" (1 "0111")
Condition 5 "2233729839" "(_readData_1_T & _mask_T_2 & ((~reset)) & ( ~ (((~(|hits))) | _GEN_11[0]) )) 1 -1" (3 "1101")
Condition 5 "2233729839" "(_readData_1_T & _mask_T_2 & ((~reset)) & ( ~ (((~(|hits))) | _GEN_11[0]) )) 1 -1" (5 "1111")
ANNOTATION: "RTL line 514: this condition only evaluates synthesis-excluded assertion diagnostics; it does not control functional state or data. Assertions remain enabled."
Condition 8 "3664488518" "(_readData_1_T & _GEN_8 & ((~reset)) & ( ~ (entryValid & _GEN_11[0]) )) 1 -1" (1 "0111")
Condition 8 "3664488518" "(_readData_1_T & _GEN_8 & ((~reset)) & ( ~ (entryValid & _GEN_11[0]) )) 1 -1" (3 "1101")
Condition 8 "3664488518" "(_readData_1_T & _GEN_8 & ((~reset)) & ( ~ (entryValid & _GEN_11[0]) )) 1 -1" (5 "1111")
Condition 10 "2913363430" "(entryValid & _GEN_11[0]) 1 -1" (1 "01")
ANNOTATION: "Home always supplies all eligible ways and never CacheOp.Read; Cache available and request ready stay 1 because responses are always consumed. Arbiter ready mirrors this. MSHR release PriorityEncoder returns slot 3 when no completed entry, so idle release cannot select slots 0..2. Four live entries own all four distinct set keys, so a full table necessarily conflicts with any legal set key."
Condition 103 "138758841" "(available & selected) 1 -1" (1 "01")
Condition 125 "2706906294" "(((~_GEN_1)) & io_request_bits_eligible[0]) 1 -1" (2 "10")
Condition 129 "307541151" "(((|io_request_bits_op)) | ((|hits)) | ((|io_request_bits_eligible))) 1 -1" (1 "000")
Condition 129 "307541151" "(((|io_request_bits_op)) | ((|hits)) | ((|io_request_bits_eligible))) 1 -1" (3 "010")
Condition 129 "307541151" "(((|io_request_bits_op)) | ((|hits)) | ((|io_request_bits_eligible))) 1 -1" (4 "100")
Condition 131 "1690393860" "(available & _GEN_7[io_request_bits_addr[7:6]]) 1 -1" (1 "01")
Condition 133 "748823815" "(((~(|io_request_bits_op))) | (io_request_bits_op == 3'b1)) 1 -1" (2 "01")
Condition 134 "3166737159" "(io_request_bits_op == 3'b1) 1 -1" (2 "1")
Condition 135 "1639134261" "((({1'b0, _responses_io_count} + {2'b0, pending}) < 3'h2) | _responses_io_deq_valid) 1 -1" (1 "00")
Condition 136 "3297793588" "(io_request_ready_0 & io_request_valid) 1 -1" (1 "01")

CHECKSUM: "3544733189 825818895"
INSTANCE: coherence_tb.dut.directory
ANNOTATION: "RTL line 1244: this condition only evaluates synthesis-excluded assertion diagnostics; it does not control functional state or data. Assertions remain enabled."
Condition 1 "883292196" "(((~reset)) & ((|_GEN_11[2:1]))) 1 -1" (1 "01")
Condition 1 "883292196" "(((~reset)) & ((|_GEN_11[2:1]))) 1 -1" (3 "11")
ANNOTATION: "RTL line 1250: this condition only evaluates synthesis-excluded assertion diagnostics; it does not control functional state or data. Assertions remain enabled."
Condition 2 "3675171543" "(((|_GEN_3)) & ((~reset)) & ( ~ (((~_next_T_6)) | (({1'b0, _next_T_13[0]} + {1'b0, _next_T_13[1]}) == 2'b1)) )) 1 -1" (1 "011")
Condition 2 "3675171543" "(((|_GEN_3)) & ((~reset)) & ( ~ (((~_next_T_6)) | (({1'b0, _next_T_13[0]} + {1'b0, _next_T_13[1]}) == 2'b1)) )) 1 -1" (2 "101")
Condition 2 "3675171543" "(((|_GEN_3)) & ((~reset)) & ( ~ (((~_next_T_6)) | (({1'b0, _next_T_13[0]} + {1'b0, _next_T_13[1]}) == 2'b1)) )) 1 -1" (4 "111")
Condition 4 "1001352585" "(((~_next_T_6)) | (({1'b0, _next_T_13[0]} + {1'b0, _next_T_13[1]}) == 2'b1)) 1 -1" (1 "00")
ANNOTATION: "RTL line 1257: this condition only evaluates synthesis-excluded assertion diagnostics; it does not control functional state or data. Assertions remain enabled."
Condition 6 "3541940063" "(((~reset)) & ((|_GEN_12[2:1]))) 1 -1" (1 "01")
Condition 6 "3541940063" "(((~reset)) & ((|_GEN_12[2:1]))) 1 -1" (3 "11")
ANNOTATION: "RTL line 1263: this condition only evaluates synthesis-excluded assertion diagnostics; it does not control functional state or data. Assertions remain enabled."
Condition 7 "165533737" "(((|_GEN_4)) & ((~reset)) & ( ~ (((~_next_T_20)) | (({1'b0, _next_T_27[0]} + {1'b0, _next_T_27[1]}) == 2'b1)) )) 1 -1" (1 "011")
Condition 7 "165533737" "(((|_GEN_4)) & ((~reset)) & ( ~ (((~_next_T_20)) | (({1'b0, _next_T_27[0]} + {1'b0, _next_T_27[1]}) == 2'b1)) )) 1 -1" (2 "101")
Condition 7 "165533737" "(((|_GEN_4)) & ((~reset)) & ( ~ (((~_next_T_20)) | (({1'b0, _next_T_27[0]} + {1'b0, _next_T_27[1]}) == 2'b1)) )) 1 -1" (4 "111")
Condition 9 "1578589649" "(((~_next_T_20)) | (({1'b0, _next_T_27[0]} + {1'b0, _next_T_27[1]}) == 2'b1)) 1 -1" (1 "00")
ANNOTATION: "RTL line 1270: this condition only evaluates synthesis-excluded assertion diagnostics; it does not control functional state or data. Assertions remain enabled."
Condition 11 "1285724628" "(((~reset)) & ((|_GEN_13[2:1]))) 1 -1" (1 "01")
Condition 11 "1285724628" "(((~reset)) & ((|_GEN_13[2:1]))) 1 -1" (3 "11")
ANNOTATION: "RTL line 1276: this condition only evaluates synthesis-excluded assertion diagnostics; it does not control functional state or data. Assertions remain enabled."
Condition 12 "736734174" "(((|_GEN_5)) & ((~reset)) & ( ~ (((~_next_T_34)) | (({1'b0, _next_T_41[0]} + {1'b0, _next_T_41[1]}) == 2'b1)) )) 1 -1" (1 "011")
Condition 12 "736734174" "(((|_GEN_5)) & ((~reset)) & ( ~ (((~_next_T_34)) | (({1'b0, _next_T_41[0]} + {1'b0, _next_T_41[1]}) == 2'b1)) )) 1 -1" (2 "101")
Condition 12 "736734174" "(((|_GEN_5)) & ((~reset)) & ( ~ (((~_next_T_34)) | (({1'b0, _next_T_41[0]} + {1'b0, _next_T_41[1]}) == 2'b1)) )) 1 -1" (4 "111")
Condition 14 "3256686846" "(((~_next_T_34)) | (({1'b0, _next_T_41[0]} + {1'b0, _next_T_41[1]}) == 2'b1)) 1 -1" (1 "00")
ANNOTATION: "RTL line 1283: this condition only evaluates synthesis-excluded assertion diagnostics; it does not control functional state or data. Assertions remain enabled."
Condition 16 "2132237777" "(((~reset)) & ((|_GEN_14[2:1]))) 1 -1" (1 "01")
Condition 16 "2132237777" "(((~reset)) & ((|_GEN_14[2:1]))) 1 -1" (3 "11")
ANNOTATION: "RTL line 1289: this condition only evaluates synthesis-excluded assertion diagnostics; it does not control functional state or data. Assertions remain enabled."
Condition 17 "1132272033" "(((|_GEN_6)) & ((~reset)) & ( ~ (((~_next_T_48)) | (({1'b0, _next_T_55[0]} + {1'b0, _next_T_55[1]}) == 2'b1)) )) 1 -1" (1 "011")
Condition 17 "1132272033" "(((|_GEN_6)) & ((~reset)) & ( ~ (((~_next_T_48)) | (({1'b0, _next_T_55[0]} + {1'b0, _next_T_55[1]}) == 2'b1)) )) 1 -1" (2 "101")
Condition 17 "1132272033" "(((|_GEN_6)) & ((~reset)) & ( ~ (((~_next_T_48)) | (({1'b0, _next_T_55[0]} + {1'b0, _next_T_55[1]}) == 2'b1)) )) 1 -1" (4 "111")
Condition 19 "2654681717" "(((~_next_T_48)) | (({1'b0, _next_T_55[0]} + {1'b0, _next_T_55[1]}) == 2'b1)) 1 -1" (1 "00")
ANNOTATION: "RTL line 1296: this condition only evaluates synthesis-excluded assertion diagnostics; it does not control functional state or data. Assertions remain enabled."
Condition 21 "3769135962" "(((~reset)) & ((|_GEN_15[2:1]))) 1 -1" (1 "01")
Condition 21 "3769135962" "(((~reset)) & ((|_GEN_15[2:1]))) 1 -1" (3 "11")
ANNOTATION: "RTL line 1302: this condition only evaluates synthesis-excluded assertion diagnostics; it does not control functional state or data. Assertions remain enabled."
Condition 22 "4085757909" "(((|_GEN_7)) & ((~reset)) & ( ~ (((~_next_T_62)) | (({1'b0, _next_T_69[0]} + {1'b0, _next_T_69[1]}) == 2'b1)) )) 1 -1" (1 "011")
Condition 22 "4085757909" "(((|_GEN_7)) & ((~reset)) & ( ~ (((~_next_T_62)) | (({1'b0, _next_T_69[0]} + {1'b0, _next_T_69[1]}) == 2'b1)) )) 1 -1" (2 "101")
Condition 22 "4085757909" "(((|_GEN_7)) & ((~reset)) & ( ~ (((~_next_T_62)) | (({1'b0, _next_T_69[0]} + {1'b0, _next_T_69[1]}) == 2'b1)) )) 1 -1" (4 "111")
Condition 24 "231885682" "(((~_next_T_62)) | (({1'b0, _next_T_69[0]} + {1'b0, _next_T_69[1]}) == 2'b1)) 1 -1" (1 "00")
ANNOTATION: "RTL line 1309: this condition only evaluates synthesis-excluded assertion diagnostics; it does not control functional state or data. Assertions remain enabled."
Condition 26 "118504481" "(((~reset)) & ((|_GEN_16[2:1]))) 1 -1" (1 "01")
Condition 26 "118504481" "(((~reset)) & ((|_GEN_16[2:1]))) 1 -1" (3 "11")
ANNOTATION: "RTL line 1315: this condition only evaluates synthesis-excluded assertion diagnostics; it does not control functional state or data. Assertions remain enabled."
Condition 27 "1607989346" "(((|_GEN_8)) & ((~reset)) & ( ~ (((~_next_T_76)) | (({1'b0, _next_T_83[0]} + {1'b0, _next_T_83[1]}) == 2'b1)) )) 1 -1" (1 "011")
Condition 27 "1607989346" "(((|_GEN_8)) & ((~reset)) & ( ~ (((~_next_T_76)) | (({1'b0, _next_T_83[0]} + {1'b0, _next_T_83[1]}) == 2'b1)) )) 1 -1" (2 "101")
Condition 27 "1607989346" "(((|_GEN_8)) & ((~reset)) & ( ~ (((~_next_T_76)) | (({1'b0, _next_T_83[0]} + {1'b0, _next_T_83[1]}) == 2'b1)) )) 1 -1" (4 "111")
Condition 29 "4059255789" "(((~_next_T_76)) | (({1'b0, _next_T_83[0]} + {1'b0, _next_T_83[1]}) == 2'b1)) 1 -1" (1 "00")
ANNOTATION: "RTL line 1322: this condition only evaluates synthesis-excluded assertion diagnostics; it does not control functional state or data. Assertions remain enabled."
Condition 31 "2561610410" "(((~reset)) & ((|_GEN_17[2:1]))) 1 -1" (1 "01")
Condition 31 "2561610410" "(((~reset)) & ((|_GEN_17[2:1]))) 1 -1" (3 "11")
ANNOTATION: "RTL line 1328: this condition only evaluates synthesis-excluded assertion diagnostics; it does not control functional state or data. Assertions remain enabled."
Condition 32 "613034987" "(((|_GEN_9)) & ((~reset)) & ( ~ (((~_next_T_90)) | (({1'b0, _next_T_97[0]} + {1'b0, _next_T_97[1]}) == 2'b1)) )) 1 -1" (1 "011")
Condition 32 "613034987" "(((|_GEN_9)) & ((~reset)) & ( ~ (((~_next_T_90)) | (({1'b0, _next_T_97[0]} + {1'b0, _next_T_97[1]}) == 2'b1)) )) 1 -1" (2 "101")
Condition 32 "613034987" "(((|_GEN_9)) & ((~reset)) & ( ~ (((~_next_T_90)) | (({1'b0, _next_T_97[0]} + {1'b0, _next_T_97[1]}) == 2'b1)) )) 1 -1" (4 "111")
Condition 34 "2676186116" "(((~_next_T_90)) | (({1'b0, _next_T_97[0]} + {1'b0, _next_T_97[1]}) == 2'b1)) 1 -1" (1 "00")
ANNOTATION: "RTL line 1335: this condition only evaluates synthesis-excluded assertion diagnostics; it does not control functional state or data. Assertions remain enabled."
Condition 36 "1326782190" "(((~reset)) & ((|_GEN_18[2:1]))) 1 -1" (1 "01")
Condition 36 "1326782190" "(((~reset)) & ((|_GEN_18[2:1]))) 1 -1" (3 "11")
ANNOTATION: "RTL line 1341: this condition only evaluates synthesis-excluded assertion diagnostics; it does not control functional state or data. Assertions remain enabled."
Condition 37 "1158646046" "(((|_GEN_10)) & ((~reset)) & ( ~ (((~_next_T_104)) | (({1'b0, _next_T_111[0]} + {1'b0, _next_T_111[1]}) == 2'b1)) )) 1 -1" (1 "011")
Condition 37 "1158646046" "(((|_GEN_10)) & ((~reset)) & ( ~ (((~_next_T_104)) | (({1'b0, _next_T_111[0]} + {1'b0, _next_T_111[1]}) == 2'b1)) )) 1 -1" (2 "101")
Condition 37 "1158646046" "(((|_GEN_10)) & ((~reset)) & ( ~ (((~_next_T_104)) | (({1'b0, _next_T_111[0]} + {1'b0, _next_T_111[1]}) == 2'b1)) )) 1 -1" (4 "111")
Condition 39 "3801203728" "(((~_next_T_104)) | (({1'b0, _next_T_111[0]} + {1'b0, _next_T_111[1]}) == 2'b1)) 1 -1" (1 "00")

CHECKSUM: "1648893030 439674848"
INSTANCE: coherence_tb.dut.cache.responses
ANNOTATION: "Cache responses inherit aligned resident addresses and MSHR IDs0..3; metadata only dirty bit0. All ways are eligible, so response available is always true."
Toggle io_enq_bits_id [7:2] "net io_enq_bits_id[7:0]"
Toggle io_enq_bits_available "net io_enq_bits_available"
Toggle io_enq_bits_addr [5:0] "net io_enq_bits_addr[43:0]"
Toggle io_enq_bits_metadata [3:1] "net io_enq_bits_metadata[3:0]"
Toggle io_deq_bits_id [7:2] "net io_deq_bits_id[7:0]"
Toggle io_deq_bits_available "net io_deq_bits_available"
Toggle io_deq_bits_addr [5:0] "net io_deq_bits_addr[43:0]"
Toggle io_deq_bits_metadata [3:1] "net io_deq_bits_metadata[3:0]"
Toggle _ram_ext_R0_data [7:2] "net _ram_ext_R0_data[571:0]"
Toggle _ram_ext_R0_data [9] "net _ram_ext_R0_data[571:0]"
Toggle _ram_ext_R0_data [17:12] "net _ram_ext_R0_data[571:0]"
Toggle _ram_ext_R0_data [571:569] "net _ram_ext_R0_data[571:0]"
ANNOTATION: "Coherence permanently accepts Cache responses. For a depth2 Queue with consumer ready=1, occupancy is 0 or1 and maybe_full equals !ptr_match; full is unreachable. Cache pending+responses is <=2 and deq.fire releases any full reservation, hence Cache request ready and cacheQueue consumer ready stay1."
Toggle io_count [1] "net io_count[1:0]"

CHECKSUM: "2773406112 83107964"
INSTANCE: coherence_tb.dut.datQueue
ANNOTATION: "Home emits full-line CompData with fixed HomeID64 and zero optional attributes; states I/SC/UC, errors OK/NDERR, DBID0..3, DataID0/2, requester IDs1..2."
Toggle io_enq_bits_tgtId [6:2] "net io_enq_bits_tgtId[6:0]"
Toggle io_enq_bits_resp [2] "net io_enq_bits_resp[2:0]"
Toggle io_enq_bits_dbid [15:2] "net io_enq_bits_dbid[15:0]"
Toggle io_enq_bits_dataId [0] "net io_enq_bits_dataId[1:0]"
Toggle io_deq_bits_qos [3:0] "net io_deq_bits_qos[3:0]"
Toggle io_deq_bits_tgtId [6:2] "net io_deq_bits_tgtId[6:0]"
Toggle io_deq_bits_srcId [6:0] "net io_deq_bits_srcId[6:0]"
Toggle io_deq_bits_homeNid [6:0] "net io_deq_bits_homeNid[6:0]"
Toggle io_deq_bits_opcode [3:0] "net io_deq_bits_opcode[3:0]"
Toggle io_deq_bits_resp [2] "net io_deq_bits_resp[2:0]"
Toggle io_deq_bits_dataSource [7:0] "net io_deq_bits_dataSource[7:0]"
Toggle io_deq_bits_dataPull "net io_deq_bits_dataPull"
Toggle io_deq_bits_cBusy [2:0] "net io_deq_bits_cBusy[2:0]"
Toggle io_deq_bits_dbid [15:2] "net io_deq_bits_dbid[15:0]"
Toggle io_deq_bits_ccid [1:0] "net io_deq_bits_ccid[1:0]"
Toggle io_deq_bits_dataId [0] "net io_deq_bits_dataId[1:0]"
Toggle io_deq_bits_cacheLineId [5:0] "net io_deq_bits_cacheLineId[5:0]"
Toggle io_deq_bits_tagOp [1:0] "net io_deq_bits_tagOp[1:0]"
Toggle io_deq_bits_tag [7:0] "net io_deq_bits_tag[7:0]"
Toggle io_deq_bits_tagUpdate [1:0] "net io_deq_bits_tagUpdate[1:0]"
Toggle io_deq_bits_traceTag "net io_deq_bits_traceTag"
Toggle io_deq_bits_copyAtHome "net io_deq_bits_copyAtHome"
Toggle io_deq_bits_numDat [1:0] "net io_deq_bits_numDat[1:0]"
Toggle io_deq_bits_replicate "net io_deq_bits_replicate"
Toggle _ram_ext_R0_data [3:0] "net _ram_ext_R0_data[388:0]"
Toggle _ram_ext_R0_data [17:6] "net _ram_ext_R0_data[388:0]"
Toggle _ram_ext_R0_data [40:30] "net _ram_ext_R0_data[388:0]"
Toggle _ram_ext_R0_data [57:45] "net _ram_ext_R0_data[388:0]"
Toggle _ram_ext_R0_data [76:60] "net _ram_ext_R0_data[388:0]"
Toggle _ram_ext_R0_data [100:78] "net _ram_ext_R0_data[388:0]"

CHECKSUM: "2258146608 144556243"
INSTANCE: coherence_tb.dut.cacheQueue
ANNOTATION: "Cache requests use full-line byte mask, all2 ways eligible, aligned64B addresses, MSHR IDs0..3; only metadata bit0 stores dirty."
Toggle io_enq_bits_id [7:2] "net io_enq_bits_id[7:0]"
Toggle io_enq_bits_addr [5:0] "net io_enq_bits_addr[43:0]"
Toggle io_enq_bits_metadata [3:1] "net io_enq_bits_metadata[3:0]"
Toggle io_deq_bits_id [7:2] "net io_deq_bits_id[7:0]"
Toggle io_deq_bits_addr [5:0] "net io_deq_bits_addr[43:0]"
Toggle io_deq_bits_mask [63:0] "net io_deq_bits_mask[63:0]"
Toggle io_deq_bits_metadata [3:1] "net io_deq_bits_metadata[3:0]"
Toggle io_deq_bits_eligible [1:0] "net io_deq_bits_eligible[1:0]"
Toggle _ram_ext_R0_data [7:2] "net _ram_ext_R0_data[637:0]"
Toggle _ram_ext_R0_data [16:11] "net _ram_ext_R0_data[637:0]"
Toggle _ram_ext_R0_data [631:568] "net _ram_ext_R0_data[637:0]"
Toggle _ram_ext_R0_data [637:633] "net _ram_ext_R0_data[637:0]"
ANNOTATION: "Coherence permanently accepts Cache responses. For a depth2 Queue with consumer ready=1, occupancy is 0 or1 and maybe_full equals !ptr_match; full is unreachable. Cache pending+responses is <=2 and deq.fire releases any full reservation, hence Cache request ready and cacheQueue consumer ready stay1."
Toggle io_enq_ready "net io_enq_ready"
Toggle io_deq_ready "net io_deq_ready"
Toggle io_enq_ready_0 "net io_enq_ready_0"

CHECKSUM: "3086174696 4028402977"
INSTANCE: coherence_tb.dut.memQueue
ANNOTATION: "Backing-memory requests use full64B mask, aligned64B addresses and MSHR IDs0..3."
Toggle io_enq_bits_id [11:2] "net io_enq_bits_id[11:0]"
Toggle io_enq_bits_addr [5:0] "net io_enq_bits_addr[43:0]"
Toggle io_deq_bits_id [11:2] "net io_deq_bits_id[11:0]"
Toggle io_deq_bits_addr [5:0] "net io_deq_bits_addr[43:0]"
Toggle io_deq_bits_mask [63:0] "net io_deq_bits_mask[63:0]"
Toggle _ram_ext_R0_data [17:2] "net _ram_ext_R0_data[632:0]"
Toggle _ram_ext_R0_data [632:569] "net _ram_ext_R0_data[632:0]"

CHECKSUM: "1385267114 4218119170"
INSTANCE: coherence_tb.dut.rspQueue
ANNOTATION: "Home emits Comp/CompDBIDResp only, no extended attributes; HomeID=64, requester IDs1..2, MSHR DBID0..3; response errors are OK or NDERR."
Toggle io_enq_bits_tgtId [6:2] "net io_enq_bits_tgtId[6:0]"
Toggle io_enq_bits_opcode [4:1] "net io_enq_bits_opcode[4:0]"
Toggle io_enq_bits_dbid [11:2] "net io_enq_bits_dbid[11:0]"
Toggle io_deq_bits_qos [3:0] "net io_deq_bits_qos[3:0]"
Toggle io_deq_bits_tgtId [6:2] "net io_deq_bits_tgtId[6:0]"
Toggle io_deq_bits_srcId [6:0] "net io_deq_bits_srcId[6:0]"
Toggle io_deq_bits_opcode [4:1] "net io_deq_bits_opcode[4:0]"
Toggle io_deq_bits_resp [2:0] "net io_deq_bits_resp[2:0]"
Toggle io_deq_bits_fwdState [2:0] "net io_deq_bits_fwdState[2:0]"
Toggle io_deq_bits_cBusy [2:0] "net io_deq_bits_cBusy[2:0]"
Toggle io_deq_bits_dbid [11:2] "net io_deq_bits_dbid[11:0]"
Toggle io_deq_bits_pCrdType [3:0] "net io_deq_bits_pCrdType[3:0]"
Toggle io_deq_bits_tagOp [1:0] "net io_deq_bits_tagOp[1:0]"
Toggle io_deq_bits_traceTag "net io_deq_bits_traceTag"
Toggle io_deq_bits_cacheLineId [5:0] "net io_deq_bits_cacheLineId[5:0]"
Toggle _ram_ext_R0_data [3:0] "net _ram_ext_R0_data[70:0]"
Toggle _ram_ext_R0_data [17:6] "net _ram_ext_R0_data[70:0]"
Toggle _ram_ext_R0_data [34:31] "net _ram_ext_R0_data[70:0]"
Toggle _ram_ext_R0_data [45:37] "net _ram_ext_R0_data[70:0]"
Toggle _ram_ext_R0_data [70:48] "net _ram_ext_R0_data[70:0]"

CHECKSUM: "2904587974 4013187535"
INSTANCE: coherence_tb.dut.snpQueue
ANNOTATION: "Home emits SnpNotSharedDirty/Unique/CleanInvalid, no forwarding/PAS/trace; DoNotGoToSD=1, HomeID64, MSHR IDs0..3, targets1..2, 64B line address shifted by3."
Toggle io_enq_bits_targetNode [6:2] "net io_enq_bits_targetNode[6:0]"
Toggle io_enq_bits_flit_txnId [11:2] "net io_enq_bits_flit_txnId[11:0]"
Toggle io_enq_bits_flit_opcode [4] "net io_enq_bits_flit_opcode[4:0]"
Toggle io_enq_bits_flit_addr [2:0] "net io_enq_bits_flit_addr[40:0]"
Toggle io_deq_bits_targetNode [6:2] "net io_deq_bits_targetNode[6:0]"
Toggle io_deq_bits_flit_qos [3:0] "net io_deq_bits_flit_qos[3:0]"
Toggle io_deq_bits_flit_srcId [6:0] "net io_deq_bits_flit_srcId[6:0]"
Toggle io_deq_bits_flit_txnId [11:2] "net io_deq_bits_flit_txnId[11:0]"
Toggle io_deq_bits_flit_fwdNid [6:0] "net io_deq_bits_flit_fwdNid[6:0]"
Toggle io_deq_bits_flit_fwdTxnId [11:0] "net io_deq_bits_flit_fwdTxnId[11:0]"
Toggle io_deq_bits_flit_opcode [4] "net io_deq_bits_flit_opcode[4:0]"
Toggle io_deq_bits_flit_addr [2:0] "net io_deq_bits_flit_addr[40:0]"
Toggle io_deq_bits_flit_pas [2:0] "net io_deq_bits_flit_pas[2:0]"
Toggle io_deq_bits_flit_doNotGoToSd "net io_deq_bits_flit_doNotGoToSd"
Toggle io_deq_bits_flit_traceTag "net io_deq_bits_flit_traceTag"
Toggle _ram_ext_R0_data [17:2] "net _ram_ext_R0_data[100:0]"
Toggle _ram_ext_R0_data [48:20] "net _ram_ext_R0_data[100:0]"
Toggle _ram_ext_R0_data [56:53] "net _ram_ext_R0_data[100:0]"
Toggle _ram_ext_R0_data [98:95] "net _ram_ext_R0_data[100:0]"
Toggle _ram_ext_R0_data [100] "net _ram_ext_R0_data[100:0]"

CHECKSUM: "1853676028 3335661445"
INSTANCE: coherence_tb.dut.cacheArb
ANNOTATION: "Coherence.scala assigns each arbiter input its fixed MSHR slot0..3, aligned64B address, full-line mask, only dirty metadata bit0. Outgoing RSP is Comp/CompDBIDResp, DAT states I/SC/UC with OK/NDERR and DataID0/2, SNP opcodes4/7/9; requester targets are1..2 in this exact parameter configuration."
Toggle io_in_0_bits_addr [5:0] "net io_in_0_bits_addr[43:0]"
Toggle io_in_0_bits_metadata [3:1] "net io_in_0_bits_metadata[3:0]"
Toggle io_in_1_bits_addr [5:0] "net io_in_1_bits_addr[43:0]"
Toggle io_in_1_bits_metadata [3:1] "net io_in_1_bits_metadata[3:0]"
Toggle io_in_2_bits_addr [5:0] "net io_in_2_bits_addr[43:0]"
Toggle io_in_2_bits_metadata [3:1] "net io_in_2_bits_metadata[3:0]"
Toggle io_in_3_bits_addr [5:0] "net io_in_3_bits_addr[43:0]"
Toggle io_in_3_bits_metadata [3:1] "net io_in_3_bits_metadata[3:0]"
Toggle io_out_bits_id [7:2] "net io_out_bits_id[7:0]"
Toggle io_out_bits_addr [5:0] "net io_out_bits_addr[43:0]"
Toggle io_out_bits_metadata [3:1] "net io_out_bits_metadata[3:0]"
ANNOTATION: "Exact packed copies of source-proven fields: fixed per-input MSHR ID literal, requester IDs 1..2, 64 B alignment, dirty-only metadata, RSP Comp/CompDBIDResp, SNP opcodes 4/7/9, or DAT DataID 0/2. No selection/payload bits excluded."
Toggle _GEN "net [3:0][7:0]_GEN"
Toggle _GEN_2 [0][5:0] "net [3:0][43:0]_GEN_2"
Toggle _GEN_2 [1][5:0] "net [3:0][43:0]_GEN_2"
Toggle _GEN_2 [2][5:0] "net [3:0][43:0]_GEN_2"
Toggle _GEN_2 [3][5:0] "net [3:0][43:0]_GEN_2"
Toggle _GEN_5 [0][3:1] "net [3:0][3:0]_GEN_5"
Toggle _GEN_5 [1][3:1] "net [3:0][3:0]_GEN_5"
Toggle _GEN_5 [2][3:1] "net [3:0][3:0]_GEN_5"
Toggle _GEN_5 [3][3:1] "net [3:0][3:0]_GEN_5"
ANNOTATION: "Exact 2-agent/4-MSHR Home profile: accepted access attributes are asserted constants, full-line aligned addresses/masks, fixed HomeID 64, requester IDs 1..2, MSHR IDs 0..3, only dirty metadata bit 0; outgoing attributes are assigned constants. Cache response consumer ready is permanently 1, so Cache request ready is 1 and depth-2 response count bit 1 is zero."
Toggle io_out_ready "net io_out_ready"

CHECKSUM: "3130163852 2143613990"
INSTANCE: coherence_tb.dut.memArb
ANNOTATION: "Coherence.scala assigns each arbiter input its fixed MSHR slot0..3, aligned64B address, full-line mask, only dirty metadata bit0. Outgoing RSP is Comp/CompDBIDResp, DAT states I/SC/UC with OK/NDERR and DataID0/2, SNP opcodes4/7/9; requester targets are1..2 in this exact parameter configuration."
Toggle io_in_0_bits_addr [5:0] "net io_in_0_bits_addr[43:0]"
Toggle io_in_1_bits_addr [5:0] "net io_in_1_bits_addr[43:0]"
Toggle io_in_2_bits_addr [5:0] "net io_in_2_bits_addr[43:0]"
Toggle io_in_3_bits_addr [5:0] "net io_in_3_bits_addr[43:0]"
Toggle io_out_bits_id [11:2] "net io_out_bits_id[11:0]"
Toggle io_out_bits_addr [5:0] "net io_out_bits_addr[43:0]"
ANNOTATION: "Exact packed copies of source-proven fields: fixed per-input MSHR ID literal, requester IDs 1..2, 64 B alignment, dirty-only metadata, RSP Comp/CompDBIDResp, SNP opcodes 4/7/9, or DAT DataID 0/2. No selection/payload bits excluded."
Toggle _GEN "net [3:0][11:0]_GEN"
Toggle _GEN_1 [0][5:0] "net [3:0][43:0]_GEN_1"
Toggle _GEN_1 [1][5:0] "net [3:0][43:0]_GEN_1"
Toggle _GEN_1 [2][5:0] "net [3:0][43:0]_GEN_1"
Toggle _GEN_1 [3][5:0] "net [3:0][43:0]_GEN_1"

CHECKSUM: "3716118738 1796417120"
INSTANCE: coherence_tb.dut.snpArb
ANNOTATION: "Coherence.scala assigns each arbiter input its fixed MSHR slot0..3, aligned64B address, full-line mask, only dirty metadata bit0. Outgoing RSP is Comp/CompDBIDResp, DAT states I/SC/UC with OK/NDERR and DataID0/2, SNP opcodes4/7/9; requester targets are1..2 in this exact parameter configuration."
Toggle io_in_0_bits_targetNode [6:2] "net io_in_0_bits_targetNode[6:0]"
Toggle io_in_0_bits_flit_opcode [4] "net io_in_0_bits_flit_opcode[4:0]"
Toggle io_in_0_bits_flit_addr [2:0] "net io_in_0_bits_flit_addr[40:0]"
Toggle io_in_1_bits_targetNode [6:2] "net io_in_1_bits_targetNode[6:0]"
Toggle io_in_1_bits_flit_opcode [4] "net io_in_1_bits_flit_opcode[4:0]"
Toggle io_in_1_bits_flit_addr [2:0] "net io_in_1_bits_flit_addr[40:0]"
Toggle io_in_2_bits_targetNode [6:2] "net io_in_2_bits_targetNode[6:0]"
Toggle io_in_2_bits_flit_opcode [4] "net io_in_2_bits_flit_opcode[4:0]"
Toggle io_in_2_bits_flit_addr [2:0] "net io_in_2_bits_flit_addr[40:0]"
Toggle io_in_3_bits_targetNode [6:2] "net io_in_3_bits_targetNode[6:0]"
Toggle io_in_3_bits_flit_opcode [4] "net io_in_3_bits_flit_opcode[4:0]"
Toggle io_in_3_bits_flit_addr [2:0] "net io_in_3_bits_flit_addr[40:0]"
Toggle io_out_bits_targetNode [6:2] "net io_out_bits_targetNode[6:0]"
Toggle io_out_bits_flit_txnId [11:2] "net io_out_bits_flit_txnId[11:0]"
Toggle io_out_bits_flit_opcode [4] "net io_out_bits_flit_opcode[4:0]"
Toggle io_out_bits_flit_addr [2:0] "net io_out_bits_flit_addr[40:0]"
ANNOTATION: "Exact packed copies of source-proven fields: fixed per-input MSHR ID literal, requester IDs 1..2, 64 B alignment, dirty-only metadata, RSP Comp/CompDBIDResp, SNP opcodes 4/7/9, or DAT DataID 0/2. No selection/payload bits excluded."
Toggle _GEN "net [3:0][11:0]_GEN"
Toggle _GEN_1 [0][6:2] "net [3:0][6:0]_GEN_1"
Toggle _GEN_1 [1][6:2] "net [3:0][6:0]_GEN_1"
Toggle _GEN_1 [2][6:2] "net [3:0][6:0]_GEN_1"
Toggle _GEN_1 [3][6:2] "net [3:0][6:0]_GEN_1"
Toggle _GEN_2 [0][4:4] "net [3:0][4:0]_GEN_2"
Toggle _GEN_2 [1][4:4] "net [3:0][4:0]_GEN_2"
Toggle _GEN_2 [2][4:4] "net [3:0][4:0]_GEN_2"
Toggle _GEN_2 [3][4:4] "net [3:0][4:0]_GEN_2"
Toggle _GEN_3 [0][2:0] "net [3:0][40:0]_GEN_3"
Toggle _GEN_3 [1][2:0] "net [3:0][40:0]_GEN_3"
Toggle _GEN_3 [2][2:0] "net [3:0][40:0]_GEN_3"
Toggle _GEN_3 [3][2:0] "net [3:0][40:0]_GEN_3"

CHECKSUM: "2004716196 3529883256"
INSTANCE: coherence_tb.dut.rspArb
ANNOTATION: "Coherence.scala assigns each arbiter input its fixed MSHR slot0..3, aligned64B address, full-line mask, only dirty metadata bit0. Outgoing RSP is Comp/CompDBIDResp, DAT states I/SC/UC with OK/NDERR and DataID0/2, SNP opcodes4/7/9; requester targets are1..2 in this exact parameter configuration."
Toggle io_in_0_bits_tgtId [6:2] "net io_in_0_bits_tgtId[6:0]"
Toggle io_in_0_bits_opcode [4:1] "net io_in_0_bits_opcode[4:0]"
Toggle io_in_1_bits_tgtId [6:2] "net io_in_1_bits_tgtId[6:0]"
Toggle io_in_1_bits_opcode [4:1] "net io_in_1_bits_opcode[4:0]"
Toggle io_in_2_bits_tgtId [6:2] "net io_in_2_bits_tgtId[6:0]"
Toggle io_in_2_bits_opcode [4:1] "net io_in_2_bits_opcode[4:0]"
Toggle io_in_3_bits_tgtId [6:2] "net io_in_3_bits_tgtId[6:0]"
Toggle io_in_3_bits_opcode [4:1] "net io_in_3_bits_opcode[4:0]"
Toggle io_out_bits_tgtId [6:2] "net io_out_bits_tgtId[6:0]"
Toggle io_out_bits_opcode [4:1] "net io_out_bits_opcode[4:0]"
Toggle io_out_bits_dbid [11:2] "net io_out_bits_dbid[11:0]"
ANNOTATION: "Exact packed copies of source-proven fields: fixed per-input MSHR ID literal, requester IDs 1..2, 64 B alignment, dirty-only metadata, RSP Comp/CompDBIDResp, SNP opcodes 4/7/9, or DAT DataID 0/2. No selection/payload bits excluded."
Toggle _GEN "net [3:0][11:0]_GEN"
Toggle _GEN_1 [0][6:2] "net [3:0][6:0]_GEN_1"
Toggle _GEN_1 [1][6:2] "net [3:0][6:0]_GEN_1"
Toggle _GEN_1 [2][6:2] "net [3:0][6:0]_GEN_1"
Toggle _GEN_1 [3][6:2] "net [3:0][6:0]_GEN_1"
Toggle _GEN_3 [0][4:1] "net [3:0][4:0]_GEN_3"
Toggle _GEN_3 [1][4:1] "net [3:0][4:0]_GEN_3"
Toggle _GEN_3 [2][4:1] "net [3:0][4:0]_GEN_3"
Toggle _GEN_3 [3][4:1] "net [3:0][4:0]_GEN_3"

CHECKSUM: "3496838959 3135397854"
INSTANCE: coherence_tb.dut.datArb
ANNOTATION: "Coherence.scala assigns each arbiter input its fixed MSHR slot0..3, aligned64B address, full-line mask, only dirty metadata bit0. Outgoing RSP is Comp/CompDBIDResp, DAT states I/SC/UC with OK/NDERR and DataID0/2, SNP opcodes4/7/9; requester targets are1..2 in this exact parameter configuration."
Toggle io_in_0_bits_tgtId [6:2] "net io_in_0_bits_tgtId[6:0]"
Toggle io_in_0_bits_resp [2] "net io_in_0_bits_resp[2:0]"
Toggle io_in_0_bits_dataId [0] "net io_in_0_bits_dataId[1:0]"
Toggle io_in_1_bits_tgtId [6:2] "net io_in_1_bits_tgtId[6:0]"
Toggle io_in_1_bits_resp [2] "net io_in_1_bits_resp[2:0]"
Toggle io_in_1_bits_dataId [0] "net io_in_1_bits_dataId[1:0]"
Toggle io_in_2_bits_tgtId [6:2] "net io_in_2_bits_tgtId[6:0]"
Toggle io_in_2_bits_resp [2] "net io_in_2_bits_resp[2:0]"
Toggle io_in_2_bits_dataId [0] "net io_in_2_bits_dataId[1:0]"
Toggle io_in_3_bits_tgtId [6:2] "net io_in_3_bits_tgtId[6:0]"
Toggle io_in_3_bits_resp [2] "net io_in_3_bits_resp[2:0]"
Toggle io_in_3_bits_dataId [0] "net io_in_3_bits_dataId[1:0]"
Toggle io_out_bits_tgtId [6:2] "net io_out_bits_tgtId[6:0]"
Toggle io_out_bits_resp [2] "net io_out_bits_resp[2:0]"
Toggle io_out_bits_dbid [15:2] "net io_out_bits_dbid[15:0]"
Toggle io_out_bits_dataId [0] "net io_out_bits_dataId[1:0]"
ANNOTATION: "Exact packed copies of source-proven fields: fixed per-input MSHR ID literal, requester IDs 1..2, 64 B alignment, dirty-only metadata, RSP Comp/CompDBIDResp, SNP opcodes 4/7/9, or DAT DataID 0/2. No selection/payload bits excluded."
Toggle _GEN "net [3:0][15:0]_GEN"
Toggle _GEN_1 [0][6:2] "net [3:0][6:0]_GEN_1"
Toggle _GEN_1 [1][6:2] "net [3:0][6:0]_GEN_1"
Toggle _GEN_1 [2][6:2] "net [3:0][6:0]_GEN_1"
Toggle _GEN_1 [3][6:2] "net [3:0][6:0]_GEN_1"
ANNOTATION: "Packed copies of CompData state I/SC/UC have Resp[2]=0; 256-bit DataID is 0 or 2, so bit 0 is zero."
Toggle _GEN_4 [0][2:2] "net [3:0][2:0]_GEN_4"
Toggle _GEN_4 [1][2:2] "net [3:0][2:0]_GEN_4"
Toggle _GEN_4 [2][2:2] "net [3:0][2:0]_GEN_4"
Toggle _GEN_4 [3][2:2] "net [3:0][2:0]_GEN_4"
Toggle _GEN_5 [0][0:0] "net [3:0][1:0]_GEN_5"
Toggle _GEN_5 [1][0:0] "net [3:0][1:0]_GEN_5"
Toggle _GEN_5 [2][0:0] "net [3:0][1:0]_GEN_5"
Toggle _GEN_5 [3][0:0] "net [3:0][1:0]_GEN_5"

CHECKSUM: "1648893030 3605751416"
INSTANCE: coherence_tb.dut.cache.responses
ANNOTATION: "Coherence permanently accepts Cache responses. For a depth2 Queue with consumer ready=1, occupancy is 0 or1 and maybe_full equals !ptr_match; full is unreachable. Cache pending+responses is <=2 and deq.fire releases any full reservation, hence Cache request ready and cacheQueue consumer ready stay1."
Condition 4 "1750755365" "(ptr_match & ((~maybe_full))) 1 -1" (1 "01")
Condition 4 "1750755365" "(ptr_match & ((~maybe_full))) 1 -1" (2 "10")
Condition 6 "2157053168" "(maybe_full & ptr_match) 1 -1" (3 "11")

CHECKSUM: "2258146608 405613642"
INSTANCE: coherence_tb.dut.cacheQueue
ANNOTATION: "Coherence permanently accepts Cache responses. For a depth2 Queue with consumer ready=1, occupancy is 0 or1 and maybe_full equals !ptr_match; full is unreachable. Cache pending+responses is <=2 and deq.fire releases any full reservation, hence Cache request ready and cacheQueue consumer ready stay1."
Condition 1 "3657552453" "(io_deq_ready & ((~empty))) 1 -1" (1 "01")
Condition 5 "1750755365" "(ptr_match & ((~maybe_full))) 1 -1" (1 "01")
Condition 5 "1750755365" "(ptr_match & ((~maybe_full))) 1 -1" (2 "10")
Condition 6 "2766317690" "(io_enq_ready_0 & io_enq_valid) 1 -1" (1 "01")
Condition 7 "1458076629" "(io_deq_ready | ( ~ (ptr_match & maybe_full) )) 1 -1" (1 "00")
Condition 7 "1458076629" "(io_deq_ready | ( ~ (ptr_match & maybe_full) )) 1 -1" (2 "01")
Condition 7 "1458076629" "(io_deq_ready | ( ~ (ptr_match & maybe_full) )) 1 -1" (3 "10")
Condition 9 "2534730988" "(ptr_match & maybe_full) 1 -1" (3 "11")

CHECKSUM: "2192584405 2636709535"
INSTANCE: coherence_tb.dut.cache
ANNOTATION: "Home eligible=2'b11; shifting by the one-bit selected way yields 11 or 01, so bit 0 is permanently one."
Toggle _order_T [0] "net _order_T[1:0]"
ANNOTATION: "Exact 2-agent/4-MSHR Home profile: accepted access attributes are asserted constants, full-line aligned addresses/masks, fixed HomeID 64, requester IDs 1..2, MSHR IDs 0..3, only dirty metadata bit 0; outgoing attributes are assigned constants. Cache response consumer ready is permanently 1, so Cache request ready is 1 and depth-2 response count bit 1 is zero."
Toggle io_request_ready "net io_request_ready"
Toggle io_request_bits_id [7:2] "net io_request_bits_id[7:0]"
Toggle io_request_bits_addr [5:0] "net io_request_bits_addr[43:0]"
Toggle io_request_bits_mask [63:0] "net io_request_bits_mask[63:0]"
Toggle io_request_bits_metadata [3:1] "net io_request_bits_metadata[3:0]"
Toggle io_request_bits_eligible [1:0] "net io_request_bits_eligible[1:0]"
Toggle io_response_bits_id [7:2] "net io_response_bits_id[7:0]"
Toggle io_response_bits_available "net io_response_bits_available"
Toggle io_response_bits_addr [5:0] "net io_response_bits_addr[43:0]"
Toggle io_response_bits_metadata [3:1] "net io_response_bits_metadata[3:0]"
Toggle _responses_io_count [1] "net _responses_io_count[1:0]"
Toggle metadata_0_0 [3:1] "reg metadata_0_0[3:0]"
Toggle metadata_0_1 [3:1] "reg metadata_0_1[3:0]"
Toggle metadata_0_2 [3:1] "reg metadata_0_2[3:0]"
Toggle metadata_0_3 [3:1] "reg metadata_0_3[3:0]"
Toggle metadata_1_0 [3:1] "reg metadata_1_0[3:0]"
Toggle metadata_1_1 [3:1] "reg metadata_1_1[3:0]"
Toggle metadata_1_2 [3:1] "reg metadata_1_2[3:0]"
Toggle metadata_1_3 [3:1] "reg metadata_1_3[3:0]"
Toggle result_id [7:2] "reg result_id[7:0]"
Toggle result_available "reg result_available"
Toggle result_addr [5:0] "reg result_addr[43:0]"
Toggle result_metadata [3:1] "reg result_metadata[3:0]"
Toggle available "net available"
Toggle io_request_ready_0 "net io_request_ready_0"

CHECKSUM: "1853676028 2074822484"
INSTANCE: coherence_tb.dut.cacheArb
ANNOTATION: "Home always supplies all eligible ways and never CacheOp.Read; Cache available and request ready stay 1 because responses are always consumed. Arbiter ready mirrors this. MSHR release PriorityEncoder returns slot 3 when no completed entry, so idle release cannot select slots 0..2. Four live entries own all four distinct set keys, so a full table necessarily conflicts with any legal set key."
Condition 1 "2548440157" "(io_out_ready & io_out_valid_0) 1 -1" (1 "01")
Condition 16 "2900323864" "(((~_ctrl_T_2)) & io_out_ready) 1 -1" (2 "10")
Condition 17 "492347255" "((ctrl_validMask_grantMask_1 | ((~_ctrl_T_3))) & io_out_ready) 1 -1" (2 "10")
Condition 19 "183447282" "(((((~ctrl_validMask_1)) & ((~ctrl_validMask_grantMask_lastGrant[1]))) | ((~_ctrl_T_4))) & io_out_ready) 1 -1" (2 "10")
Condition 22 "3475292372" "(((((~_ctrl_T_1)) & ctrl_validMask_grantMask_3) | ( ~ (_ctrl_T_4 | io_in_2_valid) )) & io_out_ready) 1 -1" (2 "10")
