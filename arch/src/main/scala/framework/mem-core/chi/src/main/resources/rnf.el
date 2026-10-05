// Format Version: 2
// Exact reviewed objects; covered invalid-payload transitions remain measured.

CHECKSUM: "2670869491 2733910728"
INSTANCE: rnf_tb.dut
ANNOTATION: "Assertion-failure diagnostic only; assertion remains enabled: Unknown CPU RSP transaction bank"
Block 2 "1350818871" "if (1)"
Block 3 "650053590" "$error(\"Assertion failed: Unknown CPU RSP transaction bank\n    at BankedChiCache.scala:122 when(io.chi.rxRsp.valid)(assert(io.chi.rxRsp.bits.txnId < banks.U, \\"Unknown CPU RSP transaction bank\\"))\n\");"
Block 5 "1350818871" "if (1)"
Block 6 "145681284" "$fatal;"
ANNOTATION: "Assertion-failure diagnostic only; assertion remains enabled: Unknown CPU DAT transaction bank"
Block 10 "689049961" "if (1)"
Block 11 "2258865167" "$error(\"Assertion failed: Unknown CPU DAT transaction bank\n    at BankedChiCache.scala:123 when(io.chi.rxDat.valid)(assert(io.chi.rxDat.bits.txnId < banks.U, \\"Unknown CPU DAT transaction bank\\"))\n\");"
Block 13 "689049961" "if (1)"
Block 14 "3452538458" "$fatal;"

CHECKSUM: "3740104159 727212703"
INSTANCE: rnf_tb.dut.caches_0
ANNOTATION: "Assertion-failure diagnostic only; assertion remains enabled: Cache client requires naturally aligned accesses"
Block 2 "845982127" "if (1)"
Block 3 "408617132" "$error(\"Assertion failed: Cache client requires naturally aligned accesses\n    at ChiCache.scala:84 assert(Mux(io.access.bits.atomicWord, io.access.bits.addr(1, 0) === 0.U, io.access.bits.addr(2, 0) === 0.U), \\"Cache client requires naturally aligned accesses\\")\n\");"
Block 5 "845982127" "if (1)"
Block 6 "1103606821" "$fatal;"
ANNOTATION: "Assertion-failure diagnostic only; assertion remains enabled: Word width requires AMO, LR or SC"
Block 10 "1099732415" "if (1)"
Block 11 "2262014216" "$error(\"Assertion failed: Word width requires AMO, LR or SC\n    at ChiCache.scala:86 assert(io.access.bits.atomic >= CacheAtomic.Swap.U && io.access.bits.atomic <= CacheAtomic.SC.U, \\"Word width requires AMO, LR or SC\\")\n\");"
Block 13 "1099732415" "if (1)"
Block 14 "3139505398" "$fatal;"
ANNOTATION: "Assertion-failure diagnostic only; assertion remains enabled: Unknown CPU atomic operation"
Block 18 "1658233315" "if (1)"
Block 19 "510928977" "$error(\"Assertion failed: Unknown CPU atomic operation\n    at ChiCache.scala:88 assert(io.access.bits.atomic <= CacheAtomic.Fence.U, \\"Unknown CPU atomic operation\\")\n\");"
Block 21 "1658233315" "if (1)"
Block 22 "2317883602" "$fatal;"
ANNOTATION: "Assertion-failure diagnostic only; assertion remains enabled: Atomic operation requires the full operand marker"
Block 26 "99230478" "if (1)"
Block 27 "1331101736" "$error(\"Assertion failed: Atomic operation requires the full operand marker\n    at ChiCache.scala:90 assert(!io.access.bits.write && io.access.bits.mask.andR, \\"Atomic operation requires the full operand marker\\")\n\");"
Block 29 "99230478" "if (1)"
Block 30 "3218197046" "$fatal;"
ANNOTATION: "Assertion-failure diagnostic only; assertion remains enabled: Unexpected eviction completion"
Block 34 "2881000889" "if (1)"
Block 35 "3786393072" "$error(\"Assertion failed: Unexpected eviction completion\n    at ChiCache.scala:181 assert(\n\");"
Block 37 "2881000889" "if (1)"
Block 38 "3049408533" "$fatal;"
ANNOTATION: "Assertion-failure diagnostic only; assertion remains enabled: WriteBackFull requires CompDBIDResp"
Block 42 "3420348596" "if (1)"
Block 43 "244731808" "$error(\"Assertion failed: WriteBackFull requires CompDBIDResp\n    at ChiCache.scala:188 assert(r.opcode === Opcode.CompDBIDResp.U, \\"WriteBackFull requires CompDBIDResp\\")\n\");"
Block 45 "3420348596" "if (1)"
Block 46 "2533988698" "$fatal;"
ANNOTATION: "Assertion-failure diagnostic only; assertion remains enabled: Evict requires Comp"
Block 50 "1187914140" "if (1)"
Block 51 "149188837" "$error(\"Assertion failed: Evict requires Comp\n    at ChiCache.scala:203 assert(r.opcode === Opcode.Comp.U, \\"Evict requires Comp\\")\n\");"
Block 53 "1187914140" "if (1)"
Block 54 "97073557" "$fatal;"
ANNOTATION: "Assertion-failure diagnostic only; assertion remains enabled: Unexpected cache fill"
Block 58 "2124224208" "if (1)"
Block 59 "4158536239" "$error(\"Assertion failed: Unexpected cache fill\n    at ChiCache.scala:215 assert(\n\");"
Block 61 "2124224208" "if (1)"
Block 62 "4260059692" "$fatal;"
ANNOTATION: "Assertion-failure diagnostic only; assertion remains enabled: Invalid fill DataID"
Block 66 "3801034810" "if (1)"
Block 67 "1945151808" "$error(\"Assertion failed: Invalid fill DataID\n    at ChiCache.scala:221 assert(\n\");"
Block 69 "3801034810" "if (1)"
Block 70 "3273890626" "$fatal;"
ANNOTATION: "Assertion-failure diagnostic only; assertion remains enabled: Duplicate fill DataID"
Block 74 "648845981" "if (1)"
Block 75 "2059032942" "$error(\"Assertion failed: Duplicate fill DataID\n    at ChiCache.scala:225 assert(!(fillSeen & UIntToOH(b, p.beatsPerLine)).orR, \\"Duplicate fill DataID\\")\n\");"
Block 77 "648845981" "if (1)"
Block 78 "2047516166" "$fatal;"
ANNOTATION: "Assertion-failure diagnostic only; assertion remains enabled: Fill DBID exceeds completion ID width"
Block 82 "1032892522" "if (1)"
Block 83 "3960963073" "$error(\"Assertion failed: Fill DBID exceeds completion ID width\n    at ChiCache.scala:229 assert(d.dbid < (BigInt(1) << p.dbIdBits).U, \\"Fill DBID exceeds completion ID width\\")\n\");"
Block 85 "1032892522" "if (1)"
Block 86 "1496444768" "$fatal;"
ANNOTATION: "Assertion-failure diagnostic only; assertion remains enabled: Inconsistent multi-beat fill metadata"
Block 90 "791243170" "if (1)"
Block 91 "1850172745" "$error(\"Assertion failed: Inconsistent multi-beat fill metadata\n    at ChiCache.scala:231 assert(d.dbid === fillDbid && d.resp === fillPermission, \\"Inconsistent multi-beat fill metadata\\")\n\");"
Block 93 "791243170" "if (1)"
Block 94 "2947389197" "$fatal;"
ANNOTATION: "Assertion-failure diagnostic only; assertion remains enabled: Home granted unexpected cache permission"
Block 98 "3641802372" "if (1)"
Block 99 "4109766014" "$error(\"Assertion failed: Home granted unexpected cache permission\n    at ChiCache.scala:242 assert(\n\");"
Block 101 "3641802372" "if (1)"
Block 102 "1549775202" "$fatal;"
ANNOTATION: "Assertion-failure diagnostic only; assertion remains enabled: Unsupported cache snoop"
Block 106 "1953754182" "if (1)"
Block 107 "2973842345" "$error(\"Assertion failed: Unsupported cache snoop\n    at ChiCache.scala:270 assert(\n\");"
Block 109 "1953754182" "if (1)"
Block 110 "3571450114" "$fatal;"
ANNOTATION: "Assertion-failure diagnostic only; assertion remains enabled: Dirty cache line without Unique ownership"
Block 114 "3062387834" "if (1)"
Block 115 "3330320875" "$error(\"Assertion failed: Dirty cache line without Unique ownership\n    at ChiCache.scala:332 assert(!dirty(i) || (valid(i) && writable(i)), \\"Dirty cache line without Unique ownership\\")\n\");"
Block 117 "3062387834" "if (1)"
Block 118 "2903973297" "$fatal;"
Block 130 "2907427957" "if (1)"
Block 131 "4273248313" "$error(\"Assertion failed: Dirty cache line without Unique ownership\n    at ChiCache.scala:332 assert(!dirty(i) || (valid(i) && writable(i)), \\"Dirty cache line without Unique ownership\\")\n\");"
Block 133 "2907427957" "if (1)"
Block 134 "2502110307" "$fatal;"
Block 146 "1565793642" "if (1)"
Block 147 "2966284053" "$error(\"Assertion failed: Dirty cache line without Unique ownership\n    at ChiCache.scala:332 assert(!dirty(i) || (valid(i) && writable(i)), \\"Dirty cache line without Unique ownership\\")\n\");"
Block 149 "1565793642" "if (1)"
Block 150 "3680129871" "$fatal;"
Block 162 "1184341349" "if (1)"
Block 163 "2298074823" "$error(\"Assertion failed: Dirty cache line without Unique ownership\n    at ChiCache.scala:332 assert(!dirty(i) || (valid(i) && writable(i)), \\"Dirty cache line without Unique ownership\\")\n\");"
Block 165 "1184341349" "if (1)"
Block 166 "3815662237" "$fatal;"
ANNOTATION: "Assertion-failure diagnostic only; assertion remains enabled: Invalid cache line has write permission"
Block 122 "3097160385" "if (1)"
Block 123 "5359524" "$error(\"Assertion failed: Invalid cache line has write permission\n    at ChiCache.scala:333 assert(!writable(i) || valid(i), \\"Invalid cache line has write permission\\")\n\");"
Block 125 "3097160385" "if (1)"
Block 126 "966852955" "$fatal;"
Block 138 "1070241886" "if (1)"
Block 139 "3206203969" "$error(\"Assertion failed: Invalid cache line has write permission\n    at ChiCache.scala:333 assert(!writable(i) || valid(i), \\"Invalid cache line has write permission\\")\n\");"
Block 141 "1070241886" "if (1)"
Block 142 "2263485630" "$fatal;"
Block 154 "3109502912" "if (1)"
Block 155 "1796083762" "$error(\"Assertion failed: Invalid cache line has write permission\n    at ChiCache.scala:333 assert(!writable(i) || valid(i), \\"Invalid cache line has write permission\\")\n\");"
Block 157 "3109502912" "if (1)"
Block 158 "1392432845" "$fatal;"
Block 170 "1040665951" "if (1)"
Block 171 "3561299415" "$error(\"Assertion failed: Invalid cache line has write permission\n    at ChiCache.scala:333 assert(!writable(i) || valid(i), \\"Invalid cache line has write permission\\")\n\");"
Block 173 "1040665951" "if (1)"
Block 174 "3988118312" "$fatal;"

CHECKSUM: "1855582600 4210254037"
INSTANCE: rnf_tb.dut.caches_1
ANNOTATION: "Assertion-failure diagnostic only; assertion remains enabled: Cache client requires naturally aligned accesses"
Block 2 "845982127" "if (1)"
Block 3 "408617132" "$error(\"Assertion failed: Cache client requires naturally aligned accesses\n    at ChiCache.scala:84 assert(Mux(io.access.bits.atomicWord, io.access.bits.addr(1, 0) === 0.U, io.access.bits.addr(2, 0) === 0.U), \\"Cache client requires naturally aligned accesses\\")\n\");"
Block 5 "845982127" "if (1)"
Block 6 "1103606821" "$fatal;"
ANNOTATION: "Assertion-failure diagnostic only; assertion remains enabled: Word width requires AMO, LR or SC"
Block 10 "1099732415" "if (1)"
Block 11 "2262014216" "$error(\"Assertion failed: Word width requires AMO, LR or SC\n    at ChiCache.scala:86 assert(io.access.bits.atomic >= CacheAtomic.Swap.U && io.access.bits.atomic <= CacheAtomic.SC.U, \\"Word width requires AMO, LR or SC\\")\n\");"
Block 13 "1099732415" "if (1)"
Block 14 "3139505398" "$fatal;"
ANNOTATION: "Assertion-failure diagnostic only; assertion remains enabled: Unknown CPU atomic operation"
Block 18 "1658233315" "if (1)"
Block 19 "510928977" "$error(\"Assertion failed: Unknown CPU atomic operation\n    at ChiCache.scala:88 assert(io.access.bits.atomic <= CacheAtomic.Fence.U, \\"Unknown CPU atomic operation\\")\n\");"
Block 21 "1658233315" "if (1)"
Block 22 "2317883602" "$fatal;"
ANNOTATION: "Assertion-failure diagnostic only; assertion remains enabled: Atomic operation requires the full operand marker"
Block 26 "99230478" "if (1)"
Block 27 "1331101736" "$error(\"Assertion failed: Atomic operation requires the full operand marker\n    at ChiCache.scala:90 assert(!io.access.bits.write && io.access.bits.mask.andR, \\"Atomic operation requires the full operand marker\\")\n\");"
Block 29 "99230478" "if (1)"
Block 30 "3218197046" "$fatal;"
ANNOTATION: "Assertion-failure diagnostic only; assertion remains enabled: Unexpected eviction completion"
Block 34 "548357375" "if (1)"
Block 35 "412125852" "$error(\"Assertion failed: Unexpected eviction completion\n    at ChiCache.scala:181 assert(\n\");"
Block 37 "548357375" "if (1)"
Block 38 "1291650937" "$fatal;"
ANNOTATION: "Assertion-failure diagnostic only; assertion remains enabled: WriteBackFull requires CompDBIDResp"
Block 42 "3420348596" "if (1)"
Block 43 "244731808" "$error(\"Assertion failed: WriteBackFull requires CompDBIDResp\n    at ChiCache.scala:188 assert(r.opcode === Opcode.CompDBIDResp.U, \\"WriteBackFull requires CompDBIDResp\\")\n\");"
Block 45 "3420348596" "if (1)"
Block 46 "2533988698" "$fatal;"
ANNOTATION: "Assertion-failure diagnostic only; assertion remains enabled: Evict requires Comp"
Block 50 "1187914140" "if (1)"
Block 51 "149188837" "$error(\"Assertion failed: Evict requires Comp\n    at ChiCache.scala:203 assert(r.opcode === Opcode.Comp.U, \\"Evict requires Comp\\")\n\");"
Block 53 "1187914140" "if (1)"
Block 54 "97073557" "$fatal;"
ANNOTATION: "Assertion-failure diagnostic only; assertion remains enabled: Unexpected cache fill"
Block 58 "3025585016" "if (1)"
Block 59 "3183937630" "$error(\"Assertion failed: Unexpected cache fill\n    at ChiCache.scala:215 assert(\n\");"
Block 61 "3025585016" "if (1)"
Block 62 "3086116957" "$fatal;"
ANNOTATION: "Assertion-failure diagnostic only; assertion remains enabled: Invalid fill DataID"
Block 66 "3801034810" "if (1)"
Block 67 "1945151808" "$error(\"Assertion failed: Invalid fill DataID\n    at ChiCache.scala:221 assert(\n\");"
Block 69 "3801034810" "if (1)"
Block 70 "3273890626" "$fatal;"
ANNOTATION: "Assertion-failure diagnostic only; assertion remains enabled: Duplicate fill DataID"
Block 74 "648845981" "if (1)"
Block 75 "2059032942" "$error(\"Assertion failed: Duplicate fill DataID\n    at ChiCache.scala:225 assert(!(fillSeen & UIntToOH(b, p.beatsPerLine)).orR, \\"Duplicate fill DataID\\")\n\");"
Block 77 "648845981" "if (1)"
Block 78 "2047516166" "$fatal;"
ANNOTATION: "Assertion-failure diagnostic only; assertion remains enabled: Fill DBID exceeds completion ID width"
Block 82 "1032892522" "if (1)"
Block 83 "3960963073" "$error(\"Assertion failed: Fill DBID exceeds completion ID width\n    at ChiCache.scala:229 assert(d.dbid < (BigInt(1) << p.dbIdBits).U, \\"Fill DBID exceeds completion ID width\\")\n\");"
Block 85 "1032892522" "if (1)"
Block 86 "1496444768" "$fatal;"
ANNOTATION: "Assertion-failure diagnostic only; assertion remains enabled: Inconsistent multi-beat fill metadata"
Block 90 "791243170" "if (1)"
Block 91 "1850172745" "$error(\"Assertion failed: Inconsistent multi-beat fill metadata\n    at ChiCache.scala:231 assert(d.dbid === fillDbid && d.resp === fillPermission, \\"Inconsistent multi-beat fill metadata\\")\n\");"
Block 93 "791243170" "if (1)"
Block 94 "2947389197" "$fatal;"
ANNOTATION: "Assertion-failure diagnostic only; assertion remains enabled: Home granted unexpected cache permission"
Block 98 "3641802372" "if (1)"
Block 99 "4109766014" "$error(\"Assertion failed: Home granted unexpected cache permission\n    at ChiCache.scala:242 assert(\n\");"
Block 101 "3641802372" "if (1)"
Block 102 "1549775202" "$fatal;"
ANNOTATION: "Assertion-failure diagnostic only; assertion remains enabled: Unsupported cache snoop"
Block 106 "1953754182" "if (1)"
Block 107 "2973842345" "$error(\"Assertion failed: Unsupported cache snoop\n    at ChiCache.scala:270 assert(\n\");"
Block 109 "1953754182" "if (1)"
Block 110 "3571450114" "$fatal;"
ANNOTATION: "Assertion-failure diagnostic only; assertion remains enabled: Dirty cache line without Unique ownership"
Block 114 "3062387834" "if (1)"
Block 115 "3330320875" "$error(\"Assertion failed: Dirty cache line without Unique ownership\n    at ChiCache.scala:332 assert(!dirty(i) || (valid(i) && writable(i)), \\"Dirty cache line without Unique ownership\\")\n\");"
Block 117 "3062387834" "if (1)"
Block 118 "2903973297" "$fatal;"
Block 130 "2907427957" "if (1)"
Block 131 "4273248313" "$error(\"Assertion failed: Dirty cache line without Unique ownership\n    at ChiCache.scala:332 assert(!dirty(i) || (valid(i) && writable(i)), \\"Dirty cache line without Unique ownership\\")\n\");"
Block 133 "2907427957" "if (1)"
Block 134 "2502110307" "$fatal;"
Block 146 "1565793642" "if (1)"
Block 147 "2966284053" "$error(\"Assertion failed: Dirty cache line without Unique ownership\n    at ChiCache.scala:332 assert(!dirty(i) || (valid(i) && writable(i)), \\"Dirty cache line without Unique ownership\\")\n\");"
Block 149 "1565793642" "if (1)"
Block 150 "3680129871" "$fatal;"
Block 162 "1184341349" "if (1)"
Block 163 "2298074823" "$error(\"Assertion failed: Dirty cache line without Unique ownership\n    at ChiCache.scala:332 assert(!dirty(i) || (valid(i) && writable(i)), \\"Dirty cache line without Unique ownership\\")\n\");"
Block 165 "1184341349" "if (1)"
Block 166 "3815662237" "$fatal;"
ANNOTATION: "Assertion-failure diagnostic only; assertion remains enabled: Invalid cache line has write permission"
Block 122 "3097160385" "if (1)"
Block 123 "5359524" "$error(\"Assertion failed: Invalid cache line has write permission\n    at ChiCache.scala:333 assert(!writable(i) || valid(i), \\"Invalid cache line has write permission\\")\n\");"
Block 125 "3097160385" "if (1)"
Block 126 "966852955" "$fatal;"
Block 138 "1070241886" "if (1)"
Block 139 "3206203969" "$error(\"Assertion failed: Invalid cache line has write permission\n    at ChiCache.scala:333 assert(!writable(i) || valid(i), \\"Invalid cache line has write permission\\")\n\");"
Block 141 "1070241886" "if (1)"
Block 142 "2263485630" "$fatal;"
Block 154 "3109502912" "if (1)"
Block 155 "1796083762" "$error(\"Assertion failed: Invalid cache line has write permission\n    at ChiCache.scala:333 assert(!writable(i) || valid(i), \\"Invalid cache line has write permission\\")\n\");"
Block 157 "3109502912" "if (1)"
Block 158 "1392432845" "$fatal;"
Block 170 "1040665951" "if (1)"
Block 171 "3561299415" "$error(\"Assertion failed: Invalid cache line has write permission\n    at ChiCache.scala:333 assert(!writable(i) || valid(i), \\"Invalid cache line has write permission\\")\n\");"
Block 173 "1040665951" "if (1)"
Block 174 "3988118312" "$fatal;"

CHECKSUM: "2670869491 4075113301"
INSTANCE: rnf_tb.dut
ANNOTATION: "CacheAccess asserts at least 4B alignment; word atomics permit bit2 to vary and it remains measured."
Toggle 0to1 io_access_bits_addr [0] "net io_access_bits_addr[43:0]"
ANNOTATION: "ChiCache.scala request/txRsp/txDat zero-initialization and explicit assignments: NodeID1, HomeID64, one Home; two banks use request TxnID0/1; 64B lines and 256b DataID0/2; only ReadNotSharedDirty/ReadUnique/Evict/WriteBackFull, CompAck/SnpResp and SnpRespData/CopyBackWriteData. Dataless snoop responses have only I/SC; dirty snoops return DAT. Optional attributes remain zero. Full payload, Home DBID and snoop TxnID are not excluded."
Toggle io_chi_req_bits_qos [3:0] "net io_chi_req_bits_qos[3:0]"
Toggle io_chi_req_bits_tgtId [6:0] "net io_chi_req_bits_tgtId[6:0]"
Toggle io_chi_req_bits_srcId [6:0] "net io_chi_req_bits_srcId[6:0]"
Toggle io_chi_req_bits_txnId [11:1] "net io_chi_req_bits_txnId[11:0]"
Toggle io_chi_req_bits_returnNid [6:0] "net io_chi_req_bits_returnNid[6:0]"
Toggle io_chi_req_bits_stashNidValidEndian "net io_chi_req_bits_stashNidValidEndian"
Toggle io_chi_req_bits_returnTxnId [11:0] "net io_chi_req_bits_returnTxnId[11:0]"
Toggle io_chi_req_bits_opcode [6] "net io_chi_req_bits_opcode[6:0]"
Toggle io_chi_req_bits_multiReq "net io_chi_req_bits_multiReq"
Toggle io_chi_req_bits_size [5:0] "net io_chi_req_bits_size[5:0]"
Toggle io_chi_req_bits_addr [5:0] "net io_chi_req_bits_addr[43:0]"
Toggle io_chi_req_bits_pas [2:0] "net io_chi_req_bits_pas[2:0]"
Toggle io_chi_req_bits_likelyShared "net io_chi_req_bits_likelyShared"
Toggle io_chi_req_bits_allowRetry "net io_chi_req_bits_allowRetry"
Toggle io_chi_req_bits_order [1:0] "net io_chi_req_bits_order[1:0]"
Toggle io_chi_req_bits_pCrdType [3:0] "net io_chi_req_bits_pCrdType[3:0]"
Toggle io_chi_req_bits_memAttr [3:0] "net io_chi_req_bits_memAttr[3:0]"
Toggle io_chi_req_bits_snpAttr "net io_chi_req_bits_snpAttr"
Toggle io_chi_req_bits_lpid [7:0] "net io_chi_req_bits_lpid[7:0]"
Toggle io_chi_req_bits_exclSnoopMe "net io_chi_req_bits_exclSnoopMe"
Toggle io_chi_req_bits_tagOp [1:0] "net io_chi_req_bits_tagOp[1:0]"
Toggle io_chi_req_bits_traceTag "net io_chi_req_bits_traceTag"
Toggle io_chi_txRsp_bits_qos [3:0] "net io_chi_txRsp_bits_qos[3:0]"
Toggle io_chi_txRsp_bits_tgtId [6:0] "net io_chi_txRsp_bits_tgtId[6:0]"
Toggle io_chi_txRsp_bits_srcId [6:0] "net io_chi_txRsp_bits_srcId[6:0]"
Toggle io_chi_txRsp_bits_opcode [4:2] "net io_chi_txRsp_bits_opcode[4:0]"
Toggle io_chi_txRsp_bits_respErr [1:0] "net io_chi_txRsp_bits_respErr[1:0]"
Toggle io_chi_txRsp_bits_resp [2:1] "net io_chi_txRsp_bits_resp[2:0]"
Toggle io_chi_txRsp_bits_fwdState [2:0] "net io_chi_txRsp_bits_fwdState[2:0]"
Toggle io_chi_txRsp_bits_cBusy [2:0] "net io_chi_txRsp_bits_cBusy[2:0]"
Toggle io_chi_txRsp_bits_dbid [11:0] "net io_chi_txRsp_bits_dbid[11:0]"
Toggle io_chi_txRsp_bits_pCrdType [3:0] "net io_chi_txRsp_bits_pCrdType[3:0]"
Toggle io_chi_txRsp_bits_tagOp [1:0] "net io_chi_txRsp_bits_tagOp[1:0]"
Toggle io_chi_txRsp_bits_traceTag "net io_chi_txRsp_bits_traceTag"
Toggle io_chi_txRsp_bits_cacheLineId [5:0] "net io_chi_txRsp_bits_cacheLineId[5:0]"
Toggle io_chi_txDat_bits_qos [3:0] "net io_chi_txDat_bits_qos[3:0]"
Toggle io_chi_txDat_bits_tgtId [6:0] "net io_chi_txDat_bits_tgtId[6:0]"
Toggle io_chi_txDat_bits_srcId [6:0] "net io_chi_txDat_bits_srcId[6:0]"
Toggle io_chi_txDat_bits_homeNid [6:0] "net io_chi_txDat_bits_homeNid[6:0]"
Toggle io_chi_txDat_bits_opcode [3:2] "net io_chi_txDat_bits_opcode[3:0]"
Toggle io_chi_txDat_bits_respErr [1:0] "net io_chi_txDat_bits_respErr[1:0]"
Toggle io_chi_txDat_bits_dataSource [7:0] "net io_chi_txDat_bits_dataSource[7:0]"
Toggle io_chi_txDat_bits_dataPull "net io_chi_txDat_bits_dataPull"
Toggle io_chi_txDat_bits_cBusy [2:0] "net io_chi_txDat_bits_cBusy[2:0]"
Toggle io_chi_txDat_bits_dbid [15:0] "net io_chi_txDat_bits_dbid[15:0]"
Toggle io_chi_txDat_bits_ccid [1:0] "net io_chi_txDat_bits_ccid[1:0]"
Toggle io_chi_txDat_bits_cacheLineId [5:0] "net io_chi_txDat_bits_cacheLineId[5:0]"
Toggle io_chi_txDat_bits_tagOp [1:0] "net io_chi_txDat_bits_tagOp[1:0]"
Toggle io_chi_txDat_bits_tag [7:0] "net io_chi_txDat_bits_tag[7:0]"
Toggle io_chi_txDat_bits_tagUpdate [1:0] "net io_chi_txDat_bits_tagUpdate[1:0]"
Toggle io_chi_txDat_bits_traceTag "net io_chi_txDat_bits_traceTag"
Toggle io_chi_txDat_bits_copyAtHome "net io_chi_txDat_bits_copyAtHome"
Toggle io_chi_txDat_bits_numDat [1:0] "net io_chi_txDat_bits_numDat[1:0]"
Toggle io_chi_txDat_bits_replicate "net io_chi_txDat_bits_replicate"
Toggle _datArb_io_out_bits_tgtId [6:0] "net _datArb_io_out_bits_tgtId[6:0]"
Toggle _datArb_io_out_bits_opcode [3:2] "net _datArb_io_out_bits_opcode[3:0]"
Toggle _datArb_io_out_bits_dataId [0] "net _datArb_io_out_bits_dataId[1:0]"
Toggle _rspArb_io_out_bits_tgtId [6:0] "net _rspArb_io_out_bits_tgtId[6:0]"
Toggle _rspArb_io_out_bits_opcode [4:2] "net _rspArb_io_out_bits_opcode[4:0]"
Toggle _rspArb_io_out_bits_resp [2:1] "net _rspArb_io_out_bits_resp[2:0]"
Toggle _reqArb_io_out_bits_txnId [11:1] "net _reqArb_io_out_bits_txnId[11:0]"
Toggle _reqArb_io_out_bits_opcode [6] "net _reqArb_io_out_bits_opcode[6:0]"
Toggle _reqArb_io_out_bits_addr [5:0] "net _reqArb_io_out_bits_addr[43:0]"
Toggle _caches_1_io_chi_req_bits_opcode [6] "net _caches_1_io_chi_req_bits_opcode[6:0]"
Toggle _caches_1_io_chi_txRsp_bits_tgtId [6:0] "net _caches_1_io_chi_txRsp_bits_tgtId[6:0]"
Toggle _caches_1_io_chi_txRsp_bits_opcode [4:2] "net _caches_1_io_chi_txRsp_bits_opcode[4:0]"
Toggle _caches_1_io_chi_txRsp_bits_resp [2:1] "net _caches_1_io_chi_txRsp_bits_resp[2:0]"
Toggle _caches_1_io_chi_txDat_bits_tgtId [6:0] "net _caches_1_io_chi_txDat_bits_tgtId[6:0]"
Toggle _caches_1_io_chi_txDat_bits_opcode [3:2] "net _caches_1_io_chi_txDat_bits_opcode[3:0]"
Toggle _caches_1_io_chi_txDat_bits_dataId [0] "net _caches_1_io_chi_txDat_bits_dataId[1:0]"
Toggle _caches_0_io_chi_req_bits_opcode [6] "net _caches_0_io_chi_req_bits_opcode[6:0]"
Toggle _caches_0_io_chi_txRsp_bits_tgtId [6:0] "net _caches_0_io_chi_txRsp_bits_tgtId[6:0]"
Toggle _caches_0_io_chi_txRsp_bits_opcode [4:2] "net _caches_0_io_chi_txRsp_bits_opcode[4:0]"
Toggle _caches_0_io_chi_txRsp_bits_resp [2:1] "net _caches_0_io_chi_txRsp_bits_resp[2:0]"
Toggle _caches_0_io_chi_txDat_bits_tgtId [6:0] "net _caches_0_io_chi_txDat_bits_tgtId[6:0]"
Toggle _caches_0_io_chi_txDat_bits_opcode [3:2] "net _caches_0_io_chi_txDat_bits_opcode[3:0]"
Toggle _caches_0_io_chi_txDat_bits_dataId [0] "net _caches_0_io_chi_txDat_bits_dataId[1:0]"
ANNOTATION: "256b DAT carries two 128b chunks; legal DataID values are0 and2, explicitly asserted on input."
Toggle io_chi_txDat_bits_dataId [0] "net io_chi_txDat_bits_dataId[1:0]"
ANNOTATION: "Elaborated BankedChiCache.sv contains this optional input only in its port declaration and has no datapath/control use; the selected endpoint does not consume this field."
ANNOTATION: "Declared single-Home configuration uses HomeID64; receive assertions enforce it."
ANNOTATION: "Supported non-forwarded cache snoops explicitly assert zero forwarding/PAS fields."
ANNOTATION: "Supported snoops are explicitly asserted to be SnpNotSharedDirty4, SnpUnique7 or SnpCleanInvalid9; bit4 is zero."
ANNOTATION: "64B coherence line address is carried shifted by3 on SNP; low3 transmitted address bits are zero."
ANNOTATION: "Declared requester NodeID1; receive assertions enforce destination."
ANNOTATION: "BankedChiCache allocates one outstanding demand per bank with request TxnID0/1, and asserts returned transaction bank range."
ANNOTATION: "Eviction completion is explicitly asserted to be Comp4 or CompDBIDResp5; only bit0 varies."
ANNOTATION: "RN-F eviction receive contract explicitly asserts RespErr0; only successful Comp/CompDBIDResp completes eviction."
ANNOTATION: "RN-F fill receive contract explicitly asserts CompData opcode4."
ANNOTATION: "CHI DAT extends the 12-bit DBID to 16 bits; high4 are asserted MBZ, independent Home IDs cover the entire low12."
ANNOTATION: "Direct-mapped eight-entry cache: tag bits[2:0] are the fixed entry index, including bank bit; all higher 35 tag bits remain measured."
Toggle io_directory_0_line [2:0] "net io_directory_0_line[37:0]"
Toggle io_directory_1_line [2:0] "net io_directory_1_line[37:0]"
Toggle io_directory_2_line [2:0] "net io_directory_2_line[37:0]"
Toggle io_directory_3_line [2:0] "net io_directory_3_line[37:0]"
Toggle io_directory_4_line [2:0] "net io_directory_4_line[37:0]"
Toggle io_directory_5_line [2:0] "net io_directory_5_line[37:0]"
Toggle io_directory_6_line [2:0] "net io_directory_6_line[37:0]"
Toggle io_directory_7_line [2:0] "net io_directory_7_line[37:0]"
ANNOTATION: "Retirement Queue has depth2, so occupancy is0..2; upper30 bits are zero for this parameter configuration."
Toggle io_outstanding [31:2] "net io_outstanding[31:0]"
ANNOTATION: "Per-bank request address is64B aligned and bit6 is the static bank selector; all higher bits remain measured."
Toggle _caches_1_io_chi_req_bits_addr [6:0] "net _caches_1_io_chi_req_bits_addr[43:0]"
Toggle _caches_0_io_chi_req_bits_addr [6:0] "net _caches_0_io_chi_req_bits_addr[43:0]"

CHECKSUM: "3740104159 3441459633"
INSTANCE: rnf_tb.dut.caches_0
ANNOTATION: "CacheAccess asserts at least 4B alignment; word atomics permit bit2 to vary and it remains measured."
Toggle 0to1 io_access_bits_addr [0] "net io_access_bits_addr[43:0]"
ANNOTATION: "ChiCache.scala request/txRsp/txDat zero-initialization and explicit assignments: NodeID1, HomeID64, one Home; two banks use request TxnID0/1; 64B lines and 256b DataID0/2; only ReadNotSharedDirty/ReadUnique/Evict/WriteBackFull, CompAck/SnpResp and SnpRespData/CopyBackWriteData. Dataless snoop responses have only I/SC; dirty snoops return DAT. Optional attributes remain zero. Full payload, Home DBID and snoop TxnID are not excluded."
Toggle io_chi_req_bits_opcode [6] "net io_chi_req_bits_opcode[6:0]"
Toggle io_chi_txRsp_bits_tgtId [6:0] "net io_chi_txRsp_bits_tgtId[6:0]"
Toggle io_chi_txRsp_bits_opcode [4:2] "net io_chi_txRsp_bits_opcode[4:0]"
Toggle io_chi_txRsp_bits_resp [2:1] "net io_chi_txRsp_bits_resp[2:0]"
Toggle io_chi_txDat_bits_tgtId [6:0] "net io_chi_txDat_bits_tgtId[6:0]"
Toggle io_chi_txDat_bits_opcode [3:2] "net io_chi_txDat_bits_opcode[3:0]"
ANNOTATION: "64B-aligned demand/victim address from this fixed cache bank; bit6 is its bank number."
Toggle io_chi_req_bits_addr [6:0] "net io_chi_req_bits_addr[43:0]"
ANNOTATION: "256b DAT carries two 128b chunks; legal DataID values are0 and2, explicitly asserted on input."
Toggle io_chi_txDat_bits_dataId [0] "net io_chi_txDat_bits_dataId[1:0]"
ANNOTATION: "Declared single-Home configuration uses HomeID64; receive assertions enforce it."
Toggle snoop_srcId [6:0] "reg snoop_srcId[6:0]"
ANNOTATION: "Supported non-forwarded cache snoops explicitly assert zero forwarding/PAS fields."
ANNOTATION: "Supported snoops are explicitly asserted to be SnpNotSharedDirty4, SnpUnique7 or SnpCleanInvalid9; bit4 is zero."
ANNOTATION: "64B coherence line address is carried shifted by3 on SNP; low3 transmitted address bits are zero."
ANNOTATION: "Declared requester NodeID1; receive assertions enforce destination."
ANNOTATION: "BankedChiCache allocates one outstanding demand per bank with request TxnID0/1, and asserts returned transaction bank range."
ANNOTATION: "Eviction completion is explicitly asserted to be Comp4 or CompDBIDResp5; only bit0 varies."
ANNOTATION: "RN-F eviction receive contract explicitly asserts RespErr0; only successful Comp/CompDBIDResp completes eviction."
ANNOTATION: "RN-F fill receive contract explicitly asserts CompData opcode4."
ANNOTATION: "CHI DAT extends the 12-bit DBID to 16 bits; high4 are asserted MBZ, independent Home IDs cover the entire low12."
ANNOTATION: "Direct-mapped eight-entry cache: tag bits[2:0] are the fixed entry index, including bank bit; all higher 35 tag bits remain measured."
Toggle io_directory_0_line [2:0] "net io_directory_0_line[37:0]"
Toggle io_directory_1_line [2:0] "net io_directory_1_line[37:0]"
Toggle io_directory_2_line [2:0] "net io_directory_2_line[37:0]"
Toggle io_directory_3_line [2:0] "net io_directory_3_line[37:0]"
Toggle tags_0 [2:0] "reg tags_0[37:0]"
Toggle tags_1 [2:0] "reg tags_1[37:0]"
Toggle tags_2 [2:0] "reg tags_2[37:0]"
Toggle tags_3 [2:0] "reg tags_3[37:0]"
ANNOTATION: "Accepted CPU addresses are naturally aligned: 4B for word atomics, 8B otherwise; low2 remain zero; this cache bank owns address bit6. Victim addresses are 64B aligned."
Toggle command_addr [1:0] "reg command_addr[43:0]"
Toggle command_addr [6] "reg command_addr[43:0]"
Toggle reservationAddress [1:0] "reg reservationAddress[43:0]"
Toggle reservationAddress [6] "reg reservationAddress[43:0]"
Toggle victimAddress [6:0] "reg victimAddress[43:0]"
ANNOTATION: "Supported snoops finish in I or SC, optionally PassDirty; no snoop grants Unique."
Toggle snoopResult [1] "reg snoopResult[2:0]"
ANNOTATION: "ChiCache.scala tags39 and lookup116: exact emitted concatenation {{tags_3},{tags_2},{tags_1},{tags_0}}. Each element copies the already-proven fixed entry-index/bank bits[2:0]; all35 high tag bits per entry remain measured."
Toggle _GEN_1 [3:0][2:0] "net [3:0][37:0]_GEN_1"
ANNOTATION: "ChiCache.scala lookup116 indexes that four-entry tag vector by command.addr[8:7]. Its low bit is the bank bit shared by all entries in this leaf. Index bits[2:1] still vary and remain measured."
Toggle _GEN_2 [0] "net _GEN_2[37:0]"
ANNOTATION: "ChiCache.scala merge word shift at124: command.addr(5,3)<<6. Exact emitted expression {503'h0, command_addr[5:3], 6'h0} proves only[8:6] can vary. Those three bits remain measured; no payload bits are excluded."
Toggle _GEN_8 [5:0] "net _GEN_8[511:0]"
Toggle _GEN_8 [511:9] "net _GEN_8[511:0]"
ANNOTATION: "ChiCache.scala DataID decode205: exact emitted expression {1'h0,io_chi_rxDat_bits_dataId[1]}; only the high zero-extension bit is waived. Selected beat bit0 remains measured."
Toggle _GEN_20 [1] "net _GEN_20[1:0]"

CHECKSUM: "1855582600 3441459633"
INSTANCE: rnf_tb.dut.caches_1
ANNOTATION: "CacheAccess asserts at least 4B alignment; word atomics permit bit2 to vary and it remains measured."
Toggle 0to1 io_access_bits_addr [0] "net io_access_bits_addr[43:0]"
ANNOTATION: "ChiCache.scala request/txRsp/txDat zero-initialization and explicit assignments: NodeID1, HomeID64, one Home; two banks use request TxnID0/1; 64B lines and 256b DataID0/2; only ReadNotSharedDirty/ReadUnique/Evict/WriteBackFull, CompAck/SnpResp and SnpRespData/CopyBackWriteData. Dataless snoop responses have only I/SC; dirty snoops return DAT. Optional attributes remain zero. Full payload, Home DBID and snoop TxnID are not excluded."
Toggle io_chi_req_bits_opcode [6] "net io_chi_req_bits_opcode[6:0]"
Toggle io_chi_txRsp_bits_tgtId [6:0] "net io_chi_txRsp_bits_tgtId[6:0]"
Toggle io_chi_txRsp_bits_opcode [4:2] "net io_chi_txRsp_bits_opcode[4:0]"
Toggle io_chi_txRsp_bits_resp [2:1] "net io_chi_txRsp_bits_resp[2:0]"
Toggle io_chi_txDat_bits_tgtId [6:0] "net io_chi_txDat_bits_tgtId[6:0]"
Toggle io_chi_txDat_bits_opcode [3:2] "net io_chi_txDat_bits_opcode[3:0]"
ANNOTATION: "64B-aligned demand/victim address from this fixed cache bank; bit6 is its bank number."
Toggle io_chi_req_bits_addr [6:0] "net io_chi_req_bits_addr[43:0]"
ANNOTATION: "256b DAT carries two 128b chunks; legal DataID values are0 and2, explicitly asserted on input."
Toggle io_chi_txDat_bits_dataId [0] "net io_chi_txDat_bits_dataId[1:0]"
ANNOTATION: "Declared single-Home configuration uses HomeID64; receive assertions enforce it."
Toggle snoop_srcId [6:0] "reg snoop_srcId[6:0]"
ANNOTATION: "Supported non-forwarded cache snoops explicitly assert zero forwarding/PAS fields."
ANNOTATION: "Supported snoops are explicitly asserted to be SnpNotSharedDirty4, SnpUnique7 or SnpCleanInvalid9; bit4 is zero."
ANNOTATION: "64B coherence line address is carried shifted by3 on SNP; low3 transmitted address bits are zero."
ANNOTATION: "Declared requester NodeID1; receive assertions enforce destination."
ANNOTATION: "BankedChiCache allocates one outstanding demand per bank with request TxnID0/1, and asserts returned transaction bank range."
ANNOTATION: "Eviction completion is explicitly asserted to be Comp4 or CompDBIDResp5; only bit0 varies."
ANNOTATION: "RN-F eviction receive contract explicitly asserts RespErr0; only successful Comp/CompDBIDResp completes eviction."
ANNOTATION: "RN-F fill receive contract explicitly asserts CompData opcode4."
ANNOTATION: "CHI DAT extends the 12-bit DBID to 16 bits; high4 are asserted MBZ, independent Home IDs cover the entire low12."
ANNOTATION: "Direct-mapped eight-entry cache: tag bits[2:0] are the fixed entry index, including bank bit; all higher 35 tag bits remain measured."
Toggle io_directory_0_line [2:0] "net io_directory_0_line[37:0]"
Toggle io_directory_1_line [2:0] "net io_directory_1_line[37:0]"
Toggle io_directory_2_line [2:0] "net io_directory_2_line[37:0]"
Toggle io_directory_3_line [2:0] "net io_directory_3_line[37:0]"
Toggle tags_0 [2:0] "reg tags_0[37:0]"
Toggle tags_1 [2:0] "reg tags_1[37:0]"
Toggle tags_2 [2:0] "reg tags_2[37:0]"
Toggle tags_3 [2:0] "reg tags_3[37:0]"
ANNOTATION: "Accepted CPU addresses are naturally aligned: 4B for word atomics, 8B otherwise; low2 remain zero; this cache bank owns address bit6. Victim addresses are 64B aligned."
Toggle command_addr [1:0] "reg command_addr[43:0]"
Toggle command_addr [6] "reg command_addr[43:0]"
Toggle reservationAddress [1:0] "reg reservationAddress[43:0]"
Toggle reservationAddress [6] "reg reservationAddress[43:0]"
Toggle victimAddress [6:0] "reg victimAddress[43:0]"
ANNOTATION: "Supported snoops finish in I or SC, optionally PassDirty; no snoop grants Unique."
Toggle snoopResult [1] "reg snoopResult[2:0]"
ANNOTATION: "ChiCache.scala tags39 and lookup116: exact emitted concatenation {{tags_3},{tags_2},{tags_1},{tags_0}}. Each element copies the already-proven fixed entry-index/bank bits[2:0]; all35 high tag bits per entry remain measured."
Toggle _GEN_1 [3:0][2:0] "net [3:0][37:0]_GEN_1"
ANNOTATION: "ChiCache.scala lookup116 indexes that four-entry tag vector by command.addr[8:7]. Its low bit is the bank bit shared by all entries in this leaf. Index bits[2:1] still vary and remain measured."
Toggle _GEN_2 [0] "net _GEN_2[37:0]"
ANNOTATION: "ChiCache.scala merge word shift at124: command.addr(5,3)<<6. Exact emitted expression {503'h0, command_addr[5:3], 6'h0} proves only[8:6] can vary. Those three bits remain measured; no payload bits are excluded."
Toggle _GEN_8 [5:0] "net _GEN_8[511:0]"
Toggle _GEN_8 [511:9] "net _GEN_8[511:0]"
ANNOTATION: "ChiCache.scala DataID decode205: exact emitted expression {1'h0,io_chi_rxDat_bits_dataId[1]}; only the high zero-extension bit is waived. Selected beat bit0 remains measured."
Toggle _GEN_20 [1] "net _GEN_20[1:0]"

CHECKSUM: "955580813 3852473397"
INSTANCE: rnf_tb.dut.datQueue
ANNOTATION: "ChiCache.scala request/txRsp/txDat zero-initialization and explicit assignments: NodeID1, HomeID64, one Home; two banks use request TxnID0/1; 64B lines and 256b DataID0/2; only ReadNotSharedDirty/ReadUnique/Evict/WriteBackFull, CompAck/SnpResp and SnpRespData/CopyBackWriteData. Dataless snoop responses have only I/SC; dirty snoops return DAT. Optional attributes remain zero. Full payload, Home DBID and snoop TxnID are not excluded."
Toggle io_enq_bits_tgtId [6:0] "net io_enq_bits_tgtId[6:0]"
Toggle io_enq_bits_opcode [3:2] "net io_enq_bits_opcode[3:0]"
Toggle io_enq_bits_dataId [0] "net io_enq_bits_dataId[1:0]"
Toggle io_deq_bits_qos [3:0] "net io_deq_bits_qos[3:0]"
Toggle io_deq_bits_tgtId [6:0] "net io_deq_bits_tgtId[6:0]"
Toggle io_deq_bits_srcId [6:0] "net io_deq_bits_srcId[6:0]"
Toggle io_deq_bits_homeNid [6:0] "net io_deq_bits_homeNid[6:0]"
Toggle io_deq_bits_opcode [3:2] "net io_deq_bits_opcode[3:0]"
Toggle io_deq_bits_respErr [1:0] "net io_deq_bits_respErr[1:0]"
Toggle io_deq_bits_dataSource [7:0] "net io_deq_bits_dataSource[7:0]"
Toggle io_deq_bits_dataPull "net io_deq_bits_dataPull"
Toggle io_deq_bits_cBusy [2:0] "net io_deq_bits_cBusy[2:0]"
Toggle io_deq_bits_dbid [15:0] "net io_deq_bits_dbid[15:0]"
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
Toggle _ram_ext_R0_data [17:0] "net _ram_ext_R0_data[388:0]"
Toggle _ram_ext_R0_data [36:30] "net _ram_ext_R0_data[388:0]"
Toggle _ram_ext_R0_data [42:39] "net _ram_ext_R0_data[388:0]"
Toggle _ram_ext_R0_data [76:46] "net _ram_ext_R0_data[388:0]"
Toggle _ram_ext_R0_data [100:78] "net _ram_ext_R0_data[388:0]"

CHECKSUM: "4151141343 2478404233"
INSTANCE: rnf_tb.dut.reqArb
ANNOTATION: "ChiCache.scala request/txRsp/txDat zero-initialization and explicit assignments: NodeID1, HomeID64, one Home; two banks use request TxnID0/1; 64B lines and 256b DataID0/2; only ReadNotSharedDirty/ReadUnique/Evict/WriteBackFull, CompAck/SnpResp and SnpRespData/CopyBackWriteData. Dataless snoop responses have only I/SC; dirty snoops return DAT. Optional attributes remain zero. Full payload, Home DBID and snoop TxnID are not excluded."
Toggle io_in_0_bits_opcode [6] "net io_in_0_bits_opcode[6:0]"
Toggle io_in_1_bits_opcode [6] "net io_in_1_bits_opcode[6:0]"
Toggle io_out_bits_txnId [11:1] "net io_out_bits_txnId[11:0]"
Toggle io_out_bits_opcode [6] "net io_out_bits_opcode[6:0]"
Toggle io_out_bits_addr [5:0] "net io_out_bits_addr[43:0]"
ANNOTATION: "Each request-arbiter input is statically wired to its cache bank; low6 address bits are line alignment and bit6 is fixed per input."
Toggle io_in_0_bits_addr [6:0] "net io_in_0_bits_addr[43:0]"
Toggle io_in_1_bits_addr [6:0] "net io_in_1_bits_addr[43:0]"

CHECKSUM: "3263459408 3693796863"
INSTANCE: rnf_tb.dut.rspQueue
ANNOTATION: "ChiCache.scala request/txRsp/txDat zero-initialization and explicit assignments: NodeID1, HomeID64, one Home; two banks use request TxnID0/1; 64B lines and 256b DataID0/2; only ReadNotSharedDirty/ReadUnique/Evict/WriteBackFull, CompAck/SnpResp and SnpRespData/CopyBackWriteData. Dataless snoop responses have only I/SC; dirty snoops return DAT. Optional attributes remain zero. Full payload, Home DBID and snoop TxnID are not excluded."
Toggle io_enq_bits_tgtId [6:0] "net io_enq_bits_tgtId[6:0]"
Toggle io_enq_bits_opcode [4:2] "net io_enq_bits_opcode[4:0]"
Toggle io_enq_bits_resp [2:1] "net io_enq_bits_resp[2:0]"
Toggle io_deq_bits_qos [3:0] "net io_deq_bits_qos[3:0]"
Toggle io_deq_bits_tgtId [6:0] "net io_deq_bits_tgtId[6:0]"
Toggle io_deq_bits_srcId [6:0] "net io_deq_bits_srcId[6:0]"
Toggle io_deq_bits_opcode [4:2] "net io_deq_bits_opcode[4:0]"
Toggle io_deq_bits_respErr [1:0] "net io_deq_bits_respErr[1:0]"
Toggle io_deq_bits_resp [2:1] "net io_deq_bits_resp[2:0]"
Toggle io_deq_bits_fwdState [2:0] "net io_deq_bits_fwdState[2:0]"
Toggle io_deq_bits_cBusy [2:0] "net io_deq_bits_cBusy[2:0]"
Toggle io_deq_bits_dbid [11:0] "net io_deq_bits_dbid[11:0]"
Toggle io_deq_bits_pCrdType [3:0] "net io_deq_bits_pCrdType[3:0]"
Toggle io_deq_bits_tagOp [1:0] "net io_deq_bits_tagOp[1:0]"
Toggle io_deq_bits_traceTag "net io_deq_bits_traceTag"
Toggle io_deq_bits_cacheLineId [5:0] "net io_deq_bits_cacheLineId[5:0]"
Toggle _ram_ext_R0_data [17:0] "net _ram_ext_R0_data[70:0]"
Toggle _ram_ext_R0_data [36:32] "net _ram_ext_R0_data[70:0]"
Toggle _ram_ext_R0_data [70:38] "net _ram_ext_R0_data[70:0]"

CHECKSUM: "2475771928 3553047771"
INSTANCE: rnf_tb.dut.rspArb
ANNOTATION: "ChiCache.scala request/txRsp/txDat zero-initialization and explicit assignments: NodeID1, HomeID64, one Home; two banks use request TxnID0/1; 64B lines and 256b DataID0/2; only ReadNotSharedDirty/ReadUnique/Evict/WriteBackFull, CompAck/SnpResp and SnpRespData/CopyBackWriteData. Dataless snoop responses have only I/SC; dirty snoops return DAT. Optional attributes remain zero. Full payload, Home DBID and snoop TxnID are not excluded."
Toggle io_in_0_bits_tgtId [6:0] "net io_in_0_bits_tgtId[6:0]"
Toggle io_in_0_bits_opcode [4:2] "net io_in_0_bits_opcode[4:0]"
Toggle io_in_0_bits_resp [2:1] "net io_in_0_bits_resp[2:0]"
Toggle io_in_1_bits_tgtId [6:0] "net io_in_1_bits_tgtId[6:0]"
Toggle io_in_1_bits_opcode [4:2] "net io_in_1_bits_opcode[4:0]"
Toggle io_in_1_bits_resp [2:1] "net io_in_1_bits_resp[2:0]"
Toggle io_out_bits_tgtId [6:0] "net io_out_bits_tgtId[6:0]"
Toggle io_out_bits_opcode [4:2] "net io_out_bits_opcode[4:0]"
Toggle io_out_bits_resp [2:1] "net io_out_bits_resp[2:0]"

CHECKSUM: "3568308182 1365416039"
INSTANCE: rnf_tb.dut.datArb
ANNOTATION: "ChiCache.scala request/txRsp/txDat zero-initialization and explicit assignments: NodeID1, HomeID64, one Home; two banks use request TxnID0/1; 64B lines and 256b DataID0/2; only ReadNotSharedDirty/ReadUnique/Evict/WriteBackFull, CompAck/SnpResp and SnpRespData/CopyBackWriteData. Dataless snoop responses have only I/SC; dirty snoops return DAT. Optional attributes remain zero. Full payload, Home DBID and snoop TxnID are not excluded."
Toggle io_in_0_bits_tgtId [6:0] "net io_in_0_bits_tgtId[6:0]"
Toggle io_in_0_bits_opcode [3:2] "net io_in_0_bits_opcode[3:0]"
Toggle io_in_0_bits_dataId [0] "net io_in_0_bits_dataId[1:0]"
Toggle io_in_1_bits_tgtId [6:0] "net io_in_1_bits_tgtId[6:0]"
Toggle io_in_1_bits_opcode [3:2] "net io_in_1_bits_opcode[3:0]"
Toggle io_in_1_bits_dataId [0] "net io_in_1_bits_dataId[1:0]"
Toggle io_out_bits_tgtId [6:0] "net io_out_bits_tgtId[6:0]"
Toggle io_out_bits_opcode [3:2] "net io_out_bits_opcode[3:0]"
Toggle io_out_bits_dataId [0] "net io_out_bits_dataId[1:0]"

CHECKSUM: "3973295261 1542809678"
INSTANCE: rnf_tb.dut.reqQueue
ANNOTATION: "ChiCache.scala request/txRsp/txDat zero-initialization and explicit assignments: NodeID1, HomeID64, one Home; two banks use request TxnID0/1; 64B lines and 256b DataID0/2; only ReadNotSharedDirty/ReadUnique/Evict/WriteBackFull, CompAck/SnpResp and SnpRespData/CopyBackWriteData. Dataless snoop responses have only I/SC; dirty snoops return DAT. Optional attributes remain zero. Full payload, Home DBID and snoop TxnID are not excluded."
Toggle io_enq_bits_txnId [11:1] "net io_enq_bits_txnId[11:0]"
Toggle io_enq_bits_opcode [6] "net io_enq_bits_opcode[6:0]"
Toggle io_enq_bits_addr [5:0] "net io_enq_bits_addr[43:0]"
Toggle io_deq_bits_qos [3:0] "net io_deq_bits_qos[3:0]"
Toggle io_deq_bits_tgtId [6:0] "net io_deq_bits_tgtId[6:0]"
Toggle io_deq_bits_srcId [6:0] "net io_deq_bits_srcId[6:0]"
Toggle io_deq_bits_txnId [11:1] "net io_deq_bits_txnId[11:0]"
Toggle io_deq_bits_returnNid [6:0] "net io_deq_bits_returnNid[6:0]"
Toggle io_deq_bits_stashNidValidEndian "net io_deq_bits_stashNidValidEndian"
Toggle io_deq_bits_returnTxnId [11:0] "net io_deq_bits_returnTxnId[11:0]"
Toggle io_deq_bits_opcode [6] "net io_deq_bits_opcode[6:0]"
Toggle io_deq_bits_multiReq "net io_deq_bits_multiReq"
Toggle io_deq_bits_size [5:0] "net io_deq_bits_size[5:0]"
Toggle io_deq_bits_addr [5:0] "net io_deq_bits_addr[43:0]"
Toggle io_deq_bits_pas [2:0] "net io_deq_bits_pas[2:0]"
Toggle io_deq_bits_likelyShared "net io_deq_bits_likelyShared"
Toggle io_deq_bits_allowRetry "net io_deq_bits_allowRetry"
Toggle io_deq_bits_order [1:0] "net io_deq_bits_order[1:0]"
Toggle io_deq_bits_pCrdType [3:0] "net io_deq_bits_pCrdType[3:0]"
Toggle io_deq_bits_memAttr [3:0] "net io_deq_bits_memAttr[3:0]"
Toggle io_deq_bits_snpAttr "net io_deq_bits_snpAttr"
Toggle io_deq_bits_lpid [7:0] "net io_deq_bits_lpid[7:0]"
Toggle io_deq_bits_exclSnoopMe "net io_deq_bits_exclSnoopMe"
Toggle io_deq_bits_tagOp [1:0] "net io_deq_bits_tagOp[1:0]"
Toggle io_deq_bits_traceTag "net io_deq_bits_traceTag"
Toggle _ram_ext_R0_data [17:0] "net _ram_ext_R0_data[136:0]"
Toggle _ram_ext_R0_data [49:19] "net _ram_ext_R0_data[136:0]"
Toggle _ram_ext_R0_data [69:56] "net _ram_ext_R0_data[136:0]"
Toggle _ram_ext_R0_data [132:108] "net _ram_ext_R0_data[136:0]"
Toggle _ram_ext_R0_data [136:134] "net _ram_ext_R0_data[136:0]"

CHECKSUM: "2670869491 103376762"
INSTANCE: rnf_tb.dut
ANNOTATION: "Normal legal-input/invariant coverage excludes only the all-true assertion-failure guard: Unknown CPU RSP transaction bank. The assertion and its valid/ready/reset controls stay enabled; no successful vector is waived."
Condition 1 "2822280289" "(io_chi_rxRsp_valid & ((~reset)) & ((|io_chi_rxRsp_bits_txnId[11:1]))) 1 -1" (4 "111")
ANNOTATION: "Normal legal-input/invariant coverage excludes only the all-true assertion-failure guard: Unknown CPU DAT transaction bank. The assertion and its valid/ready/reset controls stay enabled; no successful vector is waived."
Condition 2 "1067523315" "(io_chi_rxDat_valid & ((~reset)) & ((|io_chi_rxDat_bits_txnId[11:1]))) 1 -1" (4 "111")

CHECKSUM: "3740104159 1656763948"
INSTANCE: rnf_tb.dut.caches_0
ANNOTATION: "Normal legal-input/invariant coverage excludes only the all-true assertion-failure guard: Cache client requires naturally aligned accesses. The assertion and its valid/ready/reset controls stay enabled; no successful vector is waived."
Condition 3 "1701336068" "(_GEN_32 & ( ~ (io_access_bits_atomicWord ? (io_access_bits_addr[1:0] == 2'b0) : (io_access_bits_addr[2:0] == 3'b0)) )) 1 -1" (3 "11")
ANNOTATION: "Normal legal-input/invariant coverage excludes only the all-true assertion-failure guard: Word width requires AMO, LR or SC. The assertion and its valid/ready/reset controls stay enabled; no successful vector is waived."
Condition 8 "200834647" "(_GEN & io_access_bits_atomicWord & ((~reset)) & ( ~ (((|io_access_bits_atomic)) & (io_access_bits_atomic[3:2] != 2'h3)) )) 1 -1" (5 "1111")
ANNOTATION: "Normal legal-input/invariant coverage excludes only the all-true assertion-failure guard: Unknown CPU atomic operation. The assertion and its valid/ready/reset controls stay enabled; no successful vector is waived."
Condition 12 "2091006621" "(_GEN_32 & (io_access_bits_atomic > 4'hc)) 1 -1" (3 "11")
ANNOTATION: "Normal legal-input/invariant coverage excludes only the all-true assertion-failure guard: Atomic operation requires the full operand marker. The assertion and its valid/ready/reset controls stay enabled; no successful vector is waived."
Condition 13 "4053812370" "(_GEN & ((|io_access_bits_atomic)) & ((~reset)) & ( ~ (((~io_access_bits_write)) & ((&io_access_bits_mask))) )) 1 -1" (5 "1111")
ANNOTATION: "Normal legal-input/invariant coverage excludes only the all-true assertion-failure guard: Unexpected eviction completion. The assertion and its valid/ready/reset controls stay enabled; no successful vector is waived."
Condition 16 "1066899246" "(_GEN_17 & ((~reset)) & ( ~ ((io_chi_rxRsp_bits_tgtId == 7'b1) & (io_chi_rxRsp_bits_srcId == 7'h40) & (io_chi_rxRsp_bits_txnId == 12'b0) & (io_chi_rxRsp_bits_respErr == 2'b0)) )) 1 -1" (4 "111")
ANNOTATION: "Normal legal-input/invariant coverage excludes only the all-true assertion-failure guard: WriteBackFull requires CompDBIDResp. The assertion and its valid/ready/reset controls stay enabled; no successful vector is waived."
Condition 23 "2511136329" "(_GEN_18 & ((~reset)) & (io_chi_rxRsp_bits_opcode != 5'h05)) 1 -1" (4 "111")
ANNOTATION: "Normal legal-input/invariant coverage excludes only the all-true assertion-failure guard: Evict requires Comp. The assertion and its valid/ready/reset controls stay enabled; no successful vector is waived."
Condition 25 "1452963988" "(_GEN_17 & ((~victimWasDirty)) & ((~reset)) & (io_chi_rxRsp_bits_opcode != 5'h04)) 1 -1" (5 "1111")
ANNOTATION: "Normal legal-input/invariant coverage excludes only the all-true assertion-failure guard: Unexpected cache fill. The assertion and its valid/ready/reset controls stay enabled; no successful vector is waived."
Condition 27 "1178277413" "(_GEN_33 & ( ~ ((io_chi_rxDat_bits_opcode == 4'h4) & (io_chi_rxDat_bits_srcId == 7'h40) & (io_chi_rxDat_bits_tgtId == 7'b1) & (io_chi_rxDat_bits_txnId == 12'b0) & (io_chi_rxDat_bits_homeNid == 7'h40)) )) 1 -1" (3 "11")
ANNOTATION: "Normal legal-input/invariant coverage excludes only the all-true assertion-failure guard: Invalid fill DataID. The assertion and its valid/ready/reset controls stay enabled; no successful vector is waived."
Condition 35 "2425277191" "(_GEN_33 & ({(io_chi_rxDat_bits_dataId == 2'h2), (io_chi_rxDat_bits_dataId == 2'b0)} == 2'b0)) 1 -1" (3 "11")
ANNOTATION: "Normal legal-input/invariant coverage excludes only the all-true assertion-failure guard: Duplicate fill DataID. The assertion and its valid/ready/reset controls stay enabled; no successful vector is waived."
Condition 37 "394526459" "(_GEN_33 & ((|(fillSeen & (2'b1 << _GEN_20))))) 1 -1" (3 "11")
ANNOTATION: "Normal legal-input/invariant coverage excludes only the all-true assertion-failure guard: Fill DBID exceeds completion ID width. The assertion and its valid/ready/reset controls stay enabled; no successful vector is waived."
Condition 38 "146195846" "(_GEN_33 & ((|io_chi_rxDat_bits_dbid[15:12]))) 1 -1" (3 "11")
ANNOTATION: "Normal legal-input/invariant coverage excludes only the all-true assertion-failure guard: Inconsistent multi-beat fill metadata. The assertion and its valid/ready/reset controls stay enabled; no successful vector is waived."
Condition 39 "3121137474" "(_GEN_19 & ((|fillSeen)) & ((~reset)) & ( ~ ((io_chi_rxDat_bits_dbid == {4'b0, fillDbid}) & (io_chi_rxDat_bits_resp == fillPermission)) )) 1 -1" (5 "1111")
ANNOTATION: "Normal legal-input/invariant coverage excludes only the all-true assertion-failure guard: Home granted unexpected cache permission. The assertion and its valid/ready/reset controls stay enabled; no successful vector is waived."
Condition 44 "2935759336" "(_GEN_21 & ((~failed)) & ((~reset)) & (io_chi_rxDat_bits_resp != {1'b0, (modifies ? 2'h2 : 2'b1)})) 1 -1" (5 "1111")
ANNOTATION: "Normal legal-input/invariant coverage excludes only the all-true assertion-failure guard: Unsupported cache snoop. The assertion and its valid/ready/reset controls stay enabled; no successful vector is waived."
Condition 46 "765642000" "(_GEN_31 & ((~reset)) & ( ~ ((io_chi_snp_bits_srcId == 7'h40) & (io_chi_snp_bits_pas == 3'b0) & (io_chi_snp_bits_fwdNid == 7'b0) & (io_chi_snp_bits_fwdTxnId == 12'b0) & (invalidate | (io_chi_snp_bits_opcode == 5'h04))) )) 1 -1" (4 "111")
ANNOTATION: "ChiCache invariant: dirty implies valid and writable; writable implies valid. Reset clears all three; successful fill establishes valid before setting writable/dirty; writes require valid writable hit; snoops and eviction clear dirty/writable with invalidation. This exact vector violates the preserved invariant; assertions remain enabled."
Condition 55 "99428243" "(((~reset)) & ( ~ (((~dirty_0)) | (valid_0 & writable_0)) )) 1 -1" (1 "01")
Condition 55 "99428243" "(((~reset)) & ( ~ (((~dirty_0)) | (valid_0 & writable_0)) )) 1 -1" (3 "11")
Condition 57 "4065545810" "(((~dirty_0)) | (valid_0 & writable_0)) 1 -1" (1 "00")
Condition 58 "148761233" "(valid_0 & writable_0) 1 -1" (1 "01")
Condition 59 "3140844615" "(((~reset)) & ( ~ (((~writable_0)) | valid_0) )) 1 -1" (1 "01")
Condition 59 "3140844615" "(((~reset)) & ( ~ (((~writable_0)) | valid_0) )) 1 -1" (3 "11")
Condition 61 "2853975677" "(((~writable_0)) | valid_0) 1 -1" (1 "00")
Condition 62 "3303882976" "(((~reset)) & ( ~ (((~dirty_1)) | (valid_1 & writable_1)) )) 1 -1" (1 "01")
Condition 62 "3303882976" "(((~reset)) & ( ~ (((~dirty_1)) | (valid_1 & writable_1)) )) 1 -1" (3 "11")
Condition 64 "4032274078" "(((~dirty_1)) | (valid_1 & writable_1)) 1 -1" (1 "00")
Condition 65 "1330287737" "(valid_1 & writable_1) 1 -1" (1 "01")
Condition 66 "4120262286" "(((~reset)) & ( ~ (((~writable_1)) | valid_1) )) 1 -1" (1 "01")
Condition 66 "4120262286" "(((~reset)) & ( ~ (((~writable_1)) | valid_1) )) 1 -1" (3 "11")
Condition 68 "1875412988" "(((~writable_1)) | valid_1) 1 -1" (1 "00")
Condition 69 "3784191037" "(((~reset)) & ( ~ (((~dirty_2)) | (valid_2 & writable_2)) )) 1 -1" (1 "01")
Condition 69 "3784191037" "(((~reset)) & ( ~ (((~dirty_2)) | (valid_2 & writable_2)) )) 1 -1" (3 "11")
Condition 71 "1125482522" "(((~dirty_2)) | (valid_2 & writable_2)) 1 -1" (1 "00")
Condition 72 "1343474919" "(valid_2 & writable_2) 1 -1" (1 "01")
Condition 73 "1856381663" "(((~reset)) & ( ~ (((~writable_2)) | valid_2) )) 1 -1" (1 "01")
Condition 73 "1856381663" "(((~reset)) & ( ~ (((~writable_2)) | valid_2) )) 1 -1" (3 "11")
Condition 75 "2552466111" "(((~writable_2)) | valid_2) 1 -1" (1 "00")
Condition 76 "546196302" "(((~reset)) & ( ~ (((~dirty_3)) | (valid_3 & writable_3)) )) 1 -1" (1 "01")
Condition 76 "546196302" "(((~reset)) & ( ~ (((~dirty_3)) | (valid_3 & writable_3)) )) 1 -1" (3 "11")
Condition 78 "1091653846" "(((~dirty_3)) | (valid_3 & writable_3)) 1 -1" (1 "00")
Condition 79 "394572303" "(valid_3 & writable_3) 1 -1" (1 "01")
Condition 80 "537225238" "(((~reset)) & ( ~ (((~writable_3)) | valid_3) )) 1 -1" (1 "01")
Condition 80 "537225238" "(((~reset)) & ( ~ (((~writable_3)) | valid_3) )) 1 -1" (3 "11")
Condition 82 "1576483646" "(((~writable_3)) | valid_3) 1 -1" (1 "00")
ANNOTATION: "ChiCache.scala eviction140-194: victimAddress captures the selected tag before evictWait; the demand FSM cannot accept CPU work or execute fill while waiting, and snoops never modify tags. Therefore that victim tag cannot mismatch during the copyResp assignment."
Condition 221 "1559919346" "(_GEN_3[victimAddress[8:7]] & (_GEN_1[victimAddress[8:7]] == victimAddress[43:6])) 1 -1" (2 "10")
Condition 222 "2672344398" "(_GEN_1[victimAddress[8:7]] == victimAddress[43:6]) 1 -1" (1 "0")
ANNOTATION: "ChiCache.scala copyResp176-182: WriteBackFull starts with dirty Unique victim. Until its response, only snoops can remove dirty: NotSharedDirty makes Shared and clears writable, invalidating snoops make Invalid. No CPU/fill can execute in evictWait. Therefore the clean-and-still-present victim cannot be writable; this nested UC-clean choice is unreachable. Shared, dirty and Invalid choices remain measured."
Condition 225 "1878230501" "(_GEN_6[victimAddress[8:7]] ? 2'h2 : 2'b1) 1 -1" (2 "1")

CHECKSUM: "1855582600 2062926452"
INSTANCE: rnf_tb.dut.caches_1
ANNOTATION: "Normal legal-input/invariant coverage excludes only the all-true assertion-failure guard: Cache client requires naturally aligned accesses. The assertion and its valid/ready/reset controls stay enabled; no successful vector is waived."
Condition 3 "1701336068" "(_GEN_32 & ( ~ (io_access_bits_atomicWord ? (io_access_bits_addr[1:0] == 2'b0) : (io_access_bits_addr[2:0] == 3'b0)) )) 1 -1" (3 "11")
ANNOTATION: "Normal legal-input/invariant coverage excludes only the all-true assertion-failure guard: Word width requires AMO, LR or SC. The assertion and its valid/ready/reset controls stay enabled; no successful vector is waived."
Condition 8 "200834647" "(_GEN & io_access_bits_atomicWord & ((~reset)) & ( ~ (((|io_access_bits_atomic)) & (io_access_bits_atomic[3:2] != 2'h3)) )) 1 -1" (5 "1111")
ANNOTATION: "Normal legal-input/invariant coverage excludes only the all-true assertion-failure guard: Unknown CPU atomic operation. The assertion and its valid/ready/reset controls stay enabled; no successful vector is waived."
Condition 12 "2091006621" "(_GEN_32 & (io_access_bits_atomic > 4'hc)) 1 -1" (3 "11")
ANNOTATION: "Normal legal-input/invariant coverage excludes only the all-true assertion-failure guard: Atomic operation requires the full operand marker. The assertion and its valid/ready/reset controls stay enabled; no successful vector is waived."
Condition 13 "4053812370" "(_GEN & ((|io_access_bits_atomic)) & ((~reset)) & ( ~ (((~io_access_bits_write)) & ((&io_access_bits_mask))) )) 1 -1" (5 "1111")
ANNOTATION: "Normal legal-input/invariant coverage excludes only the all-true assertion-failure guard: Unexpected eviction completion. The assertion and its valid/ready/reset controls stay enabled; no successful vector is waived."
Condition 16 "2609149868" "(_GEN_17 & ((~reset)) & ( ~ ((io_chi_rxRsp_bits_tgtId == 7'b1) & (io_chi_rxRsp_bits_srcId == 7'h40) & (io_chi_rxRsp_bits_txnId == 12'b1) & (io_chi_rxRsp_bits_respErr == 2'b0)) )) 1 -1" (4 "111")
ANNOTATION: "Normal legal-input/invariant coverage excludes only the all-true assertion-failure guard: WriteBackFull requires CompDBIDResp. The assertion and its valid/ready/reset controls stay enabled; no successful vector is waived."
Condition 23 "2511136329" "(_GEN_18 & ((~reset)) & (io_chi_rxRsp_bits_opcode != 5'h05)) 1 -1" (4 "111")
ANNOTATION: "Normal legal-input/invariant coverage excludes only the all-true assertion-failure guard: Evict requires Comp. The assertion and its valid/ready/reset controls stay enabled; no successful vector is waived."
Condition 25 "1452963988" "(_GEN_17 & ((~victimWasDirty)) & ((~reset)) & (io_chi_rxRsp_bits_opcode != 5'h04)) 1 -1" (5 "1111")
ANNOTATION: "Normal legal-input/invariant coverage excludes only the all-true assertion-failure guard: Unexpected cache fill. The assertion and its valid/ready/reset controls stay enabled; no successful vector is waived."
Condition 27 "548705526" "(_GEN_33 & ( ~ ((io_chi_rxDat_bits_opcode == 4'h4) & (io_chi_rxDat_bits_srcId == 7'h40) & (io_chi_rxDat_bits_tgtId == 7'b1) & (io_chi_rxDat_bits_txnId == 12'b1) & (io_chi_rxDat_bits_homeNid == 7'h40)) )) 1 -1" (3 "11")
ANNOTATION: "Normal legal-input/invariant coverage excludes only the all-true assertion-failure guard: Invalid fill DataID. The assertion and its valid/ready/reset controls stay enabled; no successful vector is waived."
Condition 35 "2425277191" "(_GEN_33 & ({(io_chi_rxDat_bits_dataId == 2'h2), (io_chi_rxDat_bits_dataId == 2'b0)} == 2'b0)) 1 -1" (3 "11")
ANNOTATION: "Normal legal-input/invariant coverage excludes only the all-true assertion-failure guard: Duplicate fill DataID. The assertion and its valid/ready/reset controls stay enabled; no successful vector is waived."
Condition 37 "394526459" "(_GEN_33 & ((|(fillSeen & (2'b1 << _GEN_20))))) 1 -1" (3 "11")
ANNOTATION: "Normal legal-input/invariant coverage excludes only the all-true assertion-failure guard: Fill DBID exceeds completion ID width. The assertion and its valid/ready/reset controls stay enabled; no successful vector is waived."
Condition 38 "146195846" "(_GEN_33 & ((|io_chi_rxDat_bits_dbid[15:12]))) 1 -1" (3 "11")
ANNOTATION: "Normal legal-input/invariant coverage excludes only the all-true assertion-failure guard: Inconsistent multi-beat fill metadata. The assertion and its valid/ready/reset controls stay enabled; no successful vector is waived."
Condition 39 "3121137474" "(_GEN_19 & ((|fillSeen)) & ((~reset)) & ( ~ ((io_chi_rxDat_bits_dbid == {4'b0, fillDbid}) & (io_chi_rxDat_bits_resp == fillPermission)) )) 1 -1" (5 "1111")
ANNOTATION: "Normal legal-input/invariant coverage excludes only the all-true assertion-failure guard: Home granted unexpected cache permission. The assertion and its valid/ready/reset controls stay enabled; no successful vector is waived."
Condition 44 "2935759336" "(_GEN_21 & ((~failed)) & ((~reset)) & (io_chi_rxDat_bits_resp != {1'b0, (modifies ? 2'h2 : 2'b1)})) 1 -1" (5 "1111")
ANNOTATION: "Normal legal-input/invariant coverage excludes only the all-true assertion-failure guard: Unsupported cache snoop. The assertion and its valid/ready/reset controls stay enabled; no successful vector is waived."
Condition 46 "765642000" "(_GEN_31 & ((~reset)) & ( ~ ((io_chi_snp_bits_srcId == 7'h40) & (io_chi_snp_bits_pas == 3'b0) & (io_chi_snp_bits_fwdNid == 7'b0) & (io_chi_snp_bits_fwdTxnId == 12'b0) & (invalidate | (io_chi_snp_bits_opcode == 5'h04))) )) 1 -1" (4 "111")
ANNOTATION: "ChiCache invariant: dirty implies valid and writable; writable implies valid. Reset clears all three; successful fill establishes valid before setting writable/dirty; writes require valid writable hit; snoops and eviction clear dirty/writable with invalidation. This exact vector violates the preserved invariant; assertions remain enabled."
Condition 55 "99428243" "(((~reset)) & ( ~ (((~dirty_0)) | (valid_0 & writable_0)) )) 1 -1" (1 "01")
Condition 55 "99428243" "(((~reset)) & ( ~ (((~dirty_0)) | (valid_0 & writable_0)) )) 1 -1" (3 "11")
Condition 57 "4065545810" "(((~dirty_0)) | (valid_0 & writable_0)) 1 -1" (1 "00")
Condition 58 "148761233" "(valid_0 & writable_0) 1 -1" (1 "01")
Condition 59 "3140844615" "(((~reset)) & ( ~ (((~writable_0)) | valid_0) )) 1 -1" (1 "01")
Condition 59 "3140844615" "(((~reset)) & ( ~ (((~writable_0)) | valid_0) )) 1 -1" (3 "11")
Condition 61 "2853975677" "(((~writable_0)) | valid_0) 1 -1" (1 "00")
Condition 62 "3303882976" "(((~reset)) & ( ~ (((~dirty_1)) | (valid_1 & writable_1)) )) 1 -1" (1 "01")
Condition 62 "3303882976" "(((~reset)) & ( ~ (((~dirty_1)) | (valid_1 & writable_1)) )) 1 -1" (3 "11")
Condition 64 "4032274078" "(((~dirty_1)) | (valid_1 & writable_1)) 1 -1" (1 "00")
Condition 65 "1330287737" "(valid_1 & writable_1) 1 -1" (1 "01")
Condition 66 "4120262286" "(((~reset)) & ( ~ (((~writable_1)) | valid_1) )) 1 -1" (1 "01")
Condition 66 "4120262286" "(((~reset)) & ( ~ (((~writable_1)) | valid_1) )) 1 -1" (3 "11")
Condition 68 "1875412988" "(((~writable_1)) | valid_1) 1 -1" (1 "00")
Condition 69 "3784191037" "(((~reset)) & ( ~ (((~dirty_2)) | (valid_2 & writable_2)) )) 1 -1" (1 "01")
Condition 69 "3784191037" "(((~reset)) & ( ~ (((~dirty_2)) | (valid_2 & writable_2)) )) 1 -1" (3 "11")
Condition 71 "1125482522" "(((~dirty_2)) | (valid_2 & writable_2)) 1 -1" (1 "00")
Condition 72 "1343474919" "(valid_2 & writable_2) 1 -1" (1 "01")
Condition 73 "1856381663" "(((~reset)) & ( ~ (((~writable_2)) | valid_2) )) 1 -1" (1 "01")
Condition 73 "1856381663" "(((~reset)) & ( ~ (((~writable_2)) | valid_2) )) 1 -1" (3 "11")
Condition 75 "2552466111" "(((~writable_2)) | valid_2) 1 -1" (1 "00")
Condition 76 "546196302" "(((~reset)) & ( ~ (((~dirty_3)) | (valid_3 & writable_3)) )) 1 -1" (1 "01")
Condition 76 "546196302" "(((~reset)) & ( ~ (((~dirty_3)) | (valid_3 & writable_3)) )) 1 -1" (3 "11")
Condition 78 "1091653846" "(((~dirty_3)) | (valid_3 & writable_3)) 1 -1" (1 "00")
Condition 79 "394572303" "(valid_3 & writable_3) 1 -1" (1 "01")
Condition 80 "537225238" "(((~reset)) & ( ~ (((~writable_3)) | valid_3) )) 1 -1" (1 "01")
Condition 80 "537225238" "(((~reset)) & ( ~ (((~writable_3)) | valid_3) )) 1 -1" (3 "11")
Condition 82 "1576483646" "(((~writable_3)) | valid_3) 1 -1" (1 "00")
ANNOTATION: "ChiCache.scala eviction140-194: victimAddress captures the selected tag before evictWait; the demand FSM cannot accept CPU work or execute fill while waiting, and snoops never modify tags. Therefore that victim tag cannot mismatch during the copyResp assignment."
Condition 221 "1559919346" "(_GEN_3[victimAddress[8:7]] & (_GEN_1[victimAddress[8:7]] == victimAddress[43:6])) 1 -1" (2 "10")
Condition 222 "2672344398" "(_GEN_1[victimAddress[8:7]] == victimAddress[43:6]) 1 -1" (1 "0")
ANNOTATION: "ChiCache.scala copyResp176-182: WriteBackFull starts with dirty Unique victim. Until its response, only snoops can remove dirty: NotSharedDirty makes Shared and clears writable, invalidating snoops make Invalid. No CPU/fill can execute in evictWait. Therefore the clean-and-still-present victim cannot be writable; this nested UC-clean choice is unreachable. Shared, dirty and Invalid choices remain measured."
Condition 225 "1878230501" "(_GEN_6[victimAddress[8:7]] ? 2'h2 : 2'b1) 1 -1" (2 "1")

CHECKSUM: "3871073622 320702925"
INSTANCE: rnf_tb.dut.retirement
ANNOTATION: "BankedChiCache.scala result68 and retirement70: retirement.deq.ready is io.result.fire, while io.result.valid includes retirement.deq.valid=!empty. Dequeue-ready1 with empty1 is algebraically impossible."
Condition 1 "3657552453" "(io_deq_ready & ((~empty))) 1 -1" (2 "10")

CHECKSUM: "4151141343 2455536461"
INSTANCE: rnf_tb.dut.reqArb
ANNOTATION: "BankedChiCache request arbiter feeds the depth2 request queue from exactly two one-outstanding demand engines. Each engine waits for the externally consumed request before producing its successor. Consequently a full queue cannot coincide with an arbiter valid request; arbitration between two requesting engines remains measured."
Condition 1 "2548440157" "(io_out_ready & io_out_valid_0) 1 -1" (1 "01")

CHECKSUM: "3973295261 405613642"
INSTANCE: rnf_tb.dut.reqQueue
ANNOTATION: "BankedChiCache reqQueue depth2 equals the two cache demand engines. Each engine leaves its request state on enqueue and cannot issue another request until the Home consumes that queued request and returns its completion/data. Thus queued requests plus requesting engines is at most2; a full queue cannot coexist with a third valid enqueue."
Condition 6 "2766317690" "(io_enq_ready_0 & io_enq_valid) 1 -1" (1 "01")


CHECKSUM: "3740104159 1656763948"
INSTANCE: rnf_tb.dut.caches_0
ANNOTATION: "CacheAccess contract requires addr[2:0]=0 whenever atomicWord is false. This exact vector requires atomicWord=false and addr[2]=1 in the captured command, contradicting that asserted alignment. Both 32-bit lanes and all hit/refill word positions remain measured."
Condition 261 "1532177681" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 264 "4223066285" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 267 "185608037" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 270 "3344634722" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 285 "2484325315" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 288 "3385432395" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 291 "3299299247" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 294 "222286461" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 309 "2212333049" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 312 "589292694" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 315 "2763946419" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 318 "445429772" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 333 "3143279874" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 336 "3032913730" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 339 "3138033605" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 342 "2640723344" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 357 "4250892455" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 360 "3751110928" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 363 "621939953" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 366 "1573763276" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 381 "2467609897" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 384 "4146104497" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 387 "1535485727" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 390 "4018743804" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 405 "2637849284" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 408 "1744812935" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 411 "3747674414" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 414 "4013641966" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 429 "1705548714" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 432 "2005699998" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 435 "3782640699" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 438 "1605656809" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 579 "3515112286" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 582 "3778285623" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 585 "1869098444" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 588 "3973782904" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 603 "787686648" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 606 "3419621232" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 609 "1479055489" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 612 "4130120093" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 627 "78484656" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 630 "894614513" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 633 "461271861" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 636 "2004787347" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 651 "2315071871" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 654 "1611970491" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 657 "2817089354" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 660 "4242965520" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 675 "3823756759" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 678 "4160998201" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 681 "934636579" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 684 "2398465506" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 699 "3937676887" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 702 "3080725654" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 705 "1200218633" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 708 "3986779539" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 723 "3581727119" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 726 "1873019898" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 729 "805427420" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 732 "1072629107" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 747 "3369673591" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 750 "2598387317" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 753 "2571805770" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 756 "2624313215" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")

CHECKSUM: "1855582600 2062926452"
INSTANCE: rnf_tb.dut.caches_1
ANNOTATION: "CacheAccess contract requires addr[2:0]=0 whenever atomicWord is false. This exact vector requires atomicWord=false and addr[2]=1 in the captured command, contradicting that asserted alignment. Both 32-bit lanes and all hit/refill word positions remain measured."
Condition 261 "1532177681" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 264 "4223066285" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 267 "185608037" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 270 "3344634722" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 285 "2484325315" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 288 "3385432395" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 291 "3299299247" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 294 "222286461" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 309 "2212333049" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 312 "589292694" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 315 "2763946419" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 318 "445429772" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 333 "3143279874" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 336 "3032913730" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 339 "3138033605" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 342 "2640723344" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 357 "4250892455" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 360 "3751110928" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 363 "621939953" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 366 "1573763276" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 381 "2467609897" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 384 "4146104497" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 387 "1535485727" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 390 "4018743804" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 405 "2637849284" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 408 "1744812935" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 411 "3747674414" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 414 "4013641966" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 429 "1705548714" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 432 "2005699998" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 435 "3782640699" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 438 "1605656809" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 579 "3515112286" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 582 "3778285623" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 585 "1869098444" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 588 "3973782904" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 603 "787686648" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 606 "3419621232" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 609 "1479055489" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 612 "4130120093" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 627 "78484656" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 630 "894614513" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 633 "461271861" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 636 "2004787347" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 651 "2315071871" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 654 "1611970491" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 657 "2817089354" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 660 "4242965520" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 675 "3823756759" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 678 "4160998201" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 681 "934636579" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 684 "2398465506" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 699 "3937676887" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 702 "3080725654" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 705 "1200218633" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 708 "3986779539" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 723 "3581727119" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 726 "1873019898" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 729 "805427420" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 732 "1072629107" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 747 "3369673591" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 750 "2598387317" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 753 "2571805770" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
Condition 756 "2624313215" "(((~command_atomicWord)) | ((~command_addr[2]))) 1 -1" (3 "10")
