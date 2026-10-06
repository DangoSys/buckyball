// Format Version: 2
// Reviewed for 389-bit DAT / 32-bit chunks / populated NodeIDs {1,2,3,64}, coordinated reset.

CHECKSUM: "3171296859 1964414076"
INSTANCE: codec_tb.dut.tx
ANNOTATION: "Assertion-failure diagnostic only; assertion remains enabled: CHI Mesh target NodeID is not placed"
Block 2 "3889754848" "if (1)"
ANNOTATION: "Assertion-failure diagnostic only; assertion remains enabled: CHI Mesh target NodeID is not placed"
Block 3 "2159607100" "$error(\"Assertion failed: CHI Mesh target NodeID is not placed\n    at ChiMesh.scala:81 when(io.in.valid) { assert(placed, \\"CHI Mesh target NodeID is not placed\\") }\n\");"
ANNOTATION: "Assertion-failure diagnostic only; assertion remains enabled: CHI Mesh target NodeID is not placed"
Block 5 "3889754848" "if (1)"
ANNOTATION: "Assertion-failure diagnostic only; assertion remains enabled: CHI Mesh target NodeID is not placed"
Block 6 "4027875381" "$fatal;"

CHECKSUM: "1704682488 3830468848"
INSTANCE: codec_tb.dut.rx
ANNOTATION: "Assertion-failure diagnostic only; assertion remains enabled: CHI mesh packet started without head"
Block 2 "4264478414" "if (1)"
ANNOTATION: "Assertion-failure diagnostic only; assertion remains enabled: CHI mesh packet started without head"
Block 3 "1178357294" "$error(\"Assertion failed: CHI mesh packet started without head\n    at ChiMesh.scala:133 assert(io.in.bits.head, \\"CHI mesh packet started without head\\")\n\");"
ANNOTATION: "Assertion-failure diagnostic only; assertion remains enabled: CHI mesh packet started without head"
Block 5 "4264478414" "if (1)"
ANNOTATION: "Assertion-failure diagnostic only; assertion remains enabled: CHI mesh packet started without head"
Block 6 "520366355" "$fatal;"
ANNOTATION: "Assertion-failure diagnostic only; assertion remains enabled: CHI mesh packet has a second head"
Block 10 "991107315" "if (1)"
ANNOTATION: "Assertion-failure diagnostic only; assertion remains enabled: CHI mesh packet has a second head"
Block 11 "1926190078" "$error(\"Assertion failed: CHI mesh packet has a second head\n    at ChiMesh.scala:135 assert(!io.in.bits.head, \\"CHI mesh packet has a second head\\")\n\");"
ANNOTATION: "Assertion-failure diagnostic only; assertion remains enabled: CHI mesh packet has a second head"
Block 13 "991107315" "if (1)"
ANNOTATION: "Assertion-failure diagnostic only; assertion remains enabled: CHI mesh packet has a second head"
Block 14 "1882298097" "$fatal;"
ANNOTATION: "Assertion-failure diagnostic only; assertion remains enabled: CHI mesh packet has wrong length"
Block 18 "1840269416" "if (1)"
ANNOTATION: "Assertion-failure diagnostic only; assertion remains enabled: CHI mesh packet has wrong length"
Block 19 "74803256" "$error(\"Assertion failed: CHI mesh packet has wrong length\n    at ChiMesh.scala:137 assert(io.in.bits.tail === (beat === (chunks - 1).U), \\"CHI mesh packet has wrong length\\")\n\");"
ANNOTATION: "Assertion-failure diagnostic only; assertion remains enabled: CHI mesh packet has wrong length"
Block 21 "1840269416" "if (1)"
ANNOTATION: "Assertion-failure diagnostic only; assertion remains enabled: CHI mesh packet has wrong length"
Block 22 "1745333983" "$fatal;"

CHECKSUM: "3171296859 4271561183"
INSTANCE: codec_tb.dut.tx
ANNOTATION: "NodeID rejection failure vector; a normally accepted request must target populated NodeIDs {1,2,3,64}; bad_node checks rejection separately"
Condition 1 "690909136" "(io_in_valid & ((~reset)) & ((~_GEN_1[io_in_bits_targetNode]))) 1 -1" (4 "111")

CHECKSUM: "1704682488 2938369825"
INSTANCE: codec_tb.dut.rx
ANNOTATION: "First accepted fragment has HEAD; shared reset keeps both codec counters aligned, including the reset-disabled failure vector"
Condition 1 "2778753700" "(_GEN & ((~collecting)) & ((~reset)) & ((~io_in_bits_head))) 1 -1" (3 "1101")

CHECKSUM: "1704682488 2938369825"
INSTANCE: codec_tb.dut.rx
ANNOTATION: "First accepted fragment has HEAD; shared reset keeps both codec counters aligned, including the reset-disabled failure vector"
Condition 1 "2778753700" "(_GEN & ((~collecting)) & ((~reset)) & ((~io_in_bits_head))) 1 -1" (5 "1111")

CHECKSUM: "1704682488 2938369825"
INSTANCE: codec_tb.dut.rx
ANNOTATION: "During collection packetizer HEAD remains false even if paused; shared reset cancels both endpoints together"
Condition 2 "142507406" "(_GEN & collecting & ((~reset)) & io_in_bits_head) 1 -1" (1 "0111")

CHECKSUM: "1704682488 2938369825"
INSTANCE: codec_tb.dut.rx
ANNOTATION: "During collection packetizer HEAD remains false even if paused; shared reset cancels both endpoints together"
Condition 2 "142507406" "(_GEN & collecting & ((~reset)) & io_in_bits_head) 1 -1" (3 "1101")

CHECKSUM: "1704682488 2938369825"
INSTANCE: codec_tb.dut.rx
ANNOTATION: "During collection packetizer HEAD remains false even if paused; shared reset cancels both endpoints together"
Condition 2 "142507406" "(_GEN & collecting & ((~reset)) & io_in_bits_head) 1 -1" (5 "1111")

CHECKSUM: "1704682488 2938369825"
INSTANCE: codec_tb.dut.rx
ANNOTATION: "Packetizer and receiver counters advance on exactly the same handshake; accepted TAIL always matches beat 12, including coordinated reset"
Condition 3 "1479293268" "(_GEN & ((~reset)) & (io_in_bits_tail != (beat == 4'hc))) 1 -1" (2 "101")

CHECKSUM: "1704682488 2938369825"
INSTANCE: codec_tb.dut.rx
ANNOTATION: "Packetizer and receiver counters advance on exactly the same handshake; accepted TAIL always matches beat 12, including coordinated reset"
Condition 3 "1479293268" "(_GEN & ((~reset)) & (io_in_bits_tail != (beat == 4'hc))) 1 -1" (4 "111")

CHECKSUM: "1704682488 2938369825"
INSTANCE: codec_tb.dut.rx
ANNOTATION: "Stored words are beats 0 through 11; legal TAIL can only accompany beat 12, so tail-at-stored-word vector is unreachable"
Condition 10 "291367532" "(_GEN & ((~io_in_bits_tail)) & (beat == 4'b0)) 1 -1" (2 "101")

CHECKSUM: "1704682488 2938369825"
INSTANCE: codec_tb.dut.rx
ANNOTATION: "Stored words are beats 0 through 11; legal TAIL can only accompany beat 12, so tail-at-stored-word vector is unreachable"
Condition 12 "684678524" "(_GEN & ((~io_in_bits_tail)) & (beat == 4'b1)) 1 -1" (2 "101")

CHECKSUM: "1704682488 2938369825"
INSTANCE: codec_tb.dut.rx
ANNOTATION: "Stored words are beats 0 through 11; legal TAIL can only accompany beat 12, so tail-at-stored-word vector is unreachable"
Condition 14 "3689188981" "(_GEN & ((~io_in_bits_tail)) & (beat == 4'h2)) 1 -1" (2 "101")

CHECKSUM: "1704682488 2938369825"
INSTANCE: codec_tb.dut.rx
ANNOTATION: "Stored words are beats 0 through 11; legal TAIL can only accompany beat 12, so tail-at-stored-word vector is unreachable"
Condition 16 "3799398757" "(_GEN & ((~io_in_bits_tail)) & (beat == 4'h3)) 1 -1" (2 "101")

CHECKSUM: "1704682488 2938369825"
INSTANCE: codec_tb.dut.rx
ANNOTATION: "Stored words are beats 0 through 11; legal TAIL can only accompany beat 12, so tail-at-stored-word vector is unreachable"
Condition 18 "2405101270" "(_GEN & ((~io_in_bits_tail)) & (beat == 4'h4)) 1 -1" (2 "101")

CHECKSUM: "1704682488 2938369825"
INSTANCE: codec_tb.dut.rx
ANNOTATION: "Stored words are beats 0 through 11; legal TAIL can only accompany beat 12, so tail-at-stored-word vector is unreachable"
Condition 20 "3066583494" "(_GEN & ((~io_in_bits_tail)) & (beat == 4'h5)) 1 -1" (2 "101")

CHECKSUM: "1704682488 2938369825"
INSTANCE: codec_tb.dut.rx
ANNOTATION: "Stored words are beats 0 through 11; legal TAIL can only accompany beat 12, so tail-at-stored-word vector is unreachable"
Condition 22 "3217531303" "(_GEN & ((~io_in_bits_tail)) & (beat == 4'h6)) 1 -1" (2 "101")

CHECKSUM: "1704682488 2938369825"
INSTANCE: codec_tb.dut.rx
ANNOTATION: "Stored words are beats 0 through 11; legal TAIL can only accompany beat 12, so tail-at-stored-word vector is unreachable"
Condition 24 "2253723319" "(_GEN & ((~io_in_bits_tail)) & (beat == 4'h7)) 1 -1" (2 "101")

CHECKSUM: "1704682488 2938369825"
INSTANCE: codec_tb.dut.rx
ANNOTATION: "Stored words are beats 0 through 11; legal TAIL can only accompany beat 12, so tail-at-stored-word vector is unreachable"
Condition 26 "759940555" "(_GEN & ((~io_in_bits_tail)) & (beat == 4'h8)) 1 -1" (2 "101")

CHECKSUM: "1704682488 2938369825"
INSTANCE: codec_tb.dut.rx
ANNOTATION: "Stored words are beats 0 through 11; legal TAIL can only accompany beat 12, so tail-at-stored-word vector is unreachable"
Condition 28 "349795035" "(_GEN & ((~io_in_bits_tail)) & (beat == 4'h9)) 1 -1" (2 "101")

CHECKSUM: "1704682488 2938369825"
INSTANCE: codec_tb.dut.rx
ANNOTATION: "Stored words are beats 0 through 11; legal TAIL can only accompany beat 12, so tail-at-stored-word vector is unreachable"
Condition 30 "811415760" "(_GEN & ((~io_in_bits_tail)) & (beat == 4'ha)) 1 -1" (2 "101")

CHECKSUM: "1704682488 2938369825"
INSTANCE: codec_tb.dut.rx
ANNOTATION: "Stored words are beats 0 through 11; legal TAIL can only accompany beat 12, so tail-at-stored-word vector is unreachable"
Condition 32 "961724593" "(_GEN & ((~io_in_bits_tail)) & (beat == 4'hb)) 1 -1" (2 "101")

CHECKSUM: "2536693556 3895022188"
INSTANCE: codec_tb.dut
ANNOTATION: "Codec local injection coordinate is fixed (0,0) in this declared top"
Toggle io_observed_srcX "net io_observed_srcX"

CHECKSUM: "2536693556 3895022188"
INSTANCE: codec_tb.dut
ANNOTATION: "Codec local injection coordinate is fixed (0,0) in this declared top"
Toggle io_observed_srcY "net io_observed_srcY"

CHECKSUM: "2536693556 3895022188"
INSTANCE: codec_tb.dut
ANNOTATION: "This codec verification top transports only CHI DAT, VC2"
Toggle io_observed_vc "net io_observed_vc[1:0]"

CHECKSUM: "3171296859 1555586989"
INSTANCE: codec_tb.dut.tx
ANNOTATION: "Elaboration-produced coordinate/presence lookup table consists entirely of literal constants; no datapath object selected by generated-name prefix"
Toggle _GEN "net _GEN[127:0]"

CHECKSUM: "3171296859 1555586989"
INSTANCE: codec_tb.dut.tx
ANNOTATION: "Elaboration-produced coordinate/presence lookup table consists entirely of literal constants; no datapath object selected by generated-name prefix"
Toggle _GEN_0 "net _GEN_0[127:0]"

CHECKSUM: "3171296859 1555586989"
INSTANCE: codec_tb.dut.tx
ANNOTATION: "Elaboration-produced coordinate/presence lookup table consists entirely of literal constants; no datapath object selected by generated-name prefix"
Toggle _GEN_1 "net _GEN_1[127:0]"

CHECKSUM: "3171296859 1555586989"
INSTANCE: codec_tb.dut.tx
ANNOTATION: "Only NodeIDs 1,2,3,64 are admitted: accepted target bits 5 through 2 are always zero; bits 6,1,0 remain measured"
Toggle target [5:2] "reg target[6:0]"

CHECKSUM: "3171296859 1555586989"
INSTANCE: codec_tb.dut.tx
ANNOTATION: "Only the upper 27 bits of the 416-bit zero-extended 389-bit flit are excluded; all useful bits remain checked"
Toggle _io_out_bits_payload_T_1 [415:389] "net _io_out_bits_payload_T_1[415:0]"
