# Audio chip workloads — slice contracts first; full Whisper E2E later.

add_custom_target(audio-slice-contracts
  COMMAND ${CMAKE_COMMAND} -E echo "audio slice contracts under workloads/slices/"
  COMMENT "Audio lead slice contracts (no binaries yet)"
  VERBATIM)

add_custom_target(chip-workloads-build ALL
  DEPENDS audio-slice-contracts
  COMMENT "Building audio workloads (slice contracts; cores supply binaries)"
  VERBATIM)
