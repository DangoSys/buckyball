# `sg_dap`

PCM → mel `[1,80,3000] f32`.

Assigned entirely to slice `dap_0` / core `dap`.

Cut source: Buddy DAP `whisperPreprocess` / RFFT path (see buddy-mlir DAP + Whisper runner preprocess). Not part of the torch `forward` graph.
