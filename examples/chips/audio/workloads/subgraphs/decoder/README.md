# `sg_decoder`

One AR step: enc_states + tokens → logits (`vocab_size=51865`).

Homogeneous cut (lead):

- `dec_0`: decoder layers 0–2
- `dec_1`: decoder layers 3–5 + lm_head

Import source: Whisper `forward` decoder half. Full-window logits `[1,448,vocab]` in the reference importer may be narrowed to last-step logits in hardware contracts — mismatch must error, not silently truncate.
