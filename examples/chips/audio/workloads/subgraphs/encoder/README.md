# `sg_encoder`

mel `[1,80,3000] f32` → enc_states `[1,1500,512] f32`.

Homogeneous cut (lead):

- `enc_0`: conv stem + layers 0–1
- `enc_1`: layers 2–3
- `enc_2`: layers 4–5 + final norm

Import source: Whisper `forward` encoder half (see `import-whisper.py`). Lead owns emitting slice IR; this directory holds the stage contract until IR is checked in.
