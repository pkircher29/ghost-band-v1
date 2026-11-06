# Cover Studio — Technical Spec

## Architecture

```
Browser (app/static, no build step)
   │ HTTP (polling)
   ▼
FastAPI app (app/server.py, 127.0.0.1:8420) ── LRCLIB (official lyrics + line time codes)
   │ modal.Cls.from_name("cover-studio-ai", …).method.remote.aio(...)
   ▼
Modal app cover-studio-ai (cloud/cover_ai.py), weights in Volume cover-studio-models
   VocalLab (L4)   analyze_source · align_lyrics · check_take
   Melody   (L4)   transcribe  → SheetSage2 melody-only ABC + structure.lab
   Composer (L40S) generate    → YuE2-3B + YuE2-Vae, cot="melody"
```

`cloud/alignment.py` is shipped into the VocalLab image and is also imported by the local tests.

## Song storage (`songs/<id>/`)

| File | Written by |
|---|---|
| `song.json` | state: artist/title, stages{vocals,melody,lyrics,align}, lyrics summary, edited phrasing, takes[] |
| `source.<ext>` | upload |
| `vocals.flac`, `asr.json` | VocalLab.analyze_source |
| `melody.json`, `melody.mid` | Melody.transcribe |
| `lyrics-candidates.json`, `lyrics.json` | LRCLIB lookup / user choice (`lines`, `times`, `source`) |
| `aligned.json`, `lyrics.lrc` | VocalLab.align_lyrics, manual word edits |
| `takes/<take>.flac`, `.lrc`, `.check.json` | Composer.generate, VocalLab.check_take |

Writes are atomic (temp file + replace). On startup, anything left `running` is marked
`interrupted`; **Retry** resumes analysis and reuses finished steps.

## API

- `GET /api/health` — Modal app reachable.
- `GET /api/styles` — presets from `app/styles.json`.
- `GET /api/songs`, `POST /api/songs` (multipart `file`, optional `artist`, `title`),
  `GET|PATCH /api/songs/{id}`, `POST /api/songs/{id}/resume`.
- `GET /api/lyrics/search?artist=&title=&q=&duration=` — LRCLIB candidates, closest length first.
- `GET /api/songs/{id}/lyrics` — chosen lyrics, alignment, candidates.
- `PUT /api/songs/{id}/lyrics` `{lines,times,source,label}` or `{text}` — sets lyrics and re-aligns.
  Editing text keeps the time codes if the line count is unchanged.
- `PUT /api/songs/{id}/words` `[{index,start,end}]` — manual word edges; rewrites the LRC.
- `PUT /api/songs/{id}/generation-lyrics` `{lyrics}` — override the sectioned text (`null` resets).
- `POST /api/songs/{id}/covers` `{styles[], custom_style, vocal, seed, melody: source|free, cfg_scale}`.
- `DELETE /api/songs/{id}/takes/{take}`; `GET /api/songs/{id}/files/{path}`.

## Word timing (cloud/alignment.py)

1. `tokenize_lines`: words with spoken form (digits spelled out, `&`→and, accents stripped);
   words inside `( )` are `backing`.
2. `anchor_words`: difflib match of lyric words to Whisper words; lone short matches dropped,
   backwards-in-time matches dropped.
3. `estimate_lines`: line starts from LRC codes (shifted only by a measured offset ≥ 1 s — smaller
   disagreement is Whisper onset noise), else ASR anchors, else interpolation by word count.
4. `plan_line_windows`: one window per trusted line, `[start − 0.8 s, next start + 0.5 s]`;
   untrusted lines merge into their neighbour. `phrase_attack` pins a line to an attack that follows
   ≥ 40 ms of near-silence within ±0.6 s of its time code.
5. `ctc_align`: MMS_FA forced alignment per window, in order, each window starting after the previous
   line's last word (lines can't swap). Pinned windows drop the leading star so the first word must
   start at the attack. Crammed results (< 0.13 s/word) retry with ±1.5 s and ±3 s. Backing echoes
   are aligned afterwards in their own window and may overlap the next lead line.
6. `refine_edges` (lead words): per-frame activity threshold = max(silence floor + 10 dB, loudest
   vocal within ±1.5 s − 26 dB). Start → first frame of the attack (≤ 0.15 s back; line-initial words
   up to 1.6 s back through continuous sound to the attack). End → extended through the held note
   until a release (≥ 12 dB fall in 50 ms), a 20 dB drop below the word's peak, silence, or the next
   word; trimmed back if CTC ran into silence.
7. Review flags: no acoustic match, disagreement with Whisper (> 1.5 s with a weak match),
   interpolated, unusually long, or line start > 1.5 s from its time code.

Measured on My Chemical Romance — Helena (245 words, 43 lines): line starts median 0.19 s from the
human LRC time codes, 8 words flagged.

## Phrasing (app/phrasing.py)

Lines are assigned to the SheetSage2 structure segment they overlap most; consecutive lines in one
segment form a stanza tagged with its label. Without structure: stanzas split at ≥ 2.2 s vocal
silence, repeated stanzas become `[Chorus]`.
