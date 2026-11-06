# Cover Studio

Turn a recording into covers in other styles, sung with the song's real lyrics and phrasing.
All AI runs on [Modal](https://modal.com); the local app is a small FastAPI server and a plain web UI.

## What happens to a song

1. **Vocal isolation + speech recognition** (Modal, L4): Demucs `htdemucs_ft` separates the lead
   vocal; Whisper large-v3 (faster-whisper) transcribes it with word timestamps.
2. **Melody** (Modal, L4, in parallel): SheetSage2 transcribes the full recording into a
   melody-only ABC score (vocal and instrumental lead, no chords) plus the song's section structure.
3. **Official lyrics**: looked up on [LRCLIB](https://lrclib.net) from the artist and title
   (parsed from an `Artist - Title` filename, editable in the UI). The match whose length is closest
   to your file is used automatically; you can pick another, search again, or paste lyrics.
   LRCLIB's line time codes are kept.
4. **Word timing** (Modal): every official word is placed on the isolated vocal:
   - lines are anchored by the LRC time codes (and Whisper's words where they agree), and a line
     whose time code sits next to a clear silence-then-attack in the vocal is pinned to that frame;
   - MMS forced alignment (torchaudio `MMS_FA`) places each word inside its line's window, with
     star tokens absorbing ad-libs; a line that comes back crammed is retried with a wider window;
   - **word starts** are moved onto the real vocal attack (consonant onset); a line's held first
     vowel ("Buuurning") reaches back to the attack that began it;
   - **word ends** are extended through the sustained note until the singer releases it (a sharp
     energy fall, a 20 dB drop below the word's peak, or the next word); CTC alone ends sung words
     at the first frames of the vowel;
   - backing-vocal echoes written in `(parentheses)` are aligned as a second voice that may
     overlap the next lead line;
   - anything doubtful (no acoustic match, far from its time code, disagrees with Whisper) is
     shown in orange for review. Each word's start and end can be nudged separately and auditioned.
   - The result downloads as an enhanced (word-level) LRC.
5. **Sung phrasing**: the text YuE2 sings is built from the official lines (one phrase per line)
   grouped into `[Verse]`, `[Pre-Chorus]`, `[Chorus]`, `[Bridge]`… by SheetSage2's structure.
   It is shown and fully editable before generation.
6. **Covers** (Modal, L40S): YuE2-3B + YuE2-Vae generate one take per selected style, conditioned
   on the source melody (`cot="melody"`), up to three in parallel. Each take is saved the moment it
   finishes.
7. **Lyric check**: each take's vocal is isolated and transcribed again; the UI shows what share of
   the lyrics was actually heard, and the take gets its own word-timed LRC.

Styles are 16 presets in `app/styles.json` (genre, instruments, vocal character and mix), plus an
optional free-text style per batch.

## Setup (Windows)

```powershell
uv venv .venv --python 3.12
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
.venv\Scripts\modal.exe token new          # once, if this PC has no Modal token
cd cloud; ..\.venv\Scripts\modal.exe deploy cover_ai.py; cd ..
.\start.ps1                                 # opens http://127.0.0.1:8420
```

Models download into the `cover-studio-models` Modal Volume on first use (YuE2 ~8 GB). Songs and
takes are stored under `songs/<id>/` (`COVER_STUDIO_SONGS` overrides). `COVER_STUDIO_PARALLEL`
sets how many takes generate at once (default 3).

Rough cost per song on Modal: analysis + alignment about 1–2 GPU-minutes on an L4; each cover
take several minutes on an L40S plus about a minute of lyric checking.

## Tests

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The alignment edge logic, LRC parsing, phrasing and lyric scoring are pure Python/numpy and are
tested locally; the models only run on Modal.

## Licenses

App code: Apache-2.0. YuE2, SheetSage2 and MERT weights are **CC BY-NC 4.0 (non-commercial)**.
Lyrics, recordings and generated covers of third-party songs carry their own rights. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
