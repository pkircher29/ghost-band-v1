# Cover Studio

The style library is searchable by name, instrument, and mood. Select styles
before preparing a recording, inspect each **sound recipe**, and use **Selected
only** to review your batch. Filtering never removes a selection; **Select
visible** adds the current results and **Clear selection** clears the whole batch.
The desktop workspace places song preparation beside the style library and stacks
the panels on smaller screens.

The catalog includes 12 additional arrangements: Memphis Soul, Bossa Nova,
Afrobeat Ensemble, Trip-Hop, Western Swing, Roots Reggae, Neo-Soul, Disco House,
Cinematic Chamber Pop, Dreamy Shoegaze, Tango Nuevo, and New Orleans Second Line.
These are generation prompts, not guaranteed audio outcomes; listen to generated
takes to assess how well each arrangement fits your song.

Run the dependency-free style search tests with
`node --test tests/style-library.test.cjs` and the Python suite with
`python -m unittest discover -s tests` in the application's Python environment.

The main lyrics box includes editable section labels and line breaks. Save lyrics or Make covers preserves your exact version in the song session; automatic sectioning never replaces a reviewed version. Changing section labels alone preserves word timings. Adding, deleting, or reordering sung words may require realignment. Timed-word correction ignores section labels when locating words.

A local, one-click song-cover generator built on [YuE2](https://huggingface.co/m-a-p/YuE2-3B) via the
[audio.cpp](https://github.com/0xShug0/audio.cpp) inference engine.

Drop in a recording and it:
1. Separates vocals with Demucs `htdemucs_ft`, preserving the original duration and zero point.
2. Transcribes lyrics and word timestamps with Whisper large-v3 from the isolated vocal stem.
3. Runs SheetSage2 once on the original full recording with `melody_only=True`. Uses its exported ABC directly, including both melody parts. No stem-note merging or custom grid repair runs. Saved sessions with older combined scores require Resume preparation, which reuses their vocals and reviewed lyrics. The direct score is stored under `scores/direct-song/` alongside retained older artifacts.
4. Builds sectioned generation lyrics from the score's section boundaries and saved word onsets, with line breaks at pauses. The lyrics panel previews this exact input. Saved words and timings remain unchanged; generation history retains both the reviewed text and sectioned input. Manually sectioned lyrics are preserved. Missing or stale timing requires review rather than guessed sections. The default `melody` mode retains both melody voices without chords and leaves the arrangement free, following the [YuE2 cover workflow](https://github.com/multimodal-art-projection/YuE/blob/main/docs/covers.md). Section labels guide phrasing; they do not enforce word-to-note timing.
5. Generates a finished cover for every style you multi-select from the presets — editable, and you can add
   your own, via the "Manage styles" panel (backed by `app/styles.json`)
6. Saves each finished take to `completed/` the instant it's done — a crash mid-batch never loses
   already-finished styles

Extras: a "Keep Original Style" baseline preset, a major/minor key-flip toggle (edits the transcribed
ABC score's key signature), a vocal gender hint (Any/Male/Female), and a live VRAM gauge in the topbar
(NVIDIA GPUs only) so you can see at a glance how close a generation is to running out of graphics
memory — see [SPEC.md](SPEC.md) for how it works.

## Requirements

- Windows, with [`audio.cpp`](https://github.com/0xShug0/audio.cpp) built separately (`dev` branch,
  `windows-vulkan-release` or `windows-cpu-release` preset — see its own docs). This repo does not
  vendor that engine.
- YuE2-3B GGUF weights (`audio-cpp/Yue2-3B-GGUF` on Hugging Face), placed under the engine's `models/` directory.
- Whisper large-v3 weights, downloaded on first use into the configured Whisper model directory.
- Python 3.12 for this app's own venv (`uv venv .venv`), with `transformers==4.45.2` pinned exactly —
  SheetSage2's custom modeling code breaks on transformers 5.x.
- `pip install -r requirements.txt` (torch from the CPU wheel index is enough; no GPU required for
  the app itself — YuE2 generation runs through the separately-built engine).

## GPU setup and running

### Vocal separation setup

Demucs runs in its own environment so its dependencies do not change SheetSage2.
From this directory (Windows Git Bash, with `uv.exe` on PATH):

```bash
uv.exe venv .venv-demucs --python 3.12
uv.exe pip install --python .venv-demucs/Scripts/python.exe torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu
uv.exe pip install --python .venv-demucs/Scripts/python.exe -r requirements-demucs.txt
```

The default is CPU. `DEMUCS_PYTHON` can point to another compatible environment;
`DEMUCS_DEVICE` explicitly selects another verified PyTorch device. Generation's
Vulkan setting does not enable GPU acceleration for Demucs. Model weights download
on first use into `models/demucs/` (or `TORCH_HOME`), and separation logs live in
the session's `stems/demucs.log`.

Preparation runs once per upload. Melody mode uses both vocal and instrumental
notes without chords; Full mode also uses the full mix's chords. A preparation
failure is shown explicitly and does not silently substitute the original mix
for the vocal stem or generate a cover without its requested melody.

Songs persist in `session-files/<song-name>-<id>/`. Each directory contains the
original recording, `stems/vocals.wav`, separation metadata/logs, both SheetSage2
analyses and combined scores under `scores/`, and an atomically written
`session.json` with reviewed lyrics, generation settings, and finished-cover
filenames. Use **Saved songs → Open song** to reuse these after an app restart.
**Save lyrics** saves corrections without generating. Failed preparation can be
resumed, skipping already completed separation and lyric transcription; a ready
song skips all analysis. Each new upload creates a separate session, even when
filenames match. `SESSION_FILES_DIR` can override the storage directory.

The original audio remains separate throughout preparation and generation.
YuE2 receives lyrics and the combined symbolic score, not an audio reference.
Source-relative note timings are preserved in the session's score artifacts;
word timing uses acoustic forced alignment (syllable alignment is not implemented). Song sessions are retained
after successful or failed generation. Finished WAVs remain in `completed/` and
the saved song lists links to them.

### Native generation engine

Use the patched [audio.cpp fork](https://github.com/pkircher29/audio.cpp) and follow
[GPU setup and verification](docs/GPU_SETUP.md). The original unpatched Vulkan
engine can abort during YuE2 generation.

Run `start.ps1` (or the existing desktop shortcut). The launcher selects a matching
native executable, checks the selected device and both servers, and opens the app
only when ready. The top bar shows the actual configured generation backend/device.
Local preferences can be saved in the ignored `launch.local.json` file.

Verified on Intel Arc B580 (12 GB): repeated 48 kHz stereo generation, including
approximately 60-second outputs in 27 seconds at 4 synthesis steps. This is a
runtime check, not a musical-quality benchmark or a guarantee for arbitrary song
lengths. The two GPU-operation regressions also pass on GTX 1650 SUPER; full
YuE2 generation on that 4 GB card is not claimed as verified.

## Credits and licenses

Powered by **[YuE / YuE2](https://github.com/multimodal-art-projection/YuE)** and its
original authors. See the [official project](https://map-yue2.github.io/) and
[YuE2 model](https://huggingface.co/m-a-p/YuE2-3B). Native inference is provided by
[audio.cpp](https://github.com/0xShug0/audio.cpp); melody transcription uses
[SheetSage2](https://huggingface.co/m-a-p/SheetSage2) and MERT-v2, and lyrics
transcription uses Whisper large-v3. This is an independent community interface.

Cover Studio code: [Apache-2.0](LICENSE). **YuE2, SheetSage2 and MERT-v2 model
weights: CC BY-NC 4.0 (noncommercial).** The code license does not grant commercial
permission for those models. Models and recordings are not included in this repo.
See [NOTICE](NOTICE) and [third-party credits and license links](THIRD_PARTY_NOTICES.md).
## Whisper large-v3 lyric timing

New songs use Whisper **large-v3** through faster-whisper, with word timestamps
on the isolated, untrimmed vocal stem. The CPU runtime uses INT8. Create an
isolated Python environment, install `requirements-whisper.txt`, and set
`whisper_python` and `whisper_models` in `launch.local.json` (or
`WHISPER_PYTHON` and `WHISPER_MODELS`). The first run downloads the model.

Open an existing saved song and choose **Use Whisper large-v3 lyrics + timing**
to add timestamps without repeating separation or melody analysis. This replaces
the displayed lyrics; previous text is retained in `lyrics-history.jsonl`.
`lyrics-whisper-large-v3.json` stores the reusable transcript and word intervals.
Click a timed word to seek the original recording; hover to see overlapping
SheetSage2 vocal pitches and intervals. Both use seconds from the original start.
Word timings are estimates, not syllable boundaries. Use **Edit** beside a timed
word to correct it without moving its start/end or melody-note links. Replacing
words in the main lyrics box also preserves timing when the word count remains
the same, whether saved directly or through **Make covers**. Inserting or deleting
words makes the previous timing stale. YuE2 still consumes lyrics and ABC; these timestamps
do not enforce exact timing in generated covers.

Whisper transcriptions now receive a second, acoustic timing pass using
torchaudio's `WAV2VEC2_ASR_BASE_960H` CTC aligner on the isolated vocals.
Existing songs can use **Tighten word timing** without retranscribing. The
aligner uses `ALIGNMENT_PYTHON` (or the existing `DEMUCS_PYTHON` environment
with torch/torchaudio 2.5.1 and soundfile). Optional launcher settings are
`alignment_python` and `alignment_models`; the latter controls model storage.
Original timings are preserved beside adjusted timings in
`lyrics-aligned-<fingerprint>.json`. Hover a word to compare boundaries.
Low scoring, conflicting, unsupported, or excessive adjustments retain the
original boundary and show **review**. The score thresholds are conservative
heuristics, not calibrated accuracy guarantees for singing. Corrected lyrics
must match the saved transcript before this refinement can run.
