# GPU setup and verification

## Compatible engine

Use [pkircher29/audio.cpp](https://github.com/pkircher29/audio.cpp), branch `ghost-band` (the fork's default).
The Vulkan fixes are commit `480280a` (based on upstream `87544b5`), also reviewed
in [engine PR 1](https://github.com/pkircher29/audio.cpp/pull/1).
Upstream attribution and licenses are retained in the fork.
This branch pins the installed engine baseline plus the verified fixes; upstream
`dev` has a different history and is not the build covered by these results.

Build `audiocpp_server` with the `windows-vulkan-release` preset using the
engine's documented Windows toolchain (MSVC, CMake/Ninja, Vulkan SDK). Build the
YuE2 and Qwen3-ASR model families, or use its full model set. The app does not
download or build the native engine automatically.

Place the engine beside this repository as `../audio.cpp`, or set `engine_dir`
in `launch.local.json`. Configure YuE2 and ASR in the engine's existing
`server.json`, with model IDs `yue2` and `asr`. The launcher reads this file and
writes a runtime copy under `.runtime/engine.json`; it preserves the original.

Minimal model entries (paths are relative to the engine directory):

```json
{
  "lazy_load": true,
  "idle_unload_ms": 120000,
  "threads": 16,
  "models": [
    {
      "id": "yue2", "family": "yue2", "task": "gen", "mode": "offline",
      "path": "models/Yue2-3B-GGUF",
      "session_options": {"yue2.model_gguf": "yue2-3b-q4_0.gguf"}
    },
    {
      "id": "asr", "family": "qwen3_asr", "task": "asr", "mode": "offline",
      "path": "models/Qwen3-ASR-0.6B-GGUF"
    }
  ]
}
```

## Select a GPU

List devices with the appropriate engine binary:

```text
../audio.cpp/build/windows-vulkan-release/bin/audiocpp_server.exe --list-devices
```

Example `launch.local.json` for the tested machine:

```json
{"backend": "vulkan", "device": "B580"}
```

`device` accepts a device index or unique name fragment. Name selection survives
device-number changes. Without a selector, the launcher prefers a discrete GPU
over integrated graphics. A missing/ambiguous requested device fails visibly.
It never silently falls back to CPU. CUDA/CPU require their matching build preset.

Run the existing `start.ps1` shortcut, or:

```text
.venv/Scripts/python.exe tools/launch.py --backend vulkan --device B580
```

The launcher binds only to localhost, checks ports before starting, waits for
both services, and stops only its own child processes on exit. Logs and selected
runtime config live in `.runtime/`. Stop the old launcher before starting again.
The default app URL is http://127.0.0.1:8420.

## Host memory settings

The launcher sets YuE2's weight metadata contexts to 64 MiB and graph metadata
arenas to 128 MiB, unless explicitly overridden in the model session options.
These are host-side GGML bookkeeping contexts with `no_alloc=true`, not the
model weights, GPU tensor buffers, audio duration, or inference precision.
Upstream defaults allocated many gigabytes of host metadata and exhausted
Windows commit during the failing runs. Only one model stays loaded at a time.

## Verified behavior (2026-09-15)

Intel Arc B580, 12 GB, Windows Vulkan, YuE2 Q4_0 and F16 VAE:

| Check | Result |
|---|---|
| 256-token cold GPU generation | 10.239 s stereo audio; 10.1 s elapsed |
| Uncapped generation, 4 synthesis steps | 59.959 s stereo audio; 27.4 s elapsed |
| Repeat uncapped generation | 59.959 s stereo audio; 26.7 s elapsed |
| Cold run through the normal desktop launcher | 59.959 s stereo audio; 70.9 s elapsed |
| Browser upload and GPU Qwen3-ASR | Transcribed a generated 10-second input |
| Browser melody-conditioned cover, normal 16 steps | 89.919 s stereo audio; 136 s engine time; 218 s including melody setup |
| Browser output playback | Loaded without media error; playback clock advanced |

All WAVs are 48 kHz stereo. Timing depends on cold loads, model download/cache,
lyrics, melody and available host memory. These checks establish successful
execution; they do not measure musical similarity or production quality.

`rope_position_view_test --vulkan` and `col2im_dispatch_test --vulkan` pass on
both Arc B580 and GTX 1650 SUPER. The RoPE CPU test also passes. Full YuE2 on
the 4 GB NVIDIA card and arbitrary multi-minute songs remain unverified.

Run an actual generation against the normally launched engine:

```text
.venv/Scripts/python.exe tools/gpu_smoke.py --url http://127.0.0.1:8180
.venv/Scripts/python.exe -m unittest discover -s tests -v
node --check app/static/app.js
```

The smoke test requires a GPU backend and saves a real WAV under `.verification/`.
The normal app reports backend and device in its header. Vulkan VRAM telemetry
is currently unavailable, so the NVIDIA-only gauge is hidden instead of showing
an unrelated card's memory. Model weights and recordings are excluded from Git.

See [third-party notices](../THIRD_PARTY_NOTICES.md) before installing models:
YuE2, SheetSage2 and MERT-v2 weights are licensed for noncommercial use.
