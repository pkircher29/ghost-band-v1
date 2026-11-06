# Credits and third-party licenses

Ghost Band's original integration code is licensed under [Apache-2.0](LICENSE).
This source repository does not distribute model weights, downloaded model
implementation files, Python packages, native executables, or recordings.
Those components retain their own licenses when installed separately.

## Music generation and transcription

| Component | Creator / original source | Applicable source or model license |
|---|---|---|
| YuE / YuE2 inference code | [YuE project and contributors](https://github.com/multimodal-art-projection/YuE) | [Apache-2.0 code license](https://github.com/multimodal-art-projection/YuE/blob/main/LICENSE) |
| YuE2-3B and YuE2 VAE weights | [M-A-P YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B), [YuE2-Vae](https://huggingface.co/m-a-p/YuE2-Vae), [legacy VAE](https://huggingface.co/m-a-p/YuE2-Vae-legacy) | [YuE2 model license: CC BY-NC 4.0](https://github.com/multimodal-art-projection/YuE/blob/main/MODEL_LICENSE) |
| YuE2 GGUF conversion | [audio-cpp/Yue2-3B-GGUF](https://huggingface.co/audio-cpp/Yue2-3B-GGUF), derived from M-A-P YuE2 | CC BY-NC 4.0, as identified in the conversion's model card; quantization does not remove the upstream weight license |
| SheetSage2 weights | [M-A-P SheetSage2](https://huggingface.co/m-a-p/SheetSage2) | CC BY-NC 4.0, per the model card |
| MERT-v2-FullSong weights | [M-A-P MERT-v2-FullSong](https://huggingface.co/m-a-p/MERT-v2-FullSong) | CC BY-NC 4.0, per the model card |
| Qwen3-ASR-0.6B | [Qwen team](https://huggingface.co/Qwen/Qwen3-ASR-0.6B), [Qwen3-ASR code](https://github.com/QwenLM/Qwen3-ASR) | Apache-2.0, per the source/model distribution |
| Demucs / htdemucs_ft vocal separation | [Meta Demucs and contributors](https://github.com/facebookresearch/demucs) | MIT, per the upstream distribution |
| audio.cpp | [0xShug0/audio.cpp and contributors](https://github.com/0xShug0/audio.cpp) | [Apache-2.0](https://github.com/0xShug0/audio.cpp/blob/dev/LICENSE); vendored components retain their own notices |

**Model use is noncommercial under the listed YuE2, SheetSage2 and MERT-v2
weight licenses.** See the full [CC BY-NC 4.0 terms](https://creativecommons.org/licenses/by-nc/4.0/).
The app's code license does not grant commercial permission for those weights.
Source recordings, lyrics and generated material have separate rights; this
repository does not grant rights to third-party songs or recordings.

For code and assets in official YuE distributions, retain their
[third-party notices](https://github.com/multimodal-art-projection/YuE/blob/main/THIRD_PARTY_NOTICES.md)
and [license texts](https://github.com/multimodal-art-projection/YuE/tree/main/licenses)
if redistributing those distributions. Ghost Band calls the native engine
over HTTP and loads separately installed SheetSage2 model code through Transformers.

## Runtime dependencies

The packages listed in `requirements.txt` are installed from their own
distributions. Preserve their included license and copyright files if building
an installer or distributing an environment. This source-only release does not
bundle those packages or FFmpeg. Consult the license of the particular FFmpeg
build you install; build options can change its redistribution requirements.

## Changes and provenance

The compatible [audio.cpp fork](https://github.com/pkircher29/audio.cpp) preserves
upstream history, its Apache-2.0 license, and vendored license files. Modified
engine files carry dated change notices. The local fixes copy misaligned RoPE
position views before Vulkan operations and cap COL2IM_1D workgroups using the
shader's existing loop. See `docs/GPU_SETUP.md` for the tested revision.

License/source links were checked on 2026-09-15. Check the notices shipped with
the exact model/dependency revisions you download. This is an independent
community interface; upstream names identify the work and do not imply endorsement.
# Whisper lyric transcription

Acoustic word-boundary refinement uses torchaudio's
`WAV2VEC2_ASR_BASE_960H` pretrained alignment model:
https://docs.pytorch.org/audio/2.5.0/generated/torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H.html.
CTC forced alignment uses torchaudio APIs; WhisperX itself is not installed.

Whisper large-v3: OpenAI, MIT license, https://huggingface.co/openai/whisper-large-v3.
CPU inference and word timestamps: faster-whisper, MIT license,
https://github.com/SYSTRAN/faster-whisper. CTranslate2 converted model:
https://huggingface.co/Systran/faster-whisper-large-v3.
