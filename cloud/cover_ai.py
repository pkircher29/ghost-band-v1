"""Cover Studio AI on Modal: every model call runs here, the local app only orchestrates.

Deploy:   .venv\\Scripts\\modal deploy cloud/cover_ai.py
Classes (looked up by the local app with modal.Cls.from_name("cover-studio-ai", ...)):
  VocalLab  (L4)    Demucs htdemucs_ft vocal isolation, Whisper large-v3 word ASR,
                    MMS_FA forced alignment of official lyrics, take QA (WER).
  Melody    (L4)    SheetSage2 melody-only ABC + song structure from the source.
  Composer  (L40S)  YuE2-3B + YuE2-Vae generation from style, lyrics and the ABC melody.
Model weights live in the `cover-studio-models` Volume (HF cache), fetched on first use.
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import modal

APP_NAME = 'cover-studio-ai'
YUE_COMMIT = os.environ.get('YUE_COMMIT', '1647252d68b70cbead046ffd3f7caf036dde47ee')
app = modal.App(APP_NAME)
models = modal.Volume.from_name('cover-studio-models', create_if_missing=True)
CACHE = '/models'
ENV = {'HF_HOME': f'{CACHE}/hf', 'TORCH_HOME': f'{CACHE}/torch', 'HF_HUB_ENABLE_HF_TRANSFER': '1',
       'PYTHONUNBUFFERED': '1'}

vocal_image = (
    modal.Image.from_registry('nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04', add_python='3.11')
    .apt_install('ffmpeg')
    .pip_install('torch==2.5.1', 'torchaudio==2.5.1', index_url='https://download.pytorch.org/whl/cu124')
    .pip_install('demucs==4.0.1', 'faster-whisper==1.1.1', 'soundfile==0.12.1', 'numpy<2',
                 'hf_transfer==0.1.8', 'huggingface_hub==0.26.5')
    .env(ENV)
    .add_local_python_source('alignment')
)

melody_image = (
    modal.Image.from_registry('ubuntu:24.04', add_python='3.11')
    .apt_install('ffmpeg', 'git')
    .pip_install('torch==2.8.0', 'torchaudio==2.8.0', index_url='https://download.pytorch.org/whl/cu126')
    .pip_install('transformers==4.45.2', 'huggingface-hub==0.36.0', 'safetensors==0.5.3', 'numpy==1.24.3',
                 'scipy==1.13.1', 'mir_eval==0.8.2', 'pretty_midi==0.2.10', 'mido==1.3.3',
                 'setuptools==78.1.1', 'soundfile==0.12.1', 'hf_transfer==0.1.8', 'accelerate==1.0.1')
    .env(ENV)
)

composer_image = (
    modal.Image.debian_slim(python_version='3.12')
    .apt_install('git', 'ffmpeg')
    .pip_install(f'git+https://github.com/multimodal-art-projection/YuE.git@{YUE_COMMIT}', 'hf_transfer==0.1.8')
    .env(ENV)
)


def decode(data: bytes, rate: int, channels: int):
    """Any audio container -> float32 numpy [samples, channels] via ffmpeg."""
    import numpy as np
    proc = subprocess.run(['ffmpeg', '-v', 'error', '-i', 'pipe:0', '-f', 'f32le', '-ac', str(channels),
                           '-ar', str(rate), 'pipe:1'], input=data, capture_output=True, check=True)
    return np.frombuffer(proc.stdout, dtype=np.float32).reshape(-1, channels).copy()


def flac_bytes(samples, rate) -> bytes:
    import soundfile as sf
    buffer = io.BytesIO()
    sf.write(buffer, samples, rate, format='FLAC', subtype='PCM_24')
    return buffer.getvalue()


@app.cls(image=vocal_image, gpu='L4', volumes={CACHE: models}, timeout=1800, scaledown_window=300)
class VocalLab:
    @modal.enter()
    def load(self):
        import torch
        import torchaudio
        from demucs.pretrained import get_model
        from faster_whisper import WhisperModel
        self.device = 'cuda'
        self.demucs = get_model('htdemucs_ft')
        self.demucs.to(self.device).eval()
        self.whisper = WhisperModel('large-v3', device='cuda', compute_type='float16',
                                    download_root=f'{CACHE}/whisper')
        bundle = torchaudio.pipelines.MMS_FA
        self.fa_model = bundle.get_model(with_star=True).to(self.device).eval()
        self.fa_dict = bundle.get_dict(star='*')
        self.fa_star = self.fa_dict['*']
        self.fa_rate = bundle.sample_rate
        models.commit()
        torch.backends.cudnn.benchmark = True

    def _isolate(self, audio_bytes: bytes):
        import torch
        from demucs.apply import apply_model
        mix = decode(audio_bytes, self.demucs.samplerate, 2)
        wav = torch.from_numpy(mix.T).to(self.device)
        ref = wav.mean(0)
        mean, std = ref.mean(), ref.std() + 1e-8
        with torch.inference_mode():
            stems = apply_model(self.demucs, ((wav - mean) / std)[None], device=self.device,
                                shifts=1, split=True, overlap=0.25, progress=False)[0]
        stems = stems * std + mean
        vocals = stems[self.demucs.sources.index('vocals')].T.cpu().numpy()
        return vocals, self.demucs.samplerate

    def _mono16k(self, vocals, rate):
        import torch
        import torchaudio.functional as AF
        mono = torch.from_numpy(vocals.mean(axis=1)).float()[None]
        return AF.resample(mono, rate, self.fa_rate)

    def _asr(self, mono16k, language=None):
        segments, info = self.whisper.transcribe(
            mono16k[0].numpy(), language=language, word_timestamps=True, vad_filter=False,
            condition_on_previous_text=False, beam_size=5, no_speech_threshold=None,
            hallucination_silence_threshold=2.0, compression_ratio_threshold=2.8)
        words, text = [], []
        for segment in segments:
            text.append(segment.text.strip())
            for w in segment.words or []:
                words.append({'text': w.word.strip(), 'start': round(w.start, 3), 'end': round(w.end, 3),
                              'probability': round(w.probability, 3)})
        return {'model': 'whisper-large-v3 (faster-whisper)', 'language': info.language,
                'text': ' '.join(text), 'words': words}

    @modal.method()
    def analyze_source(self, audio_bytes: bytes, language: str | None = None) -> dict:
        """Isolate the lead vocal and transcribe it with word timestamps."""
        t = time.time()
        vocals, rate = self._isolate(audio_bytes)
        separated = time.time() - t
        mono = self._mono16k(vocals, rate)
        asr = self._asr(mono, language)
        return {'vocals_flac': flac_bytes(vocals, rate), 'vocal_rate': rate,
                'duration': len(vocals) / rate, 'asr': asr,
                'timing': {'separation_s': round(separated, 1), 'total_s': round(time.time() - t, 1)}}

    @modal.method()
    def align_lyrics(self, vocals_flac: bytes, lines: list[str], asr_words: list[dict],
                     lrc_times: list | None = None) -> dict:
        """Force-align official lyric lines to the isolated vocal; refine word edges."""
        import alignment
        vocals = decode(vocals_flac, 44100, 2)
        mono = self._mono16k(vocals, 44100)
        result = alignment.align(self.fa_model, self.fa_dict, self.fa_star, mono, self.fa_rate,
                                 vocals.mean(axis=1), 44100, lines, asr_words, lrc_times,
                                 duration=len(vocals) / 44100, device=self.device)
        result['lrc'] = alignment.to_lrc(result['words'], lines)
        return result

    @modal.method()
    def check_take(self, take_bytes: bytes, lines: list[str], language: str | None = 'en') -> dict:
        """Did the generated take sing the right words? Isolate, ASR, WER, timed lyrics."""
        import alignment
        vocals, rate = self._isolate(take_bytes)
        mono = self._mono16k(vocals, rate)
        asr = self._asr(mono, language)
        score = alignment.lyric_check(lines, [w['text'] for w in asr['words']])
        timed = alignment.align(self.fa_model, self.fa_dict, self.fa_star, mono, self.fa_rate,
                                vocals.mean(axis=1), rate, lines, asr['words'],
                                duration=len(vocals) / rate, device=self.device)
        return {'asr_text': asr['text'], 'lyric_accuracy': score, 'words': timed['words'],
                'lrc': alignment.to_lrc(timed['words'], lines), 'stats': timed['stats']}


@app.cls(image=melody_image, gpu='L4', volumes={CACHE: models}, timeout=1800, scaledown_window=300)
class Melody:
    @modal.enter()
    def load(self):
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained('m-a-p/SheetSage2', trust_remote_code=True).eval().to('cuda')
        models.commit()

    @modal.method()
    def transcribe(self, audio_bytes: bytes, suffix: str = '.mp3') -> dict:
        """Melody-only ABC (vocal + instrumental lead, no chords) and song sections."""
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / f'source{suffix}'
            source.write_bytes(audio_bytes)
            out = Path(tmp) / 'score'
            t = time.time()
            result = self.model.transcribe(str(source), output_dir=str(out), dtype='bf16', melody_only=True)
            if result.get('abc_error') or not result.get('abc'):
                raise RuntimeError(f"SheetSage2 produced no usable ABC: {result.get('abc_error')}")
            labs = result.get('labs') or {}
            return {'abc': result['abc'], 'structure_lab': labs.get('structure', ''),
                    'key_lab': labs.get('key', ''), 'beat_lab': labs.get('beat', ''),
                    'warnings': [str(w) for w in result.get('warnings', [])],
                    'melody_mid': (out / 'melody.mid').read_bytes() if (out / 'melody.mid').is_file() else b'',
                    'seconds': round(time.time() - t, 1)}


@app.cls(image=composer_image, gpu='L40S', volumes={CACHE: models}, timeout=3600, scaledown_window=240)
class Composer:
    @modal.enter()
    def load(self):
        from yue2 import YuE2Pipeline
        self.pipe = YuE2Pipeline.from_pretrained('m-a-p/YuE2-3B', vae='m-a-p/YuE2-Vae', device='cuda',
                                                 progress=False, memory_budget_gib=40)
        models.commit()

    @modal.method()
    def generate(self, style: str, lyrics: str, abc: str | None = None, cot: str = 'melody',
                 seed: int = 831001, cfg_scale: float | None = None, request_id: str = 'cover') -> dict:
        kwargs = dict(id=request_id, style=style, lyrics=lyrics, cot=cot, seed=int(seed))
        if abc and cot != 'off':
            kwargs['abc'] = abc
        if cfg_scale:
            kwargs['cfg_scale'] = float(cfg_scale)
        t = time.time()
        song = self.pipe(**kwargs)
        return {'flac': flac_bytes(song.audio, song.sample_rate), 'sample_rate': song.sample_rate,
                'seconds': round(time.time() - t, 1), 'audio_seconds': round(len(song.audio) / song.sample_rate, 1),
                'abc': song.abc, 'truncated': song.truncated,
                'request': kwargs | {'abc': bool(kwargs.get('abc'))}}


@app.local_entrypoint()
def smoke(path: str, mode: str = 'vocal'):
    """modal run cloud/cover_ai.py --path song.mp3 --mode vocal|melody"""
    data = Path(path).read_bytes()
    if mode == 'vocal':
        r = VocalLab().analyze_source.remote(data)
        print(json.dumps(r['asr'], indent=1)[:3000], r['timing'])
    else:
        r = Melody().transcribe.remote(data, Path(path).suffix)
        print(r['abc'][:2000], r['structure_lab'], r['warnings'])
