"""Ghost Band app server.

Bridges a plain browser UI to two local engines:
- audiocpp_server (native, Vulkan-accelerated) for YuE2 music generation.
- an in-process CPU transcription helper for SheetSage2, whose GGUF port has
  no published weights yet, so it runs the original PyTorch model instead.
"""
from __future__ import annotations

import asyncio
import base64
import binascii
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import psutil
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from starlette.concurrency import run_in_threadpool

import sheetsage2_transcribe
import vocal_separation
import whisper_transcribe
import session_store
import cover_lyrics
from session_store import Session

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ghost_band")

AUDIOCPP_URL = os.environ.get("AUDIOCPP_URL", "http://127.0.0.1:8180")
YUE2_MODEL_ID = os.environ.get("YUE2_MODEL_ID", "yue2")
ASR_MODEL_ID = os.environ.get("ASR_MODEL_ID", "asr")
ENGINE_LOG_PATH = Path(os.environ.get("ENGINE_LOG_PATH", "D:/GITHUB/audio.cpp/engine_live.log"))
ENGINE_PROCESS_NAME = "audiocpp_server.exe"
COMPLETED_DIR = Path(os.environ.get("COMPLETED_DIR", Path(__file__).parent.parent / "completed"))
COMPLETED_DIR.mkdir(parents=True, exist_ok=True)

# Style presets covering distinct genres, so a cover run can fan out across
# several very different takes in one pass. "Keep Original Style" is a
# baseline take: no genre transform, just a faithful re-render.
#
# Presets live in styles.json next to this file (human-readable, hand-editable
# JSON) rather than hardcoded here, so they can be edited directly or through
# the "Manage styles" panel in the UI without touching code.
STYLES_PATH = Path(__file__).parent / "styles.json"


def _load_styles() -> list[dict]:
    if STYLES_PATH.exists():
        try:
            return json.loads(STYLES_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("styles.json unreadable (%s), starting from an empty list", exc)
    return []


def _save_styles(presets: list[dict]) -> None:
    STYLES_PATH.write_text(json.dumps(presets, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _rebuild_styles_by_id() -> None:
    STYLE_PRESETS_BY_ID.clear()
    STYLE_PRESETS_BY_ID.update({s["id"]: s for s in STYLE_PRESETS})


def _slugify(label: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    return slug or "style"


def _unique_style_id(label: str) -> str:
    base = _slugify(label)
    slug = base
    n = 2
    while slug in STYLE_PRESETS_BY_ID:
        slug = f"{base}_{n}"
        n += 1
    return slug


STYLE_PRESETS: list[dict] = _load_styles()
STYLE_PRESETS_BY_ID: dict[str, dict] = {}
_rebuild_styles_by_id()

app = FastAPI(title="Ghost Band")

# Stage names in pipeline order. audio.cpp logs a distinct marker string per
# stage transition; a cold model load only happens on the first request after
# an idle-unload, so the frontend treats any stage it never sees as skipped
# rather than stalled.
STAGE_SEQUENCE = [
    "Reading lyrics & style",
    "Loading model into memory",
    "Performing the song (AR)",
    "Synthesizing audio (NAR)",
    "Decoding to audio",
    "Finishing up",
]
STAGE_MARKERS = [
    ("yue2.plan", "Reading lyrics & style"),
    ("ar.prefix_weights_upload_ms", "Loading model into memory"),
    ("ar.weights_load_ms", "Loading model into memory"),
    ("ar.cfg.start_decode_ms", "Performing the song (AR)"),
    ("ar.batched_decode", "Performing the song (AR)"),
    ("yue2.semantic", "Performing the song (AR)"),
    ("yue2.nar", "Synthesizing audio (NAR)"),
    ("oobleck_audio_vae", "Decoding to audio"),
    ("yue2.vae_decode", "Decoding to audio"),
    ("session.wall_ms", "Finishing up"),
]


def _stage_name_for_line(line: str) -> str | None:
    for marker, name in STAGE_MARKERS:
        if marker in line:
            return name
    return None


def _engine_process() -> psutil.Process | None:
    if os.environ.get("ENGINE_PID"):
        try:
            proc = psutil.Process(int(os.environ["ENGINE_PID"]))
            return proc if proc.name().lower() == ENGINE_PROCESS_NAME.lower() else None
        except (ValueError, psutil.NoSuchProcess, psutil.AccessDenied):
            return None
    for proc in psutil.process_iter(["name"]):
        try:
            if proc.info["name"] and proc.info["name"].lower() == ENGINE_PROCESS_NAME.lower():
                return proc
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None


_NVIDIA_SMI_PATH: str | None | bool = False  # False = not yet searched


def _find_nvidia_smi() -> str | None:
    """Locate nvidia-smi.exe, caching the result. Its usual home
    (C:\\Program Files\\NVIDIA Corporation\\NVSMI) isn't always present or on
    PATH; the driver always ships a copy under System32's DriverStore, but
    that path fragment (e.g. nvmdsi.inf_amd64_<hash>) changes across driver
    updates, so it has to be globbed rather than hardcoded."""
    global _NVIDIA_SMI_PATH
    if _NVIDIA_SMI_PATH is not False:
        return _NVIDIA_SMI_PATH  # type: ignore[return-value]

    candidates = [
        Path(r"C:\Windows\System32\nvidia-smi.exe"),
        Path(r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"),
    ]
    for c in candidates:
        if c.exists():
            _NVIDIA_SMI_PATH = str(c)
            return _NVIDIA_SMI_PATH

    driver_store = Path(r"C:\Windows\System32\DriverStore\FileRepository")
    if driver_store.is_dir():
        for hit in driver_store.glob("*/nvidia-smi.exe"):
            _NVIDIA_SMI_PATH = str(hit)
            return _NVIDIA_SMI_PATH

    _NVIDIA_SMI_PATH = None
    return None


@app.get("/api/gpu/vram")
async def gpu_vram():
    # NVIDIA's first device is not evidence of the active Vulkan GPU's memory.
    # Until device-specific telemetry is available, hide the gauge for Vulkan/CPU.
    state = await health()
    if not state.get("ok") or state.get("backend") != "cuda":
        return {"available": False}
    smi = _find_nvidia_smi()
    if smi is None:
        return {"available": False}
    try:
        proc = await asyncio.create_subprocess_exec(
            smi,
            "--query-gpu=name,memory.used,memory.total",
            "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5.0)
        lines = stdout.decode("utf-8", "ignore").strip().splitlines()
        matching = [line for line in lines if line.split(",")[0].strip() == state.get("device")]
        if len(matching) != 1:
            return {"available": False}
        line = matching[0]
        name, used, total = [p.strip() for p in line.split(",")]
        return {
            "available": True,
            "name": name,
            "used_mib": int(used),
            "total_mib": int(total),
        }
    except Exception as exc:  # nvidia-smi missing, no NVIDIA GPU, parse failure, etc.
        log.warning("gpu vram query failed: %s", exc)
        return {"available": False}


@dataclass
class Job:
    id: str
    kind: str  # "generate" | "transcribe"
    status: str = "running"  # running | done | error
    stage: str = "Starting"
    error: str | None = None
    result: object | None = None
    started_at: float = field(default_factory=time.monotonic)


JOBS: dict[str, Job] = {}


class GenerateRequest(BaseModel):
    lyrics: str
    style: str
    cot: str = "melody"
    abc: str = ""
    seed: int = 831001
    num_inference_steps: int = 16
    cfg_scale: float | None = None


@app.get("/api/health")
async def health():
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            res = await client.get(f"{AUDIOCPP_URL}/health")
            res.raise_for_status()
            data = res.json()
            models_res = await client.get(f"{AUDIOCPP_URL}/v1/models")
            models_res.raise_for_status()
            models = models_res.json().get("data", [])
    except Exception as exc:  # engine not up yet, or unreachable
        log.warning("audiocpp_server health check failed: %s", exc)
        return {"ok": False}

    yue2 = next((m for m in models if m.get("id") == YUE2_MODEL_ID), None)
    backend = data.get("backend", "unknown")
    device = os.environ.get("ENGINE_DEVICE_NAME") if backend == os.environ.get("ENGINE_BACKEND") else None
    return {"ok": yue2 is not None, "yue2_loaded": bool(yue2 and yue2.get("loaded")),
            "backend": backend, "device": device}


async def _run_transcribe_job(job: Job, audio_path: str, tmp_dir: str) -> None:
    job.stage = "Transcribing melody"
    try:
        job.result = await run_in_threadpool(
            sheetsage2_transcribe.transcribe, audio_path, tmp_dir
        )
        job.status = "done"
        job.stage = "Done"
    except RuntimeError as exc:
        job.status = "error"
        job.error = str(exc)
    except Exception as exc:  # noqa: BLE001 - surfaced to the client as-is
        log.exception("transcription failed")
        job.status = "error"
        job.error = f"Transcription error: {exc}"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.post("/api/transcribe")
async def transcribe(file: UploadFile = File(...)):
    tmp_dir = tempfile.mkdtemp(prefix="ghost-band-")
    suffix = Path(file.filename or "source.wav").suffix or ".wav"
    source_path = Path(tmp_dir) / f"source{suffix}"
    source_path.write_bytes(await file.read())

    job = Job(id=uuid.uuid4().hex, kind="transcribe")
    JOBS[job.id] = job
    asyncio.create_task(_run_transcribe_job(job, str(source_path), tmp_dir))
    return {"job_id": job.id}


def _extract_audio_bytes(content_type: str, body: bytes) -> bytes:
    """audio.cpp's generic /v1/tasks/run response shape for a gen-task model
    isn't documented; handle the shapes it's known to return."""
    if content_type.startswith("audio/"):
        return body

    if content_type.startswith("application/json"):
        data = json.loads(body)
        for key in ("audio_base64", "audio_b64"):
            if data.get(key):
                return base64.b64decode(data[key])

        # audio.cpp's /v1/tasks/run wraps a gen-task result as a plain
        # base64-encoded WAV string under "audio", alongside informational
        # "sample_rate"/"channels"/"timing" fields.
        audio = data.get("audio")
        if isinstance(audio, str) and audio:
            return base64.b64decode(audio)
        if isinstance(audio, dict) and audio.get("data"):
            payload = audio["data"]
            if payload.startswith("data:"):
                payload = payload.split(",", 1)[1]
            return base64.b64decode(payload)

        for key in ("output_path", "path", "artifact_path", "out"):
            if data.get(key):
                candidate = Path(data[key])
                if candidate.exists():
                    return candidate.read_bytes()

        raise HTTPException(
            status_code=502,
            detail=f"Unrecognized audio.cpp response shape: {list(data.keys())}",
        )

    raise HTTPException(
        status_code=502, detail=f"Unexpected content type from audio.cpp: {content_type}"
    )


def _build_generate_body(lyrics: str, style: str, cot: str, abc: str, seed: int,
                          num_inference_steps: int, cfg_scale: float | None = None) -> dict:
    # audio.cpp's /v1/tasks/run only special-cases a handful of top-level
    # fields (lyrics, seed, num_inference_steps, ...); everything else,
    # including style/cot/abc/cfg_scale, must be nested under "options".
    options: dict = {"style": style, "cot": cot}
    if abc and cot != "off":
        options["abc"] = abc
    if cfg_scale is not None:
        options["cfg_scale"] = cfg_scale
    request_body = {
        "lyrics": lyrics,
        "seed": seed,
        "num_inference_steps": num_inference_steps,
        "options": options,
    }
    return {"model": YUE2_MODEL_ID, "request": request_body}


async def _call_yue2_generate(body: dict, on_stage=None) -> bytes:
    """POST to the engine and return decoded WAV bytes, optionally reporting
    stage-marker updates (parsed from the engine's own log) via on_stage."""
    log_pos = ENGINE_LOG_PATH.stat().st_size if ENGINE_LOG_PATH.exists() else 0

    async def watch_log() -> None:
        nonlocal log_pos
        while True:
            if ENGINE_LOG_PATH.exists():
                size = ENGINE_LOG_PATH.stat().st_size
                if size > log_pos:
                    with open(ENGINE_LOG_PATH, "r", encoding="utf-8", errors="ignore") as f:
                        f.seek(log_pos)
                        chunk = f.read()
                    log_pos = size
                    if on_stage:
                        for line in chunk.splitlines():
                            name = _stage_name_for_line(line)
                            if name:
                                on_stage(name)
                elif size < log_pos:
                    log_pos = size  # log file was rotated/truncated
            await asyncio.sleep(0.5)

    watcher = asyncio.create_task(watch_log()) if on_stage else None
    try:
        # CPU-backend generation time is unbounded (scales with lyrics length
        # and whatever else is running on the machine), so this call must not
        # time out on its own — the frontend's progress stream is what tells
        # the user whether it's still alive.
        async with httpx.AsyncClient(timeout=None) as client:
            res = await client.post(f"{AUDIOCPP_URL}/v1/tasks/run", json=body)
        if res.status_code != 200:
            raise RuntimeError(f"audio.cpp returned {res.status_code}: {res.text[:500]}")
        return _extract_audio_bytes(res.headers.get("content-type", ""), res.content)
    except httpx.ConnectError as exc:
        raise RuntimeError("audio.cpp engine is not reachable. Is it running?") from exc
    finally:
        if watcher:
            watcher.cancel()


def _ensure_wav(audio_path: str) -> str:
    """The engine's ASR endpoint only accepts WAV. Source uploads can be any
    format (mp3, m4a, ...), so transcode to a temp WAV alongside it first."""
    path = Path(audio_path)
    if path.suffix.lower() == ".wav":
        return audio_path
    wav_path = path.with_name(path.stem + "_asr.wav")
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(path), "-ac", "1", "-ar", "44100", "-c:a", "pcm_s16le", str(wav_path)],
        check=True, capture_output=True,
    )
    return str(wav_path)


async def _call_yue2_transcribe_lyrics(audio_path: str) -> str:
    """Speech-to-text over the source recording via the engine's ASR model.

    Qwen3-ASR auto-detects across 30+ languages; without a hint, singing can
    get misidentified as a different language entirely. Pin it to English
    since that's this app's only supported lyrics language.
    """
    wav_path = await run_in_threadpool(_ensure_wav, audio_path)
    async with httpx.AsyncClient(timeout=None) as client:
        res = await client.post(
            f"{AUDIOCPP_URL}/v1/audio/transcriptions",
            json={"model": ASR_MODEL_ID, "audio": wav_path, "language": "en"},
        )
    if res.status_code != 200:
        raise RuntimeError(f"Lyrics transcription failed: {res.status_code} {res.text[:300]}")
    text = res.json().get("text", "")
    log.info("transcribed lyrics: %r", text)
    return text


async def _run_job(job: Job, body: dict) -> None:
    try:
        job.result = await _call_yue2_generate(body, on_stage=lambda name: setattr(job, "stage", name))
        job.status = "done"
        job.stage = "Done"
    except Exception as exc:  # noqa: BLE001 - surfaced to the client as-is
        job.status = "error"
        job.error = str(exc)


@app.post("/api/generate")
async def generate(req: GenerateRequest):
    body = _build_generate_body(
        req.lyrics, req.style, req.cot, req.abc, req.seed, req.num_inference_steps, req.cfg_scale
    )
    job = Job(id=uuid.uuid4().hex, kind="generate")
    JOBS[job.id] = job
    asyncio.create_task(_run_job(job, body))
    return {"job_id": job.id, "stage_sequence": STAGE_SEQUENCE}


_APP_PROCESS = psutil.Process(os.getpid())


def _job_activity(job: Job) -> bool:
    # "generate" and "transcribe_lyrics" jobs do their real work in the
    # audio.cpp engine process; a plain "transcribe" (melody, SheetSage2)
    # runs in-process via a threadpool, so it's this app's own process.
    proc = _engine_process() if job.kind in ("generate", "transcribe_lyrics") else _APP_PROCESS
    if proc is None:
        return False
    try:
        return proc.cpu_percent(interval=0.2) > 5.0
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False


@app.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job")

    async def event_stream():
        while True:
            yield {
                "event": "progress",
                "data": json.dumps(
                    {
                        "status": job.status,
                        "stage": job.stage,
                        "elapsed_s": round(time.monotonic() - job.started_at, 1),
                        "engine_active": _job_activity(job),
                    }
                ),
            }
            if job.status != "running":
                yield {
                    "event": "final",
                    "data": json.dumps({"status": job.status, "error": job.error}),
                }
                return
            await asyncio.sleep(1.0)

    return EventSourceResponse(event_stream())


@app.get("/api/jobs/{job_id}/result")
async def job_result(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job")
    if job.status == "error":
        raise HTTPException(status_code=502, detail=job.error or "job failed")
    if job.status != "done" or job.result is None:
        raise HTTPException(status_code=409, detail="job not finished yet")
    result = job.result
    del JOBS[job_id]
    if job.kind == "generate":
        return Response(content=result, media_type="audio/wav")
    return result


@dataclass
class StyleRun:
    id: str
    label: str
    status: str = "queued"  # queued | running | done | error
    filename: str | None = None
    error: str | None = None


@dataclass
class Batch:
    id: str
    styles: list[StyleRun]
    status: str = "running"  # running | done | error
    stage: str = "Starting"
    phase: str = "app"  # "app" (this process) or "engine" (audio.cpp), for activity sampling
    error: str | None = None
    started_at: float = field(default_factory=time.monotonic)


BATCHES: dict[str, Batch] = {}


def _safe_filename_part(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in text)[:60]


_ABC_KEY_RE = re.compile(r"^(K:\s*)([A-G])([#b]?)(maj|min|m)?(.*)$", re.MULTILINE)


def flip_abc_key(abc: str) -> str:
    """Flip an ABC score's key between parallel major and minor (same tonic,
    e.g. K:C <-> K:Cm) by editing only the K: line. ABC applies each key's
    accidentals implicitly to bare note letters, so this one-line edit is
    enough to reharmonize the melody without touching the note data itself.
    """

    def repl(m: re.Match) -> str:
        prefix, tonic, accidental, mode, rest = m.groups()
        mode = (mode or "").lower()
        is_minor = mode.startswith("m") and not mode.startswith("maj")
        new_mode = "" if is_minor else "m"
        return f"{prefix}{tonic}{accidental}{new_mode}{rest}"

    return _ABC_KEY_RE.sub(repl, abc, count=1)


async def _run_batch(
    batch: Batch,
    source_path: str,
    tmp_dir: str,
    source_name: str,
    lyrics: str,
    cot: str,
    seed: int,
    num_inference_steps: int,
    flip_key: bool = False,
    vocal: str = "any",
    prepared_abc: str | None = None,
    session: Session | None = None,
) -> None:
    try:
        if not lyrics.strip():
            lyrics = "[Verse]\n(instrumental - no lyrics detected)"

        batch.stage = "Preparing cover"
        batch.phase = "app"
        abc = ""
        try:
            if prepared_abc is None:
                raise RuntimeError("Prepare vocal and instrumental melodies before generating")
            abc = prepared_abc
            log.info("transcribed melody key line: %r", next((l for l in abc.splitlines() if l.startswith("K:")), None))
            if flip_key and abc:
                abc = flip_abc_key(abc)
        except Exception as exc:
            raise RuntimeError(f"Cannot use the prepared melody: {exc}") from exc

        base_name = _safe_filename_part(Path(source_name).stem) or "cover"
        vocal_hint = {"male": "male lead vocal", "female": "female lead vocal"}.get(vocal, "")
        batch.phase = "engine"
        for style_run in batch.styles:
            preset = STYLE_PRESETS_BY_ID[style_run.id]
            style_run.status = "running"
            batch.stage = f"Generating: {preset['label']}"
            try:
                style_prompt = f"{preset['prompt']}, {vocal_hint}" if vocal_hint else preset["prompt"]
                body = _build_generate_body(lyrics, style_prompt, cot, abc, seed, num_inference_steps)
                audio_bytes = await _call_yue2_generate(body)
                filename = f"{base_name}_{style_run.id}_{batch.id[:8]}.wav"
                (COMPLETED_DIR / filename).write_bytes(audio_bytes)
                style_run.status = "done"
                style_run.filename = filename
            except Exception as exc:  # noqa: BLE001 - one style failing shouldn't stop the rest
                log.exception("generation failed for style %s", style_run.id)
                style_run.status = "error"
                style_run.error = str(exc)
            if session is not None:
                _save_session_batch(session, batch)

        batch.status = "error" if all(s.status == "error" for s in batch.styles) else "done"
        batch.stage = "Done"
    except Exception as exc:  # noqa: BLE001 - surfaced to the client as-is
        batch.status = "error"
        batch.error = str(exc)
    finally:
        if session is not None:
            session.active_batch_id = None
            _save_session_batch(session, batch)
        else:
            shutil.rmtree(tmp_dir, ignore_errors=True)


@app.get("/api/styles")
async def list_styles():
    return {"styles": [{"id": s["id"], "label": s["label"]} for s in STYLE_PRESETS]}


@app.get("/api/styles/full")
async def list_styles_full():
    """Full preset details (including prompt text) for the styles editor."""
    return {"styles": STYLE_PRESETS}


class StyleIn(BaseModel):
    label: str
    prompt: str


@app.post("/api/styles")
async def create_style(body: StyleIn):
    label = body.label.strip()
    prompt = body.prompt.strip()
    if not label or not prompt:
        raise HTTPException(status_code=400, detail="Label and prompt are required.")
    preset = {"id": _unique_style_id(label), "label": label, "prompt": prompt}
    STYLE_PRESETS.append(preset)
    _rebuild_styles_by_id()
    _save_styles(STYLE_PRESETS)
    return {"styles": STYLE_PRESETS}


@app.put("/api/styles/{style_id}")
async def update_style(style_id: str, body: StyleIn):
    preset = STYLE_PRESETS_BY_ID.get(style_id)
    if preset is None:
        raise HTTPException(status_code=404, detail="Unknown style id.")
    label = body.label.strip()
    prompt = body.prompt.strip()
    if not label or not prompt:
        raise HTTPException(status_code=400, detail="Label and prompt are required.")
    preset["label"] = label
    preset["prompt"] = prompt
    _save_styles(STYLE_PRESETS)
    return {"styles": STYLE_PRESETS}


@app.delete("/api/styles/{style_id}")
async def delete_style(style_id: str):
    if style_id not in STYLE_PRESETS_BY_ID:
        raise HTTPException(status_code=404, detail="Unknown style id.")
    if len(STYLE_PRESETS) <= 1:
        raise HTTPException(status_code=400, detail="At least one style must remain.")
    STYLE_PRESETS[:] = [s for s in STYLE_PRESETS if s["id"] != style_id]
    _rebuild_styles_by_id()
    _save_styles(STYLE_PRESETS)
    return {"styles": STYLE_PRESETS}


SESSIONS: dict[str, Session] = session_store.load_all()
for _session in SESSIONS.values():
    if _session.melody and _session.melody.get('vocal_source') == 'stems/vocals.wav':
        # Keep the old analysis files and covers, but never silently reuse a
        # combined score after switching back to the direct model export.
        _session.melody = None
        _session.ready = False
        _session.error = 'Resume preparation to rebuild the melody from the full recording. Saved lyrics and vocals will be reused.'


def _save_session_batch(session: Session, batch: Batch) -> None:
    record = next(item for item in session.generations if item["batch_id"] == batch.id)
    record.update(status=batch.status, error=batch.error, styles=[
        {"id": s.id, "label": s.label, "status": s.status,
         "filename": s.filename, "error": s.error} for s in batch.styles
    ])
    session_store.save(session)


def _session_result(session: Session) -> dict:
    return {"session_id": session.id, "source_name": session.source_name,
            "ready": session.ready, "error": session.error,
            "lyrics": session.lyrics or "", "melody": session.melody,
            "generation_lyrics": ({"lyrics": session.lyrics or "", "error": None, "method": "user-edited"}
                if session.lyrics_reviewed else cover_lyrics.prepare(session.lyrics or "", session.transcription, session.tmp_dir)),
            "lyric_timing": whisper_transcribe.timing_view(session.transcription, session.lyrics, session.tmp_dir),
            "vocals_url": f"/api/covers/{session.id}/vocals" if session.vocals_path else None,
            "source_url": f"/api/sessions/{session.id}/source",
            "separation_model": "htdemucs_ft", "generations": session.generations,
            "active_job_id": session.active_job_id, "active_batch_id": session.active_batch_id}


async def _run_lyrics_job(job: Job, session: Session) -> None:
    job.stage = "Separating vocals (htdemucs_ft)"
    session.ready = False
    try:
        session.error = None
        if not session.vocals_path or not Path(session.vocals_path).is_file():
            session.vocals_path = await vocal_separation.separate(session.source_path, session.tmp_dir)
            session_store.save(session)
        job.stage = "Transcribing vocals with Whisper large-v3 and word timing"
        if session.lyrics is None:
            session.transcription = await whisper_transcribe.transcribe(session.vocals_path, session.tmp_dir)
            session.lyrics = session.transcription["text"]
            session_store.save(session)
        if session.melody is None:
            session.melody = await run_in_threadpool(
                sheetsage2_transcribe.transcribe_cover, session.vocals_path,
                session.source_path, str(Path(session.tmp_dir) / "scores"),
                lambda stage: setattr(job, "stage", stage),
            )
        session.ready = True
        job.result = _session_result(session)
        job.status = "done"
        job.stage = "Done"
    except Exception as exc:  # noqa: BLE001 - surfaced to the client as-is
        job.status = "error"
        job.error = f"Song preparation failed during {job.stage}: {exc}"
        log.exception("song preparation failed")
        session.error = job.error
    finally:
        session.active_job_id = None
        session_store.save(session)


@app.post("/api/covers/prepare")
async def prepare_covers(file: UploadFile = File(...)):
    """Separate vocals, transcribe lyrics, and prepare both melody parts."""
    session_id = uuid.uuid4().hex
    song_name = _safe_filename_part(Path(file.filename or "song").stem) or "song"
    tmp_dir = str(session_store.ROOT / f"{song_name}-{session_id[:12]}")
    Path(tmp_dir).mkdir(parents=True, exist_ok=False)
    suffix = Path(file.filename or "source.wav").suffix or ".wav"
    source_path = Path(tmp_dir) / f"source{suffix}"
    source_path.write_bytes(await file.read())

    session = Session(
        id=session_id, source_path=str(source_path), tmp_dir=tmp_dir,
        source_name=file.filename or "source",
    )
    SESSIONS[session.id] = session

    job = Job(id=uuid.uuid4().hex, kind="transcribe_lyrics")
    session.active_job_id = job.id
    session_store.save(session)
    JOBS[job.id] = job
    asyncio.create_task(_run_lyrics_job(job, session))
    return {"session_id": session.id, "job_id": job.id}


@app.get("/api/sessions")
async def list_sessions():
    return [{"session_id": s.id, "source_name": s.source_name, "ready": s.ready,
             "error": s.error, "created_at": s.created_at}
            for s in sorted(SESSIONS.values(), key=lambda s: s.created_at, reverse=True)]


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    session = SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(404, "Unknown saved song")
    return _session_result(session)


@app.get("/api/sessions/{session_id}/source")
async def session_source(session_id: str):
    session = SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(404, "Unknown saved song")
    return FileResponse(session.source_path)


class LyricsUpdate(BaseModel):
    lyrics: str
    expected_lyrics: str | None = None


@app.post("/api/sessions/{session_id}/transcribe-whisper")
async def transcribe_saved_song(session_id: str):
    session = SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(404, "Unknown saved song")
    if not session.ready or session.active_job_id or session.active_batch_id:
        raise HTTPException(409, "Wait for the current preparation or generation to finish")
    job = Job(id=uuid.uuid4().hex, kind="transcribe_lyrics")
    session.active_job_id = job.id
    JOBS[job.id] = job
    session_store.save(session)
    asyncio.create_task(_run_whisper_job(job, session))
    return {"session_id": session.id, "job_id": job.id}


async def _run_whisper_job(job: Job, session: Session, timing_only: bool = False):
    job.stage = "Transcribing vocals with Whisper large-v3 and word timing"
    try:
        if timing_only:
            job.stage = "Tightening word boundaries against isolated vocals"
            result = await whisper_transcribe.refine(session.vocals_path, session.tmp_dir, session.transcription)
        else:
            result = await whisper_transcribe.transcribe(session.vocals_path, session.tmp_dir)
        # Keep reviewed text in a recoverable history before explicitly replacing it.
        history = Path(session.tmp_dir) / "lyrics-history.jsonl"
        with history.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"saved_at": time.time(), "lyrics": session.lyrics}) + "\n")
        session.transcription = result
        if not timing_only:
            session.lyrics = result["text"]
            session.lyrics_reviewed = False
        job.status = "done"
        job.stage = "Done"
        job.result = _session_result(session)
    except Exception as exc:
        job.status = "error"
        job.error = str(exc)
        log.exception("Whisper transcription failed")
    finally:
        session.active_job_id = None
        session_store.save(session)


@app.post("/api/sessions/{session_id}/tighten-timing")
async def tighten_word_timing(session_id: str):
    session = SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(404, "Unknown saved song")
    if not session.ready or session.active_job_id or session.active_batch_id:
        raise HTTPException(409, "Wait for the current run to finish")
    if not session.transcription or whisper_transcribe.normalized(session.lyrics or '') != whisper_transcribe.normalized(session.transcription['text']):
        raise HTTPException(409, "Timings must match the saved lyrics before tightening")
    job = Job(id=uuid.uuid4().hex, kind="transcribe_lyrics")
    session.active_job_id = job.id
    JOBS[job.id] = job
    session_store.save(session)
    asyncio.create_task(_run_whisper_job(job, session, timing_only=True))
    return {"session_id": session.id, "job_id": job.id}


@app.put("/api/sessions/{session_id}/lyrics")
async def save_session_lyrics(session_id: str, request: LyricsUpdate):
    session = SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(404, "Unknown saved song")
    if not session.ready or session.active_job_id or session.active_batch_id:
        raise HTTPException(409, "Wait for preparation to finish")
    if request.expected_lyrics is not None and session.lyrics != request.expected_lyrics:
        raise HTTPException(409, "Lyrics changed in another window. Reopen the song before editing.")
    session.transcription = whisper_transcribe.correct_words(session.transcription, session.lyrics, request.lyrics)
    session.lyrics = request.lyrics
    session.lyrics_reviewed = True
    session_store.save(session)
    return {"saved": True, "lyric_timing": whisper_transcribe.timing_view(session.transcription, session.lyrics, session.tmp_dir),
            "generation_lyrics": {"lyrics": session.lyrics, "error": None, "method": "user-edited"}}


@app.post("/api/sessions/{session_id}/prepare")
async def resume_session(session_id: str):
    session = SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(404, "Unknown saved song")
    if session.active_job_id:
        return {"session_id": session.id, "job_id": session.active_job_id}
    if session.active_batch_id:
        raise HTTPException(409, "A cover is still generating for this song")
    job = Job(id=uuid.uuid4().hex, kind="transcribe_lyrics")
    session.active_job_id = job.id
    session_store.save(session)
    JOBS[job.id] = job
    asyncio.create_task(_run_lyrics_job(job, session))
    return {"session_id": session.id, "job_id": job.id}


@app.get("/api/covers/{session_id}/vocals")
async def session_vocals(session_id: str):
    session = SESSIONS.get(session_id)
    if session is None or not session.ready or not session.vocals_path:
        raise HTTPException(status_code=404, detail="No prepared vocal stem for this session")
    return FileResponse(session.vocals_path, media_type="audio/wav")


@app.post("/api/covers/{session_id}/generate")
async def generate_covers(
    session_id: str,
    lyrics: str = Form(...),
    styles: str = Form(...),
    cot: str = Form("melody"),
    seed: int = Form(831001),
    num_inference_steps: int = Form(16),
    flip_key: bool = Form(False),
    vocal: str = Form("any"),
    lyrics_reviewed: bool = Form(False),
):
    session = SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="unknown or already-used session")
    if not session.ready or session.melody is None or session.active_job_id:
        raise HTTPException(status_code=409, detail="Wait for vocal and instrumental transcription to finish")
    if session.active_batch_id:
        raise HTTPException(409, "A cover is already generating for this song")
    if cot not in ("melody", "full", "off"):
        raise HTTPException(status_code=400, detail="Unknown melody adherence mode")
    if cot == 'full' and 'full_abc' not in session.melody:
        raise HTTPException(400, 'Choose Melody mode for the direct SheetSage2 score.')

    try:
        style_ids = json.loads(styles)
    except json.JSONDecodeError as exc:
        raise HTTPException(400, "Styles must be a JSON list") from exc
    if not isinstance(style_ids, list) or any(not isinstance(s, str) for s in style_ids):
        raise HTTPException(400, "Styles must be a list of style IDs")
    style_ids = list(dict.fromkeys(style_ids))
    unknown = [s for s in style_ids if s not in STYLE_PRESETS_BY_ID]
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown style id(s): {unknown}")
    if not style_ids:
        raise HTTPException(status_code=400, detail="Select at least one style.")

    corrected = whisper_transcribe.correct_words(session.transcription, session.lyrics, lyrics)
    manual = session.lyrics_reviewed or lyrics_reviewed is True
    prepared = ({"lyrics": lyrics, "error": None, "method": "user-edited"} if manual
                else cover_lyrics.prepare(lyrics, corrected, session.tmp_dir))
    if prepared['error']:
        raise HTTPException(409, prepared['error'])
    batch = Batch(
        id=uuid.uuid4().hex,
        styles=[StyleRun(id=sid, label=STYLE_PRESETS_BY_ID[sid]["label"]) for sid in style_ids],
    )
    BATCHES[batch.id] = batch
    session.transcription = corrected
    session.lyrics = lyrics
    session.lyrics_reviewed = manual
    session.active_batch_id = batch.id
    session.generations.append({"batch_id": batch.id, "created_at": time.time(),
        "status": "running", "lyrics": lyrics, "generation_lyrics": prepared['lyrics'], "styles": [],
        "settings": {"styles": style_ids, "cot": cot, "seed": seed,
                     "num_inference_steps": num_inference_steps, "flip_key": flip_key, "vocal": vocal}})
    session_store.save(session)
    asyncio.create_task(
        _run_batch(
            batch, session.source_path, session.tmp_dir, session.source_name,
            prepared['lyrics'], cot, seed, num_inference_steps, flip_key, vocal,
            prepared_abc=session.melody["full_abc" if cot == "full" else "abc"] if cot != "off" else "",
            session=session,
        )
    )
    return {"batch_id": batch.id}


def _batch_activity(batch: Batch) -> bool:
    proc = _engine_process() if batch.phase == "engine" else _APP_PROCESS
    if proc is None:
        return False
    try:
        return proc.cpu_percent(interval=0.2) > 5.0
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False


@app.get("/api/covers/{batch_id}/events")
async def cover_batch_events(batch_id: str):
    batch = BATCHES.get(batch_id)
    if batch is None:
        raise HTTPException(status_code=404, detail="unknown batch")

    async def event_stream():
        while True:
            yield {
                "event": "progress",
                "data": json.dumps(
                    {
                        "status": batch.status,
                        "stage": batch.stage,
                        "elapsed_s": round(time.monotonic() - batch.started_at, 1),
                        "engine_active": _batch_activity(batch),
                        "styles": [
                            {
                                "id": s.id,
                                "label": s.label,
                                "status": s.status,
                                "filename": s.filename,
                                "error": s.error,
                            }
                            for s in batch.styles
                        ],
                    }
                ),
            }
            if batch.status != "running":
                yield {
                    "event": "final",
                    "data": json.dumps({"status": batch.status, "error": batch.error}),
                }
                return
            await asyncio.sleep(1.0)

    return EventSourceResponse(event_stream())


app.mount("/completed", StaticFiles(directory=COMPLETED_DIR), name="completed")

static_dir = Path(__file__).parent / "static"
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
