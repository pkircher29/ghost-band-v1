"""Cover Studio local app: song library, lyric review, and Modal orchestration.

Every AI step runs on Modal (cloud/cover_ai.py). This process stores files,
looks up official lyrics, and keeps per-song state durable on disk so a closed
browser or a restart never loses finished work.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid
from pathlib import Path

import modal
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import lyrics_lookup
import phrasing

ROOT = Path(__file__).resolve().parents[1]
SONGS = Path(os.environ.get('COVER_STUDIO_SONGS', ROOT / 'songs'))
STATIC = Path(__file__).parent / 'static'
STYLES = json.loads((Path(__file__).parent / 'styles.json').read_text(encoding='utf-8'))
STYLE_BY_ID = {s['id']: s for s in STYLES}
MODAL_APP = os.environ.get('COVER_STUDIO_MODAL_APP', 'cover-studio-ai')
PARALLEL_TAKES = int(os.environ.get('COVER_STUDIO_PARALLEL', '3'))
log = logging.getLogger('cover_studio')
app = FastAPI(title='Cover Studio')
_locks: dict[str, asyncio.Lock] = {}
_tasks: set[asyncio.Task] = set()


def cloud(name: str):
    return modal.Cls.from_name(MODAL_APP, name)()


# ---------------------------------------------------------------- song storage

def folder(song_id: str) -> Path:
    if not re.fullmatch(r'[a-f0-9]{12}', song_id):
        raise HTTPException(404, 'Unknown song')
    path = SONGS / song_id
    if not (path / 'song.json').is_file():
        raise HTTPException(404, 'Unknown song')
    return path


def load(song_id: str) -> dict:
    return json.loads((folder(song_id) / 'song.json').read_text(encoding='utf-8'))


def save(song: dict) -> None:
    path = SONGS / song['id']
    path.mkdir(parents=True, exist_ok=True)
    song['updated'] = time.time()
    tmp = path / f'.song-{uuid.uuid4().hex}.tmp'
    tmp.write_text(json.dumps(song, indent=2, ensure_ascii=False), encoding='utf-8')
    tmp.replace(path / 'song.json')


def read_json(song_id: str, name: str):
    path = SONGS / song_id / name
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else None


def write_json(song_id: str, name: str, data) -> None:
    path = SONGS / song_id / name
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding='utf-8')
    tmp.replace(path)


def lock(song_id: str) -> asyncio.Lock:
    return _locks.setdefault(song_id, asyncio.Lock())


async def update(song_id: str, **fields) -> dict:
    async with lock(song_id):
        song = load(song_id)
        for name, value in fields.items():
            if name == 'stage' and isinstance(value, dict):
                song.setdefault('stages', {}).update(value)
            else:
                song[name] = value
        save(song)
        return song


def spawn(coro) -> None:
    task = asyncio.create_task(coro)
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)


async def preview(source: Path) -> str | None:
    """MP3 next to a FLAC for in-browser playback; 24-bit FLAC seeks too slowly."""
    target = source.with_suffix('.mp3')
    try:
        proc = await asyncio.create_subprocess_exec(
            os.environ.get('FFMPEG', 'ffmpeg'), '-v', 'error', '-y', '-i', str(source),
            '-c:a', 'libmp3lame', '-b:a', '192k', str(target),
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
        await proc.wait()
    except OSError:
        return None
    return target.name if proc.returncode == 0 and target.is_file() else None


def stage(name, status, **extra):
    return {'stage': {name: {'status': status, 'at': time.time(), **extra}}}


@app.on_event('startup')
async def recover():
    SONGS.mkdir(parents=True, exist_ok=True)
    for manifest in SONGS.glob('*/song.json'):
        song = json.loads(manifest.read_text(encoding='utf-8'))
        changed = False
        for info in song.get('stages', {}).values():
            if info.get('status') == 'running':
                info.update(status='interrupted', error='Cover Studio restarted while this was running.')
                changed = True
        for take in song.get('takes', []):
            if take.get('status') in ('queued', 'running', 'checking'):
                take.update(status='interrupted')
                changed = True
        if changed:
            save(song)


# ---------------------------------------------------------------- analysis

async def run_analysis(song_id: str) -> None:
    song = load(song_id)
    data = (SONGS / song_id / song['source']).read_bytes()
    suffix = Path(song['source']).suffix

    async def vocals():
        if read_json(song_id, 'asr.json') and (SONGS / song_id / 'vocals.flac').is_file():
            return
        await update(song_id, **stage('vocals', 'running'))
        try:
            result = await cloud('VocalLab').analyze_source.remote.aio(data)
            (SONGS / song_id / 'vocals.flac').write_bytes(result.pop('vocals_flac'))
            write_json(song_id, 'asr.json', result)
            await update(song_id, duration=result['duration'],
                         vocals_preview=await preview(SONGS / song_id / 'vocals.flac'),
                         **stage('vocals', 'done', seconds=result['timing']['total_s']))
        except Exception as exc:
            log.exception('vocal analysis failed')
            await update(song_id, **stage('vocals', 'failed', error=str(exc)[:600]))

    async def melody():
        if read_json(song_id, 'melody.json'):
            return
        await update(song_id, **stage('melody', 'running'))
        try:
            result = await cloud('Melody').transcribe.remote.aio(data, suffix)
            midi = result.pop('melody_mid', b'')
            if midi:
                (SONGS / song_id / 'melody.mid').write_bytes(midi)
            write_json(song_id, 'melody.json', result)
            await update(song_id, **stage('melody', 'done', seconds=result['seconds'],
                                          warnings=result['warnings'][:8]))
        except Exception as exc:
            log.exception('melody transcription failed')
            await update(song_id, **stage('melody', 'failed', error=str(exc)[:600]))

    await asyncio.gather(vocals(), melody())
    song = load(song_id)
    if not song.get('lyrics') and song['stages'].get('vocals', {}).get('status') == 'done':
        await auto_lyrics(song_id)


async def auto_lyrics(song_id: str) -> None:
    """Pick the best official-lyrics match and align it, unless the user already chose."""
    song = load(song_id)
    try:
        found = await lyrics_lookup.search(song.get('artist', ''), song.get('title', ''), song.get('duration'))
    except Exception as exc:
        await update(song_id, **stage('lyrics', 'failed', error=f'Lyrics lookup failed: {exc}'))
        return
    write_json(song_id, 'lyrics-candidates.json', found)
    best = next((c for c in found if c['duration_diff'] is None or c['duration_diff'] <= 8), None)
    if not best:
        await update(song_id, **stage('lyrics', 'needs-input',
                                      error='No official lyrics found for this artist/title. Search or paste them.'))
        return
    await set_lyrics(song_id, best['lines'], best['times'], best['source'],
                     f"{best['artist']} — {best['title']}")


async def set_lyrics(song_id, lines, times, source, label) -> None:
    lines = [l.strip() for l in lines if l.strip()]
    if times and len(times) != len(lines):
        times = None
    write_json(song_id, 'lyrics.json', {'lines': lines, 'times': times, 'source': source, 'label': label})
    await update(song_id, lyrics={'source': source, 'label': label, 'lines': len(lines),
                                  'synced': bool(times)}, generation_lyrics=None,
                 **stage('lyrics', 'done'))
    spawn(run_alignment(song_id))


async def run_alignment(song_id: str) -> None:
    lyrics = read_json(song_id, 'lyrics.json')
    asr = read_json(song_id, 'asr.json')
    if not lyrics or not asr:
        return
    await update(song_id, **stage('align', 'running'))
    try:
        result = await cloud('VocalLab').align_lyrics.remote.aio(
            (SONGS / song_id / 'vocals.flac').read_bytes(), lyrics['lines'], asr['asr']['words'], lyrics['times'])
        result['lyrics_source'] = lyrics['source']
        write_json(song_id, 'aligned.json', result)
        (SONGS / song_id / 'lyrics.lrc').write_text(result['lrc'], encoding='utf-8')
        await update(song_id, **stage('align', 'done', **result['stats']))
    except Exception as exc:
        log.exception('alignment failed')
        await update(song_id, **stage('align', 'failed', error=str(exc)[:600]))


# ---------------------------------------------------------------- generation

def vocal_phrase(vocal: str) -> str:
    return {'female': 'female lead vocal', 'male': 'male lead vocal'}.get(vocal, '')


def generation_lyrics(song_id: str) -> dict:
    song = load(song_id)
    if song.get('generation_lyrics'):
        return {'lyrics': song['generation_lyrics'], 'method': 'edited by you', 'sections': []}
    aligned = read_json(song_id, 'aligned.json')
    if not aligned:
        return {'lyrics': '', 'method': 'waiting for aligned lyrics', 'sections': []}
    melody = read_json(song_id, 'melody.json') or {}
    return phrasing.build(aligned['lines'], melody.get('structure_lab', ''))


async def run_take(song_id: str, take_id: str, semaphore: asyncio.Semaphore) -> None:
    async with semaphore:
        song = load(song_id)
        take = next(t for t in song['takes'] if t['id'] == take_id)
        await set_take(song_id, take_id, status='running', started=time.time())
        try:
            result = await cloud('Composer').generate.remote.aio(
                take['style_prompt'], take['lyrics'], take.get('abc'), take['cot'], take['seed'],
                take.get('cfg_scale'), take_id)
            name = f"{take_id}.flac"
            (SONGS / song_id / 'takes').mkdir(exist_ok=True)
            (SONGS / song_id / 'takes' / name).write_bytes(result.pop('flac'))
            mp3 = await preview(SONGS / song_id / 'takes' / name)
            await set_take(song_id, take_id, status='checking', file=f'takes/{name}',
                           preview=f'takes/{mp3}' if mp3 else None,
                           seconds=result['seconds'], audio_seconds=result['audio_seconds'],
                           truncated=result['truncated'])
        except Exception as exc:
            log.exception('generation failed')
            await set_take(song_id, take_id, status='failed', error=str(exc)[:600])
            return
    # Lyric check runs outside the generation slot so the next style starts now.
    try:
        lines = [l for l in take['lyrics'].splitlines() if l.strip() and not re.fullmatch(r'\[[^\]]+\]', l.strip())]
        check = await cloud('VocalLab').check_take.remote.aio(
            (SONGS / song_id / 'takes' / f'{take_id}.flac').read_bytes(), lines)
        (SONGS / song_id / 'takes' / f'{take_id}.lrc').write_text(check['lrc'], encoding='utf-8')
        write_json(song_id, f'takes/{take_id}.check.json', check)
        await set_take(song_id, take_id, status='done', lyric_accuracy=check['lyric_accuracy'],
                       lrc=f'takes/{take_id}.lrc')
    except Exception as exc:
        log.exception('take check failed')
        await set_take(song_id, take_id, status='done', check_error=str(exc)[:400])


async def set_take(song_id, take_id, **fields):
    async with lock(song_id):
        song = load(song_id)
        for take in song['takes']:
            if take['id'] == take_id:
                take.update(fields)
        save(song)


# ---------------------------------------------------------------- API

@app.get('/api/styles')
async def styles():
    return STYLES


@app.get('/api/health')
async def health():
    try:
        await asyncio.wait_for(asyncio.to_thread(lambda: modal.Cls.from_name(MODAL_APP, 'Composer').hydrate()), 20)
        return {'ok': True, 'modal_app': MODAL_APP}
    except Exception as exc:
        return {'ok': False, 'modal_app': MODAL_APP, 'error': str(exc)[:300]}


@app.get('/api/songs')
async def list_songs():
    songs = [json.loads(p.read_text(encoding='utf-8')) for p in SONGS.glob('*/song.json')]
    return sorted(songs, key=lambda s: -s.get('created', 0))


@app.post('/api/songs')
async def create_song(file: UploadFile = File(...), artist: str = Form(''), title: str = Form('')):
    name = Path(file.filename or 'song.mp3').name
    suffix = Path(name).suffix.lower() or '.mp3'
    if suffix not in {'.mp3', '.wav', '.flac', '.m4a', '.ogg', '.aac', '.opus', '.wma', '.mp4', '.webm'}:
        raise HTTPException(400, 'Unsupported audio file type')
    guess_artist, guess_title = lyrics_lookup.guess_artist_title(name)
    song_id = uuid.uuid4().hex[:12]
    path = SONGS / song_id
    path.mkdir(parents=True)
    (path / f'source{suffix}').write_bytes(await file.read())
    save({'id': song_id, 'name': name, 'source': f'source{suffix}', 'created': time.time(),
          'artist': artist.strip() or guess_artist, 'title': title.strip() or guess_title,
          'stages': {}, 'takes': [], 'lyrics': None})
    spawn(run_analysis(song_id))
    return load(song_id)


@app.get('/api/songs/{song_id}')
async def get_song(song_id: str):
    song = load(song_id)
    song['generation'] = generation_lyrics(song_id)
    return song


@app.post('/api/songs/{song_id}/resume')
async def resume(song_id: str):
    song = load(song_id)
    if (SONGS / song_id / 'vocals.flac').is_file() and not song.get('vocals_preview'):
        await update(song_id, vocals_preview=await preview(SONGS / song_id / 'vocals.flac'))
    spawn(run_analysis(song_id))
    return {'ok': True}


@app.patch('/api/songs/{song_id}')
async def rename(song_id: str, body: dict):
    fields = {k: str(v).strip() for k, v in body.items() if k in ('artist', 'title')}
    return await update(song_id, **fields)


@app.get('/api/songs/{song_id}/asr')
async def song_asr(song_id: str):
    return read_json(song_id, 'asr.json') or {}


@app.get('/api/songs/{song_id}/lyrics')
async def song_lyrics(song_id: str):
    return {'lyrics': read_json(song_id, 'lyrics.json'), 'aligned': read_json(song_id, 'aligned.json'),
            'candidates': read_json(song_id, 'lyrics-candidates.json') or []}


@app.get('/api/lyrics/search')
async def lyrics_search(artist: str = '', title: str = '', q: str = '', duration: float | None = None):
    try:
        return await lyrics_lookup.search(artist, title, duration, q)
    except Exception as exc:
        raise HTTPException(502, f'Lyrics lookup failed: {exc}')


class LyricsIn(BaseModel):
    text: str | None = None
    lines: list[str] | None = None
    times: list[float] | None = None
    source: str = 'pasted by you'
    label: str = ''


@app.put('/api/songs/{song_id}/lyrics')
async def put_lyrics(song_id: str, body: LyricsIn):
    load(song_id)
    lines = body.lines if body.lines is not None else lyrics_lookup.plain_lines(body.text or '')
    if not lines:
        raise HTTPException(400, 'No lyric lines')
    old = read_json(song_id, 'lyrics.json')
    times = body.times
    # Editing words keeps the line time codes as long as the line count still matches.
    if times is None and old and old.get('times') and len(old['times']) == len(lines):
        times = old['times']
    await set_lyrics(song_id, lines, times, body.source, body.label)
    return load(song_id)


class WordEdit(BaseModel):
    index: int
    start: float
    end: float


@app.put('/api/songs/{song_id}/words')
async def edit_words(song_id: str, edits: list[WordEdit]):
    aligned = read_json(song_id, 'aligned.json')
    if not aligned:
        raise HTTPException(409, 'Lyrics are not aligned yet')
    words = aligned['words']
    for e in edits:
        if not 0 <= e.index < len(words) or e.end <= e.start:
            raise HTTPException(400, f'Bad timing for word {e.index}')
        word = words[e.index]
        if not word.get('backing'):
            lead = [i for i, w in enumerate(words) if not w.get('backing')]
            k = lead.index(e.index)
            before = words[lead[k - 1]] if k else None
            after = words[lead[k + 1]] if k + 1 < len(lead) else None
            if before and e.start < before['end'] - 1e-3:
                raise HTTPException(400, f'"{word["text"]}" would start before "{before["text"]}" ends ({before["end"]:.3f}s).')
            if after and e.end > after['start'] + 1e-3:
                raise HTTPException(400, f'"{word["text"]}" would end after "{after["text"]}" starts ({after["start"]:.3f}s).')
        words[e.index].update(start=round(e.start, 3), end=round(e.end, 3), start_source='manual',
                              end_source='manual', review=[])
    for line in aligned['lines']:
        line['start'] = line['end'] = None
    for n, line in enumerate(aligned['lines']):
        members = [w for w in words if w['line'] == n]
        if members:
            line.update(start=members[0]['start'], end=members[-1]['end'])
    lyrics = read_json(song_id, 'lyrics.json')
    sys_path_lrc = SONGS / song_id / 'lyrics.lrc'
    sys_path_lrc.write_text(_lrc(words, lyrics['lines']), encoding='utf-8')
    aligned['lrc'] = sys_path_lrc.read_text(encoding='utf-8')
    write_json(song_id, 'aligned.json', aligned)
    return aligned


def _lrc(words, lines):
    def stamp(t):
        return f'{int(t // 60):02d}:{t % 60:05.2f}'
    out = []
    for n, text in enumerate(lines):
        members = [w for w in words if w['line'] == n]
        if members:
            out.append(f'[{stamp(members[0]["start"])}]' + ' '.join(
                f'<{stamp(w["start"])}>{w["text"]}' for w in members) + f' <{stamp(members[-1]["end"])}>')
    return '\n'.join(out) + '\n'


class GenerationLyricsIn(BaseModel):
    lyrics: str | None


@app.put('/api/songs/{song_id}/generation-lyrics')
async def put_generation_lyrics(song_id: str, body: GenerationLyricsIn):
    text = (body.lyrics or '').strip() or None
    if text and not re.search(r'^\s*\[[^\]]+\]\s*$', text, re.M):
        raise HTTPException(400, 'Keep at least one section tag such as [Verse] or [Chorus].')
    await update(song_id, generation_lyrics=text)
    return generation_lyrics(song_id)


class CoversIn(BaseModel):
    styles: list[str] = []
    custom_style: str = ''
    vocal: str = 'auto'
    seed: int | None = None
    melody: str = 'source'        # 'source' = SheetSage2 melody, 'free' = let YuE2 compose
    cfg_scale: float | None = None


@app.post('/api/songs/{song_id}/covers')
async def make_covers(song_id: str, body: CoversIn):
    song = load(song_id)
    lyrics = generation_lyrics(song_id)['lyrics']
    if not lyrics:
        raise HTTPException(409, 'Lyrics are not ready yet')
    melody = read_json(song_id, 'melody.json')
    if body.melody == 'source' and not melody:
        raise HTTPException(409, 'The source melody has not been transcribed yet')
    chosen = [STYLE_BY_ID[s] for s in body.styles if s in STYLE_BY_ID]
    if body.custom_style.strip():
        chosen.append({'id': 'custom', 'label': 'Custom', 'prompt': body.custom_style.strip()})
    if not chosen:
        raise HTTPException(400, 'Pick at least one style')
    seed = body.seed if body.seed is not None else int.from_bytes(os.urandom(3), 'big')
    vocal = vocal_phrase(body.vocal)
    new = []
    for style in chosen:
        prompt = style['prompt'] + (f', {vocal}' if vocal else '')
        new.append({'id': f"{style['id']}-{uuid.uuid4().hex[:6]}", 'style': style['id'], 'label': style['label'],
                    'style_prompt': prompt, 'lyrics': lyrics, 'seed': seed,
                    'cot': 'melody' if body.melody == 'source' else 'full',
                    'abc': melody['abc'] if body.melody == 'source' else None,
                    'cfg_scale': body.cfg_scale, 'status': 'queued', 'created': time.time()})
    async with lock(song_id):
        song = load(song_id)
        song['takes'].extend(new)
        save(song)
    semaphore = asyncio.Semaphore(PARALLEL_TAKES)
    for take in new:
        spawn(run_take(song_id, take['id'], semaphore))
    return load(song_id)


@app.delete('/api/songs/{song_id}/takes/{take_id}')
async def delete_take(song_id: str, take_id: str):
    async with lock(song_id):
        song = load(song_id)
        take = next((t for t in song['takes'] if t['id'] == take_id), None)
        if not take or take['status'] in ('queued', 'running', 'checking'):
            raise HTTPException(409, 'Only finished takes can be removed')
        song['takes'].remove(take)
        save(song)
    for suffix in ('.flac', '.mp3', '.lrc', '.check.json'):
        (SONGS / song_id / 'takes' / f'{take_id}{suffix}').unlink(missing_ok=True)
    return {'ok': True}


@app.get('/api/songs/{song_id}/files/{name:path}')
async def song_file(song_id: str, name: str):
    base = folder(song_id).resolve()
    path = (base / name).resolve()
    if base not in path.parents or not path.is_file() or path.name == 'song.json':
        raise HTTPException(404)
    return FileResponse(path)


class UIFiles(StaticFiles):
    """Revalidate UI files on every load so an edited app.js is never served stale."""
    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers['Cache-Control'] = 'no-cache'
        return response


app.mount('/', UIFiles(directory=STATIC, html=True), name='static')
