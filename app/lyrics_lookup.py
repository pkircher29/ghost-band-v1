"""Official lyrics from LRCLIB (free, keyless): plain text plus line time codes."""
from __future__ import annotations

import re

import httpx

API = 'https://lrclib.net/api'
HEADERS = {'User-Agent': 'CoverStudio/2.0 (https://github.com/pkircher29/cover-studio)'}
_STAMP = re.compile(r'\[(\d+):(\d+(?:\.\d+)?)\]')


def guess_artist_title(filename: str) -> tuple[str, str]:
    """'My Chemical Romance - Helena (Official Video).mp3' -> ('My Chemical Romance', 'Helena')."""
    stem = re.sub(r'\.[A-Za-z0-9]{2,5}$', '', filename).replace('_', ' ')
    stem = re.sub(r'[\(\[][^\)\]]*(official|video|audio|lyric|remaster|hd|hq)[^\)\]]*[\)\]]', '', stem, flags=re.I)
    if ' - ' in stem:
        artist, title = stem.split(' - ', 1)
        return artist.strip(), title.strip()
    return '', stem.strip()


def parse_lrc(synced: str) -> tuple[list[str], list[float | None]]:
    lines, times = [], []
    for row in synced.splitlines():
        stamps = list(_STAMP.finditer(row))
        text = _STAMP.sub('', row).strip()
        if not stamps or not text:
            continue
        for stamp in stamps:   # "[00:10.00][01:20.00]chorus" repeats a line
            lines.append(text)
            times.append(int(stamp.group(1)) * 60 + float(stamp.group(2)))
    order = sorted(range(len(lines)), key=lambda i: times[i])
    return [lines[i] for i in order], [times[i] for i in order]


def plain_lines(text: str) -> list[str]:
    return [line.strip() for line in (text or '').splitlines()
            if line.strip() and not re.fullmatch(r'\[[^\]]*\]', line.strip())]


def _candidate(item: dict, duration: float | None) -> dict:
    synced = item.get('syncedLyrics') or ''
    lines, times = parse_lrc(synced) if synced else (plain_lines(item.get('plainLyrics') or ''), [])
    diff = abs((item.get('duration') or 0) - duration) if duration else None
    return {'id': item['id'], 'artist': item.get('artistName'), 'title': item.get('trackName'),
            'album': item.get('albumName'), 'duration': item.get('duration'),
            'duration_diff': None if diff is None else round(diff, 1),
            'synced': bool(synced), 'instrumental': bool(item.get('instrumental')),
            'lines': lines, 'times': times or None, 'source': f'LRCLIB #{item["id"]}'}


async def search(artist: str = '', title: str = '', duration: float | None = None,
                 query: str = '') -> list[dict]:
    """Candidates best-first: closest duration, synced before plain, more lines first."""
    params = {'track_name': title, 'artist_name': artist} if title else {'q': query}
    if title and not artist:
        params = {'q': title}
    async with httpx.AsyncClient(headers=HEADERS, timeout=20) as client:
        response = await client.get(f'{API}/search', params=params)
        response.raise_for_status()
        items = response.json()
        if not items and title:
            response = await client.get(f'{API}/search', params={'q': f'{artist} {title}'.strip()})
            response.raise_for_status()
            items = response.json()
    out = [_candidate(i, duration) for i in items if not i.get('instrumental') and
           (i.get('syncedLyrics') or i.get('plainLyrics'))]
    # Prefer a timing that matches this recording; reject wild-length versions
    # (live, extended, mislabelled multi-track files).
    out.sort(key=lambda c: ((c['duration_diff'] if c['duration_diff'] is not None else 0) > 8,
                            not c['synced'],
                            c['duration_diff'] if c['duration_diff'] is not None else 0,
                            -len(c['lines'])))
    return out[:12]
