"""Sectioned generation lyrics: official lines as phrases, sections from timing + structure.

YuE2 takes no timestamps; what it does take is section tags and one sung phrase
per line. So phrasing comes from the official lyric lines (how the song is
actually phrased), stanzas from real silences in the aligned vocal, and section
names from SheetSage2's structure analysis, with repetition as the fallback.
"""
from __future__ import annotations

import re

LABELS = [('pre-chorus', 'Pre-Chorus'), ('prechorus', 'Pre-Chorus'), ('chorus', 'Chorus'),
          ('refrain', 'Chorus'), ('verse', 'Verse'), ('bridge', 'Bridge'), ('intro', 'Intro'),
          ('outro', 'Outro'), ('coda', 'Outro'), ('inst', 'Bridge'), ('solo', 'Bridge'), ('break', 'Bridge')]
STANZA_GAP = 2.2   # seconds of silence between lines that starts a new stanza


def parse_structure(lab: str) -> list[tuple[float, float, str]]:
    sections = []
    for row in (lab or '').splitlines():
        parts = row.split()
        if len(parts) < 3:
            continue
        try:
            start, end = float(parts[0]), float(parts[1])
        except ValueError:
            continue
        raw = ' '.join(parts[2:]).lower()
        name = next((tag for needle, tag in LABELS if needle in raw), None)
        if name and end > start:
            sections.append((start, end, name))
    return sections


def _norm(line: str) -> str:
    return ' '.join(re.findall(r"[a-z']+", line.lower()))


def stanzas(lines: list[dict]) -> list[list[int]]:
    """Split timed lines ({text,start,end}) into stanzas at real vocal silences."""
    groups: list[list[int]] = []
    for i, line in enumerate(lines):
        if groups:
            previous = lines[groups[-1][-1]]
            gap = (line['start'] or 0) - (previous['end'] or 0)
            if gap >= STANZA_GAP or len(groups[-1]) >= 8:
                groups.append([i])
                continue
            groups[-1].append(i)
        else:
            groups.append([i])
    return groups


def _label_by_structure(group, lines, sections):
    overlap = {}
    for i in group:
        s, e = lines[i]['start'], lines[i]['end']
        if s is None or e is None:
            continue
        for a, b, name in sections:
            overlap[name] = overlap.get(name, 0) + max(0.0, min(e, b) - max(s, a))
    if overlap and max(overlap.values()) > 0:
        return max(overlap, key=overlap.get)
    return None


def structure_groups(lines: list[dict], sections) -> tuple[list[list[int]], list[str]] | None:
    """Group lines by the structure segment each one overlaps most (segment, not label:
    two verses in a row stay two stanzas)."""
    owner = []
    for line in lines:
        s, e = line['start'], line['end']
        if s is None or e is None:
            owner.append(owner[-1] if owner else None)
            continue
        overlaps = [max(0.0, min(e, b) - max(s, a)) for a, b, _ in sections]
        best = max(range(len(sections)), key=lambda k: overlaps[k])
        owner.append(best if overlaps[best] > 0 else (owner[-1] if owner else None))
    if any(o is None for o in owner):
        return None
    groups, labels = [], []
    for i, o in enumerate(owner):
        if groups and owner[groups[-1][-1]] == o:
            groups[-1].append(i)
        else:
            groups.append([i])
            labels.append(sections[o][2])
    return groups, labels


def build(lines: list[dict], structure_lab: str = '') -> dict:
    """Return {'lyrics': sectioned text, 'sections': [...], 'method': ...}."""
    if not lines:
        return {'lyrics': '', 'sections': [], 'method': 'empty'}
    sections = parse_structure(structure_lab)
    grouped = structure_groups(lines, sections) if sections else None
    if grouped:
        groups, labels = grouped
        out, meta = [], []
        for group, label in zip(groups, labels):
            if out:
                out.append('')
            out.append(f'[{label}]')
            out.extend(lines[i]['text'] for i in group)
            meta.append({'label': label, 'lines': group,
                         'start': lines[group[0]]['start'], 'end': lines[group[-1]]['end']})
        return {'lyrics': '\n'.join(out), 'sections': meta,
                'method': 'official lines, sections from SheetSage2 song structure'}
    groups = stanzas(lines)
    texts = [[_norm(lines[i]['text']) for i in g] for g in groups]
    # A stanza is a chorus if most of its lines recur in another stanza.
    repeated = []
    for n, group_text in enumerate(texts):
        others = {t for m, ts in enumerate(texts) if m != n for t in ts}
        repeated.append(sum(t in others for t in group_text) >= max(1, len(group_text) // 2 + 1)
                        if len(group_text) > 1 else group_text[0] in others)
    labels = []
    for n, group in enumerate(groups):
        label = _label_by_structure(group, lines, sections) if sections else None
        if label in (None, 'Intro', 'Outro') or (label == 'Bridge' and repeated[n]):
            label = 'Chorus' if repeated[n] else ('Bridge' if label == 'Bridge' else 'Verse')
        if label == 'Verse' and repeated[n] and sections:
            # structure said verse, but the words are a refrain: trust the words
            label = 'Chorus'
        labels.append(label)
    out, meta = [], []
    for group, label in zip(groups, labels):
        if out:
            out.append('')
        out.append(f'[{label}]')
        out.extend(lines[i]['text'] for i in group)
        meta.append({'label': label, 'lines': group,
                     'start': lines[group[0]]['start'], 'end': lines[group[-1]]['end']})
    return {'lyrics': '\n'.join(out), 'sections': meta,
            'method': 'official lines, stanzas from vocal silences, labels from repetition'}
