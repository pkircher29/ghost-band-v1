"""Official-lyrics word timing: anchor, force-align, then refine both word edges.

Pure numpy/stdlib except `ctc_align`, which takes an already-loaded torchaudio
MMS_FA model. Everything that decides *where a word starts and ends* lives here
so it can be unit-tested locally without a GPU.

Pipeline
1. `tokenize_lines` turns official lyric lines into words (keeping display text,
   a normalized spoken form, line index and whether it is a backing-vocal echo).
2. `anchor_words` matches those words against Whisper's ASR words (difflib on
   normalized tokens) so every confident match pins a rough time.
3. `estimate_lines` gives every lyric line a time window from anchors, falling
   back to LRC line time codes (offset-corrected) and interpolation.
4. `plan_blocks` groups lines into <=40 s blocks split at instrumental gaps.
5. `ctc_align` force-aligns each block's words with MMS_FA, star tokens between
   lines absorbing ad-libs and anything the lyric sheet doesn't contain.
6. `refine_edges` moves each word's start onto the real vocal onset and extends
   its end through the sustained note, which CTC alone always cuts short.
"""
from __future__ import annotations


from difflib import SequenceMatcher
import re
import unicodedata

import numpy as np

FRAME_HOP = 0.005          # envelope hop (s)
ONSET_BACK = 0.15          # max a start may move earlier than CTC
ONSET_FORWARD = 0.15       # max a start may move later (CTC landed in silence)
MAX_SUSTAIN = 4.0          # max a word end may extend past CTC
DROP_DB = 20.0             # end a sustain once energy falls this far below the word peak
BRIDGE_FRAMES = 6          # tolerate <=30 ms dropouts inside a sustained note
MIN_WORD = 0.06
EDGE_DB = 12.0             # a rise/fall this big within EDGE_SPAN is a note attack/release
EDGE_SPAN = 0.05
HELD_ONSET_BACK = 1.6      # line-initial words may start this much before CTC (held first vowel)

_ONES = 'zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen'.split()
_TENS = 'twenty thirty forty fifty sixty seventy eighty ninety'.split()


def number_words(n: int) -> str:
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10 - 2] + ('' if n % 10 == 0 else ' ' + _ONES[n % 10])
    if n < 1000:
        rest = n % 100
        return _ONES[n // 100] + ' hundred' + ('' if not rest else ' ' + number_words(rest))
    if 1000 < n < 2100 and n % 100:   # years: 1984 -> nineteen eighty four
        return number_words(n // 100) + ' ' + number_words(n % 100)
    if n < 1_000_000:
        rest = n % 1000
        return number_words(n // 1000) + ' thousand' + ('' if not rest else ' ' + number_words(rest))
    return ' '.join(_ONES[int(d)] for d in str(n))


def spoken(word: str) -> str:
    """Normalized spoken form: lowercase a-z and apostrophes, digits spelled out."""
    text = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode()
    text = text.lower().replace('&', ' and ').replace('’', "'").replace('`', "'")
    text = re.sub(r'\d+', lambda m: ' ' + number_words(int(m.group())) + ' ', text)
    text = re.sub(r"[^a-z' ]", ' ', text)
    text = re.sub(r"(^|\s)'+|'+(\s|$)", ' ', text)   # quotes, not contractions
    return ' '.join(text.split())


def key(word: str) -> str:
    return spoken(word).replace("'", '').replace(' ', '')


def tokenize_lines(lines: list[str]) -> list[dict]:
    """Split lyric lines into words; text in parentheses is a backing-vocal echo."""
    words = []
    for index, line in enumerate(lines):
        depth = 0
        for raw in line.split():
            opens = raw.count('(')
            backing = depth > 0 or opens > 0
            depth = max(0, depth + opens - raw.count(')'))
            form = spoken(raw)
            if not form:
                continue
            words.append({'text': raw, 'spoken': form, 'key': key(raw), 'line': index,
                          'backing': backing})
    return words


def anchor_words(words: list[dict], asr_words: list[dict]) -> list[dict | None]:
    """Return per-lyric-word ASR match ({start,end}) or None. Only trustworthy matches."""
    a = [w['key'] for w in words]
    b = [key(w['text']) for w in asr_words]
    anchors: list[dict | None] = [None] * len(words)
    for block in SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks():
        if not block.size:
            continue
        for k in range(block.size):
            i, j = block.a + k, block.b + k
            # Lone short matches ("you", "the") drift; demand context or a long word.
            if block.size >= 2 or len(a[i]) >= 5:
                anchors[i] = {'start': float(asr_words[j]['start']), 'end': float(asr_words[j]['end'])}
    # Drop anchors that go backwards in time (ASR hallucinated a repeat).
    last = -1.0
    for i, anchor in enumerate(anchors):
        if anchor is None:
            continue
        if anchor['start'] < last - 0.05:
            anchors[i] = None
        else:
            last = anchor['start']
    return anchors


def estimate_lines(words, anchors, line_count, lrc_times=None, duration=None, per_word=0.32,
                   sources=None):
    """Estimated (start, end) for every line, from LRC codes, anchors, interpolation.

    Human-made LRC line codes (shifted by their measured offset against this
    recording) are the most reliable line starts for sung audio, so they win when
    they agree with the ASR anchors; anchors place the rest. `sources`, if given a
    list, receives 'lrc' | 'asr' | 'interpolated' per line.
    """
    starts = [None] * line_count
    ends = [None] * line_count
    src = [None] * line_count
    for line in range(line_count):
        idx = [i for i, w in enumerate(words) if w['line'] == line]
        hits = [(n, anchors[i]) for n, i in enumerate(idx) if anchors[i]]
        if hits:
            first_n, first = hits[0]
            last_n, last = hits[-1]
            starts[line] = max(0.0, first['start'] - first_n * per_word)
            ends[line] = last['end'] + (len(idx) - 1 - last_n) * per_word
            src[line] = 'asr'
    # LRC codes, shifted by the median disagreement with anchored lines.
    if lrc_times and any(t is not None for t in lrc_times):
        deltas = [starts[l] - lrc_times[l] for l in range(line_count)
                  if starts[l] is not None and l < len(lrc_times) and lrc_times[l] is not None]
        offset = float(np.median(deltas)) if deltas else 0.0
        spread = float(np.median([abs(d - offset) for d in deltas])) if deltas else 0.0
        if abs(offset) < 1.0:
            # Sub-second disagreement is ASR word-onset noise, not a different
            # edit of the song; shifting by it pushes windows off the real lines.
            offset = 0.0
        if not deltas or spread < 1.5:          # LRC agrees with this recording's timing
            for l in range(line_count):
                if l < len(lrc_times) and lrc_times[l] is not None:
                    lrc_start = max(0.0, lrc_times[l] + offset)
                    if starts[l] is None or abs(starts[l] - lrc_start) < 2.5:
                        if starts[l] is not None and ends[l] is not None:
                            ends[l] = max(ends[l], lrc_start + 0.2)
                        starts[l] = lrc_start
                        src[l] = 'lrc'
    if sources is not None:
        sources[:] = [s or 'interpolated' for s in src]
    # Interpolate whatever is still unknown by word count between known lines.
    counts = [sum(1 for w in words if w['line'] == l) for l in range(line_count)]
    known = [l for l in range(line_count) if starts[l] is not None]
    total = duration or (max((e for e in ends if e is not None), default=0) + 5)
    for l in range(line_count):
        if starts[l] is not None:
            continue
        before = max((k for k in known if k < l), default=None)
        after = min((k for k in known if k > l), default=None)
        t0 = 0.0 if before is None else (ends[before] or starts[before] + counts[before] * per_word)
        t1 = total if after is None else starts[after]
        span = list(range((before + 1) if before is not None else 0, after if after is not None else line_count))
        weight = sum(max(1, counts[k]) for k in span) or 1
        offset = sum(max(1, counts[k]) for k in span if k < l)
        starts[l] = t0 + (t1 - t0) * offset / weight
    for l in range(line_count):
        if ends[l] is None:
            ends[l] = starts[l] + max(1, counts[l]) * per_word
        nxt = starts[l + 1] if l + 1 < line_count else None
        if nxt is not None and ends[l] > nxt:
            ends[l] = max(starts[l] + 0.2, nxt)
    return list(zip(starts, ends))


def plan_line_windows(line_times, sources, duration, before=0.8, after=0.5):
    """One alignment window per line whose start is trusted (LRC or ASR).

    A line runs from just before its own start to just after the next line's
    start (its last vowel can be held right up to the next line). Lines whose
    start was only interpolated are merged into their trusted neighbour's window
    so the aligner can place them freely between trusted lines.
    """
    groups: list[list[int]] = []
    leading: list[int] = []
    for l, source in enumerate(sources):
        if source != 'interpolated':
            groups.append(leading + [l])
            leading = []
        elif groups:
            groups[-1].append(l)
        else:
            leading.append(l)
    if leading:
        groups.append(leading)
    out = []
    for n, group in enumerate(groups):
        first, last = group[0], group[-1]
        t0 = line_times[first][0] - before
        if n:
            t0 = max(t0, line_times[groups[n - 1][0]][0] + 0.1)
        nxt = line_times[groups[n + 1][0]][0] if n + 1 < len(groups) else None
        t1 = (nxt + after) if nxt is not None else line_times[last][1] + 4.0
        t1 = min(t1, line_times[last][0] + 15.0 + 3.0 * (last - first)) if nxt is not None and nxt - line_times[last][0] > 15 else t1
        out.append((first, last, max(0.0, t0), min(duration, max(t0 + 0.5, t1))))
    return out


def plan_blocks(line_times, duration, max_block=15.0, gap=6.0, pad=1.5):
    """Group lines into alignment blocks; returns [(first_line, last_line, t0, t1)]."""
    blocks, current = [], []
    for l, (s, e) in enumerate(line_times):
        if current:
            first_s = line_times[current[0]][0]
            prev_e = line_times[current[-1]][1]
            if s - prev_e > gap or e - first_s > max_block:
                blocks.append(current)
                current = []
        current.append(l)
    if current:
        blocks.append(current)
    out = []
    for n, block in enumerate(blocks):
        t0 = max(0.0, line_times[block[0]][0] - pad)
        t1 = min(duration, line_times[block[-1]][1] + pad + 1.0)
        if n:
            t0 = max(t0, (line_times[blocks[n - 1][-1]][1] + line_times[block[0]][0]) / 2 - pad)
        if n + 1 < len(blocks):
            t1 = min(t1, line_times[blocks[n + 1][0]][0] - 0.1)
        out.append((block[0], block[-1], t0, max(t0 + 0.5, t1)))
    return out


def envelope(samples: np.ndarray, rate: int, hop=FRAME_HOP, win=0.02) -> np.ndarray:
    """RMS energy in dB at `hop` resolution, centred frames."""
    size = max(1, int(round(win * rate)))
    padded = np.pad(samples.astype(np.float32), (size // 2, size // 2))
    squares = np.concatenate([[0.0], np.cumsum(padded.astype(np.float64) ** 2)])
    frames = int((len(samples) / rate) / hop) + 1
    # Exact frame positions (k * hop seconds): a rounded integer step drifts
    # (220 vs 220.5 samples at 44.1 kHz is 0.45 s by the end of a song).
    starts = np.minimum(np.round(np.arange(frames) * hop * rate).astype(np.int64), len(padded) - size)
    rms = np.sqrt(np.maximum(squares[starts + size] - squares[starts], 0) / size)
    return 20 * np.log10(rms + 1e-9)


def activity_threshold(env_db: np.ndarray, hop=FRAME_HOP, local=1.5, floor_margin=10.0, below_peak=26.0):
    """Per-frame 'the singer is making sound' level that follows the song's dynamics.

    A single threshold per block treats a soft intro next to a loud chorus as
    silence. Instead: the larger of (true silence floor + margin) and (loudest
    vocal within +-`local` s minus `below_peak` dB).
    """
    floor = float(np.percentile(env_db, 2))
    radius = max(1, int(local / hop))
    padded = np.pad(env_db, (radius, radius), constant_values=-200.0)
    windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * radius + 1)
    # max over a window = max of block maxima; stride keeps memory linear.
    stride = max(1, radius // 8)
    coarse = windows[::stride].max(axis=1)
    local_peak = np.repeat(coarse, stride)[:len(env_db)]
    if len(local_peak) < len(env_db):
        local_peak = np.pad(local_peak, (0, len(env_db) - len(local_peak)), mode='edge')
    return np.maximum(floor + floor_margin, local_peak - below_peak)


def attacks(env_db: np.ndarray, hop=FRAME_HOP) -> np.ndarray:
    """Frames where energy rose >= EDGE_DB over the previous EDGE_SPAN (note/consonant attacks)."""
    k = max(1, int(round(EDGE_SPAN / hop)))
    padded = np.pad(env_db, (k, 0), mode='edge')
    trough = np.lib.stride_tricks.sliding_window_view(padded, k + 1)[:, :k].min(axis=1)
    return (env_db - trough[:len(env_db)]) >= EDGE_DB


def releases(env_db: np.ndarray, hop=FRAME_HOP) -> np.ndarray:
    """Frames after which energy falls >= EDGE_DB within EDGE_SPAN (note releases)."""
    k = max(1, int(round(EDGE_SPAN / hop)))
    padded = np.pad(env_db, (0, k), mode='edge')
    trough = np.lib.stride_tricks.sliding_window_view(padded, k + 1)[:, 1:].min(axis=1)
    return (env_db - trough[:len(env_db)]) >= EDGE_DB


def phrase_attack(env_db, threshold, attack, estimate, hop=FRAME_HOP, reach=0.6, gap=0.04):
    """The sung-phrase start nearest `estimate`: an attack right after near-silence.

    Returns seconds or None. Used to pin a line whose time code we trust onto the
    exact frame the singer comes in, which the CTC aligner alone misses under
    choirs, doubled vocals and long held notes.
    """
    n = len(env_db)
    lo, hi = max(1, int((estimate - reach) / hop)), min(n - 1, int((estimate + reach) / hop))
    quiet = max(1, int(gap / hop))
    best = None
    for k in range(lo, hi):
        if not attack[k] or attack[k - 1]:
            continue
        before = env_db[max(0, k - quiet - 4):k]
        if len(before) and (before < threshold[max(0, k - quiet - 4):k]).sum() >= quiet:
            t = k * hop
            if best is None or abs(t - estimate) < abs(best - estimate):
                best = t
    return best


def refine_edges(words: list[dict], env_db: np.ndarray, hop=FRAME_HOP, threshold=None) -> None:
    """Move CTC boundaries onto the vocal envelope in place.

    start: onto the onset of the vocal energy (consonant attack), never before the
    previous word's end; if CTC put the start in silence, move to the first sound.
    end: through the sustained vowel until the energy stops, drops DROP_DB below the
    word's own peak, or reaches the next word. CTC alone ends a sung word at the
    first frames of its vowel, so this is where most of the end-time accuracy comes from.
    """
    n = len(env_db)
    if not words or not n:
        return
    if threshold is None:
        threshold = activity_threshold(env_db, hop)
    active = env_db > threshold
    attack = attacks(env_db, hop)
    release = releases(env_db, hop)

    def frame(t):
        return int(min(n - 1, max(0, round(t / hop))))

    timed = [w for w in words if w.get('ctc_start') is not None]
    for i, w in enumerate(timed):
        s, e = w['ctc_start'], w['ctc_end']
        prev_end = timed[i - 1]['end'] if i else 0.0
        nxt = timed[i + 1] if i + 1 < len(timed) else None
        next_start = nxt['ctc_start'] if nxt else min((n - 1) * hop, e + MAX_SUSTAIN)
        lo, hi = frame(s), frame(e)
        if not active[lo:hi + 1].any():
            # No audible vocal under this word at all: keep the aligner's span.
            begin = max(s, prev_end)
            w.update(start=round(begin, 3), end=round(max(e, begin + MIN_WORD), 3),
                     start_source='ctc', end_source='ctc')
            continue

        si = lo
        line_initial = not i or timed[i - 1].get('line') != w.get('line')
        held = None
        if line_initial:
            # A held first vowel ("Buuurning") makes CTC latch onto the word's tail.
            # Walk back through continuous sound to the attack that began it.
            floor_i = frame(max(prev_end, s - HELD_ONSET_BACK))
            k = lo
            while k - 1 >= floor_i and active[k - 1]:
                k -= 1
                if attack[k]:
                    while k - 1 >= floor_i and attack[k - 1] and active[k - 1]:
                        k -= 1          # first frame of the attack, not its peak
                    held = k
                    break
        if held is not None and lo - held > frame(ONSET_BACK):
            si = held
            w['start_source'] = 'held-onset'
        elif active[si]:
            floor_i = frame(max(prev_end, s - ONSET_BACK))
            while si - 1 >= floor_i and active[si - 1]:
                si -= 1
                if attack[si]:
                    while si - 1 >= floor_i and attack[si - 1] and active[si - 1]:
                        si -= 1
                    break
            w['start_source'] = 'onset'
        else:
            limit = frame(min(s + ONSET_FORWARD, max(s, e - MIN_WORD)))
            j = si
            while j < limit and not active[j]:
                j += 1
            w['start_source'] = 'onset' if active[j] else 'ctc'
            si = j if active[j] else si
        start = max(si * hop, prev_end)

        ei = hi
        limit = frame(min(next_start - 0.01, e + MAX_SUSTAIN))
        if active[ei]:
            word_peak = float(env_db[si:ei + 1].max())
            while ei + 1 <= limit:
                if release[ei] and ei > hi:
                    # the note is being released: end where the fall completes
                    while ei + 1 <= limit and release[ei + 1] and active[ei + 1]:
                        ei += 1
                    break
                if active[ei + 1] and env_db[ei + 1] > word_peak - DROP_DB:
                    ei += 1
                    word_peak = max(word_peak, float(env_db[ei]))
                    continue
                # bridge a short dropout (vibrato dip, breathy consonant)
                ahead = [k for k in range(ei + 2, min(limit, ei + 1 + BRIDGE_FRAMES) + 1)
                         if active[k] and env_db[k] > word_peak - DROP_DB]
                if ahead:
                    ei = ahead[0]
                    continue
                break
            w['end_source'] = 'sustain'
        else:
            # CTC ran past the sound: end at the last audible frame of the word.
            while ei - 1 > si and not active[ei]:
                ei -= 1
            w['end_source'] = 'trimmed'
        end = max(ei * hop + hop, start + MIN_WORD)
        if nxt is not None:
            end = min(end, max(start + MIN_WORD, next_start - 0.005))
        w['start'], w['end'] = round(start, 3), round(end, 3)


def _ctc_window(model, dictionary, star, waveform, rate, words, idx, t0, t1, device, lead_star=True):
    """Align words[idx] inside [t0, t1]; returns {word index: (start, end, score)} or None."""
    import torch
    import torchaudio.functional as F

    clip = waveform[:, int(t0 * rate):int(t1 * rate)]
    if clip.shape[-1] < rate // 4:
        return None
    with torch.inference_mode():
        emission, _ = model(clip.to(device))
    tokens, owners = ([star], [None]) if lead_star else ([], [])
    previous_line = None
    for i in idx:
        if previous_line is not None and words[i]['line'] != previous_line:
            tokens.append(star)
            owners.append(None)
        previous_line = words[i]['line']
        for ch in words[i]['spoken'].replace(' ', ''):
            if ch in dictionary:
                tokens.append(dictionary[ch])
                owners.append(i)
    tokens.append(star)
    owners.append(None)
    if emission.shape[1] < len(tokens):
        return None
    targets = torch.tensor([tokens], dtype=torch.int32, device=emission.device)
    try:
        path, scores = F.forced_align(emission.float(), targets, blank=0)
    except RuntimeError:
        return None
    spans = F.merge_tokens(path[0].cpu(), scores[0].exp().cpu())
    if len(spans) != len(tokens):
        return None
    ratio = clip.shape[-1] / rate / emission.shape[1]
    by_word: dict[int, list] = {}
    for owner, span in zip(owners, spans):
        if owner is not None:
            by_word.setdefault(owner, []).append(span)
    out = {}
    for i, chars in by_word.items():
        length = sum(c.end - c.start for c in chars) or 1
        out[i] = (round(t0 + chars[0].start * ratio, 3), round(t0 + chars[-1].end * ratio, 3),
                  round(float(sum(c.score * (c.end - c.start) for c in chars) / length), 4))
    return out


def _crammed(placed: dict, count: int) -> bool:
    """A sung line squeezed into a sliver means the aligner gave up on it."""
    if not placed:
        return True
    span = max(e for _, e, _ in placed.values()) - min(s for s, _, _ in placed.values())
    return count >= 3 and span < 0.13 * count


def ctc_align(model, dictionary, star, waveform, rate, words, blocks, device='cpu', duration=None):
    """Force-align `words` (from tokenize_lines) window by window with MMS_FA.

    Windows are processed in order; each starts no earlier than the previous
    window's last placed word, so lines can never swap or overlap. Star tokens
    at both ends (and between lines) absorb ad-libs, intros and the neighbouring
    line's audio. A line that comes back crammed into a sliver is retried with a
    wider window. Sets ctc_start/ctc_end/score/window on every word it can place.
    `waveform` is a mono 16 kHz float tensor [1, samples].
    """
    duration = duration or waveform.shape[-1] / rate
    floor_t = 0.0
    for block in blocks:
        first, last, t0, t1 = block[:4]
        pinned = block[4] if len(block) > 4 else None
        everything = [i for i, w in enumerate(words) if first <= w['line'] <= last]
        # Backing-vocal echoes "(worst you take)" are a second voice that overlaps
        # the next lead line; aligning them inline drags the lead late. Lead first.
        idx = [i for i in everything if not words[i]['backing']] or everything
        echoes = [i for i in everything if i not in idx]
        if not idx:
            continue
        t0 = max(t0, floor_t)
        if t1 - t0 < 0.3 + 0.2 * len(idx):
            t1 = t0 + 0.3 + 0.35 * len(idx)
        placed, retried = None, False
        attempts = [(0.0, True), (1.5, True), (3.0, True)]
        if pinned is not None and pinned >= floor_t - 0.02:
            # Start exactly at the phrase attack, no star to swallow the first word.
            attempts.insert(0, ('pin', False))
        for widen, lead_star in attempts:
            if widen == 'pin':
                a, b = max(0.0, pinned - 0.06), t1
            else:
                a, b = max(floor_t, t0 - widen), min(duration, t1 + widen)
            placed = _ctc_window(model, dictionary, star, waveform, rate, words, idx, a, b, device, lead_star)
            if placed and not _crammed(placed, len(idx)):
                if widen == 'pin':
                    first_word = min(placed)
                    s0, e0, sc = placed[first_word]
                    placed[first_word] = (min(s0, pinned), e0, sc)
                    for i in idx:
                        words[i]['pinned'] = round(pinned, 3)
                break
            retried = True
        if not placed:
            continue
        for i, (start, end, score) in placed.items():
            words[i].update(ctc_start=start, ctc_end=end, score=score, block_start=a, block_end=b,
                            ctc_retried=retried)
        floor_t = max(floor_t, max(v[0] for v in placed.values()) + 0.05)
        if echoes:
            lead_end = max(v[1] for v in placed.values())
            ea, eb = max(0.0, lead_end - 0.3), min(duration, max(t1, lead_end + 0.5) + 2.0)
            echo = _ctc_window(model, dictionary, star, waveform, rate, words, echoes, ea, eb, device)
            for i, (start, end, score) in (echo or {}).items():
                words[i].update(ctc_start=start, ctc_end=end, score=score, block_start=ea, block_end=eb)


def fill_unaligned(words: list[dict]) -> None:
    """Words CTC could not place get interpolated times and are marked for review."""
    timed = [i for i, w in enumerate(words) if w.get('start') is not None]
    for i, w in enumerate(words):
        if w.get('start') is not None:
            continue
        before = max((k for k in timed if k < i), default=None)
        after = min((k for k in timed if k > i), default=None)
        t0 = words[before]['end'] if before is not None else (words[after]['start'] - 0.5 if after is not None else 0.0)
        t1 = words[after]['start'] if after is not None else t0 + 0.5
        w.update(start=round(t0, 3), end=round(max(t0 + MIN_WORD, t1), 3),
                 start_source='interpolated', end_source='interpolated', score=0.0)


def review_flags(words, anchors):
    for w, anchor in zip(words, anchors):
        reasons = []
        if w.get('score') is not None and w['score'] < 0.01 and w.get('start_source') != 'onset':
            reasons.append('low acoustic match')
        if anchor and not w['backing'] and abs(w['start'] - anchor['start']) > 1.5 and (w.get('score') or 0) < 0.2:
            reasons.append('disagrees with speech recognition')
        if w.get('start_source') == 'interpolated':
            reasons.append('not found in audio')
        if w['end'] - w['start'] > 6.0:
            reasons.append('unusually long')
        w['review'] = reasons


def align(model, dictionary, star, waveform, rate, env_samples, env_rate,
          lines, asr_words, lrc_times=None, duration=None, device='cpu'):
    """Full alignment of official lyric lines to an isolated vocal recording."""
    words = tokenize_lines(lines)
    if not words:
        return {'words': [], 'lines': []}
    duration = duration or waveform.shape[-1] / rate
    anchors = anchor_words(words, asr_words)
    sources: list[str] = []
    line_times = estimate_lines(words, anchors, len(lines), lrc_times, duration, sources=sources)
    env = envelope(env_samples, env_rate)
    threshold = activity_threshold(env)
    attack = attacks(env)
    if sum(x != 'interpolated' for x in sources) >= max(1, len(lines) // 3):
        blocks = plan_line_windows(line_times, sources, duration)
        pinned = 0
        for n, block in enumerate(blocks):
            first = block[0]
            if sources[first] == 'lrc':
                t = phrase_attack(env, threshold, attack, line_times[first][0])
                if t is not None:
                    blocks[n] = (*block, t)
                    pinned += 1
        window_mode = f'per-line ({pinned} pinned to phrase attacks)'
    else:
        blocks = plan_blocks(line_times, duration)
        window_mode = 'blocks'
    ctc_align(model, dictionary, star, waveform, rate, words, blocks, device, duration)
    lead = [w for w in words if not w['backing']]
    refine_edges(lead, env, threshold=threshold)
    # Echo words overlap the lead, so the vocal envelope can't separate their
    # edges; keep each echo inside its aligned span, ordered and non-overlapping.
    echo = [w for w in words if w['backing'] and w.get('ctc_start') is not None]
    for n, w in enumerate(echo):
        start = max(w['ctc_start'], echo[n - 1]['end'] if n and echo[n - 1]['line'] == w['line'] else 0)
        nxt = echo[n + 1] if n + 1 < len(echo) and echo[n + 1]['line'] == w['line'] else None
        end = max(start + MIN_WORD, (nxt['ctc_start'] if nxt else w['ctc_end'] + 0.25) - 0.005)
        w.update(start=round(start, 3), end=round(end, 3), start_source='ctc', end_source='ctc')
    fill_unaligned(words)
    review_flags(words, anchors)
    for l, (estimate, _) in enumerate(line_times):
        first = next((w for w in words if w['line'] == l), None)
        if first and sources[l] != 'interpolated' and abs(first['start'] - estimate) > 1.5:
            first['review'].append(f'line starts {first["start"] - estimate:+.1f}s from its time code')
    out_lines = []
    for l, text in enumerate(lines):
        members = [w for w in words if w['line'] == l]
        out_lines.append({'text': text, 'start': members[0]['start'] if members else None,
                          'end': members[-1]['end'] if members else None})
    clean = [{k: w.get(k) for k in ('text', 'line', 'backing', 'start', 'end', 'ctc_start', 'ctc_end',
                                    'score', 'start_source', 'end_source', 'review')} for w in words]
    matched = sum(1 for a in anchors if a)
    return {'words': clean, 'lines': out_lines,
            'stats': {'words': len(words), 'asr_anchors': matched, 'window_mode': window_mode,
                      'line_sources': {k: sources.count(k) for k in sorted(set(sources))},
                      'aligned': sum(1 for w in words if w.get('ctc_start') is not None),
                      'review': sum(1 for w in words if w['review'])},
            'method': 'mms-fa-ctc + envelope-edge-refinement v1'}


def _merge_compounds(ref: list[str], hyp: list[str]) -> tuple[list[str], list[str]]:
    """'star fall' vs 'starfall', 'good night' vs 'goodnight' are the same sung words."""
    def merge(tokens, vocabulary):
        out, i = [], 0
        while i < len(tokens):
            if i + 1 < len(tokens) and tokens[i] + tokens[i + 1] in vocabulary:
                out.append(tokens[i] + tokens[i + 1])
                i += 2
            else:
                out.append(tokens[i])
                i += 1
        return out
    return merge(ref, set(hyp)), merge(hyp, set(ref))


def word_error_rate(reference: list[str], hypothesis: list[str], optional: list[bool] | None = None) -> dict:
    """Word edit distance on normalized keys. `optional` reference words (backing
    echoes) cost nothing to skip, so a take is scored on its lead lyric."""
    pairs = [(key(w), bool(optional[i]) if optional else False) for i, w in enumerate(reference) if key(w)]
    hyp = [key(w) for w in hypothesis if key(w)]
    vocabulary = set(hyp)
    merged, i = [], 0
    while i < len(pairs):
        if i + 1 < len(pairs) and pairs[i][0] + pairs[i + 1][0] in vocabulary:
            merged.append((pairs[i][0] + pairs[i + 1][0], pairs[i][1] and pairs[i + 1][1]))
            i += 2
        else:
            merged.append(pairs[i])
            i += 1
    pairs = merged
    _, hyp = _merge_compounds([k for k, _ in pairs], hyp)
    ref = [k for k, _ in pairs]
    skip = [1 - o for _, o in pairs]       # deletion cost per reference word
    d = list(range(len(hyp) + 1))
    for i in range(1, len(ref) + 1):
        prev, d[0] = d[0], d[0] + skip[i - 1]
        for j in range(1, len(hyp) + 1):
            cur = d[j]
            d[j] = min(d[j] + skip[i - 1], d[j - 1] + 1, prev + (ref[i - 1] != hyp[j - 1]))
            prev = cur
    errors = d[len(hyp)]
    required = sum(skip) or 1
    return {'wer': round(min(1.0, errors / required), 4), 'errors': errors, 'reference_words': sum(skip),
            'optional_words': len(ref) - sum(skip)}


def lyric_check(lines: list[str], heard: list[str]) -> dict:
    """How well a take sang the lead lyric; backing echoes in ( ) are optional."""
    words = tokenize_lines(lines)
    return word_error_rate([w['text'] for w in words], heard, [w['backing'] for w in words])


def to_lrc(words: list[dict], lines: list[str]) -> str:
    """Enhanced (A2) LRC: line time plus <mm:ss.xx> per word."""
    def stamp(t):
        t = max(0.0, t)
        return f'{int(t // 60):02d}:{t % 60:05.2f}'
    out = []
    for l, text in enumerate(lines):
        members = [w for w in words if w['line'] == l]
        if not members:
            continue
        parts = [f'<{stamp(w["start"])}>{w["text"]}' for w in members]
        out.append(f'[{stamp(members[0]["start"])}]' + ' '.join(parts) + f' <{stamp(members[-1]["end"])}>')
    return '\n'.join(out) + '\n'


__all__ = ['align', 'tokenize_lines', 'anchor_words', 'estimate_lines', 'plan_blocks', 'envelope',
           'refine_edges', 'word_error_rate', 'to_lrc', 'spoken', 'key']
