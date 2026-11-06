'use strict';
const $ = (id) => document.getElementById(id);
const api = async (path, opts = {}) => {
  const r = await fetch(path, opts);
  if (!r.ok) {
    let detail = r.statusText;
    try { detail = (await r.json()).detail || detail; } catch (_) {}
    throw new Error(detail);
  }
  return r.json();
};
const json = (method, body) => ({ method, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
const esc = (s) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const clock = (t) => t == null ? '–' : `${Math.floor(t / 60)}:${(t % 60).toFixed(2).padStart(5, '0')}`;

let styles = [];
let current = null;        // song id
let song = null;           // last song payload
let aligned = null;        // aligned word timing
let selected = null;       // word index being edited
let picked = new Set();
let pollTimer = null;
let phrasingDirty = false;
const STAGES = [['vocals', 'Vocal + speech'], ['melody', 'Melody'], ['lyrics', 'Lyrics'], ['align', 'Word timing']];

// ------------------------------------------------------------- library
async function loadSongs() {
  const songs = await api('/api/songs');
  $('songs').innerHTML = songs.map((s) => `<li data-id="${s.id}" class="${s.id === current ? 'active' : ''}">
    ${esc(s.title || s.name)}<small>${esc(s.artist || '')}</small></li>`).join('');
}
$('songs').onclick = (e) => { const li = e.target.closest('li'); if (li) openSong(li.dataset.id); };

async function upload(file) {
  const form = new FormData();
  form.append('file', file);
  const s = await api('/api/songs', { method: 'POST', body: form });
  await loadSongs();
  openSong(s.id);
}
$('file').onchange = (e) => e.target.files[0] && upload(e.target.files[0]);
const drop = $('drop');
['dragenter', 'dragover'].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add('over'); }));
['dragleave', 'drop'].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove('over'); }));
drop.addEventListener('drop', (e) => e.dataTransfer.files[0] && upload(e.dataTransfer.files[0]));

// ------------------------------------------------------------- song view
async function openSong(id) {
  current = id; aligned = null; selected = null; phrasingDirty = false;
  $('empty').hidden = true; $('song').hidden = false;
  $('finder').hidden = true; $('editor').hidden = true; $('wordEditor').hidden = true;
  [...$('songs').children].forEach((li) => li.classList.toggle('active', li.dataset.id === id));
  await refresh(true);
}

function busy(s) {
  return Object.values(s.stages || {}).some((x) => x.status === 'running') ||
    (s.takes || []).some((t) => ['queued', 'running', 'checking'].includes(t.status));
}

async function refresh(full = false) {
  if (!current) return;
  clearTimeout(pollTimer);
  const id = current;
  try {
    const s = await api(`/api/songs/${id}`);
    if (id !== current) return;
    const alignChanged = !song || song.id !== s.id || JSON.stringify(song.stages?.align) !== JSON.stringify(s.stages?.align);
    song = s;
    renderHead();
    if (full || alignChanged) await loadLyrics();
    renderPhrasing();
    renderTakes();
  } catch (err) { console.error(err); }
  pollTimer = setTimeout(refresh, song && busy(song) ? 2000 : 8000);
}

function renderHead() {
  if (document.activeElement !== $('artist')) $('artist').value = song.artist || '';
  if (document.activeElement !== $('title')) $('title').value = song.title || '';
  $('stages').innerHTML = STAGES.map(([k, label]) => {
    const st = song.stages?.[k];
    const status = st?.status || 'waiting';
    const tip = st?.error || (st?.seconds ? `${st.seconds}s` : '');
    return `<span class="stage ${status}" title="${esc(tip)}">${label}: ${status}</span>`;
  }).join('') + (Object.values(song.stages || {}).some((x) => ['failed', 'interrupted'].includes(x.status))
    ? '<button class="ghost" id="resume">Retry</button>' : '');
  const r = $('resume');
  if (r) r.onclick = async () => { await api(`/api/songs/${current}/resume`, { method: 'POST' }); refresh(); };
  const lyr = song.lyrics;
  $('lyricsSource').textContent = lyr ? `${lyr.label || ''} · ${lyr.source}${lyr.synced ? ' · line time codes' : ''}` :
    (song.stages?.lyrics?.error || 'found automatically after analysis');
  const want = $('hearMix').checked ? `/api/songs/${current}/files/${song.source}` :
    (song.stages?.vocals?.status === 'done' ? `/api/songs/${current}/files/${song.vocals_preview || 'vocals.flac'}` : '');
  // Assigning src (even the same URL) reloads the media; only change it when it differs.
  if ((player.getAttribute('src') || '') !== want) {
    if (want) player.setAttribute('src', want); else player.removeAttribute('src');
  }
}

['artist', 'title'].forEach((k) => $(k).addEventListener('change', async () => {
  await api(`/api/songs/${current}`, json('PATCH', { [k]: $(k).value }));
  loadSongs();
}));
$('hearMix').onchange = () => { $('player').removeAttribute('src'); renderHead(); };

// ------------------------------------------------------------- lyrics
async function loadLyrics() {
  const data = await api(`/api/songs/${current}/lyrics`);
  aligned = data.aligned;
  if (data.lyrics && $('editor').hidden) $('lyricsText').value = data.lyrics.lines.join('\n');
  renderTimeline();
}

$('findLyrics').onclick = () => {
  $('finder').hidden = !$('finder').hidden;
  $('qArtist').value = song.artist || ''; $('qTitle').value = song.title || '';
  if (!$('finder').hidden) searchLyrics();
};
$('search').onclick = () => searchLyrics();
async function searchLyrics() {
  $('candidates').innerHTML = '<li class="muted">Searching…</li>';
  const params = new URLSearchParams({ artist: $('qArtist').value, title: $('qTitle').value });
  if (song.duration) params.set('duration', song.duration);
  try {
    const found = await api(`/api/lyrics/search?${params}`);
    window._candidates = found;
    $('candidates').innerHTML = found.length ? found.map((c, i) => `<li><div class="meta"><b>${esc(c.title)}</b> — ${esc(c.artist)}
      <small class="muted">${esc(c.album || '')} · ${c.duration ? Math.round(c.duration) + 's' : '?'}${c.duration_diff != null ? ` (±${c.duration_diff}s)` : ''}</small>
      ${c.synced ? '<span class="badge">time codes</span>' : ''}<span class="badge">${c.lines.length} lines</span>
      <div class="muted small">${esc(c.lines.slice(0, 2).join(' / '))}</div></div>
      <button data-use="${i}">Use</button></li>`).join('') : '<li class="muted">No matches. Try another spelling, or paste lyrics with “Edit text”.</li>';
  } catch (err) { $('candidates').innerHTML = `<li class="err">${esc(err.message)}</li>`; }
}
$('candidates').onclick = async (e) => {
  const b = e.target.closest('[data-use]'); if (!b) return;
  const c = window._candidates[+b.dataset.use];
  await api(`/api/songs/${current}/lyrics`, json('PUT', { lines: c.lines, times: c.times, source: c.source, label: `${c.artist} — ${c.title}` }));
  $('finder').hidden = true; phrasingDirty = false;
  refresh(true);
};
$('editLyrics').onclick = () => { $('editor').hidden = !$('editor').hidden; };
$('cancelLyrics').onclick = () => { $('editor').hidden = true; loadLyrics(); };
$('saveLyrics').onclick = async () => {
  await api(`/api/songs/${current}/lyrics`, json('PUT', { text: $('lyricsText').value, source: 'edited by you', label: song.lyrics?.label || '' }));
  $('editor').hidden = true; phrasingDirty = false;
  refresh(true);
};

// ------------------------------------------------------------- word timing
function renderTimeline() {
  const st = song.stages?.align;
  if (!aligned) {
    $('timeline').innerHTML = `<p class="muted">${st?.status === 'running' ? 'Timing every word against the isolated vocal…' :
      st?.error ? esc(st.error) : 'Appears once lyrics are found and the vocal is isolated.'}</p>`;
    $('alignStats').textContent = ''; $('lrcLink').hidden = true;
    return;
  }
  const s = aligned.stats || {};
  $('alignStats').textContent = `${s.words} words · ${s.aligned} aligned · ${s.review} to check`;
  $('lrcLink').hidden = false; $('lrcLink').href = `/api/songs/${current}/files/lyrics.lrc`;
  $('lrcLink').download = `${song.artist || 'song'} - ${song.title || ''}.lrc`;
  const byLine = aligned.lines.map(() => []);
  aligned.words.forEach((w, i) => byLine[w.line]?.push([w, i]));
  $('timeline').innerHTML = aligned.lines.map((line, n) => `<div class="line"><time data-t="${line.start}">${clock(line.start)}</time><div class="words">${
    byLine[n].map(([w, i]) => `<span class="w ${w.backing ? 'backing' : ''} ${w.review?.length ? 'review' : ''} ${i === selected ? 'selected' : ''}"
      data-i="${i}" title="${clock(w.start)} → ${clock(w.end)}${w.review?.length ? ' · ' + w.review.join(', ') : ''}">${esc(w.text)}</span>`).join('')}</div></div>`).join('');
}

let stopAt = null;
const player = $('player');
function play(start, end) {
  if (!player.src) return;
  player.currentTime = Math.max(0, start);
  stopAt = end;
  player.play();
}
player.addEventListener('timeupdate', () => {
  if (stopAt != null && player.currentTime >= stopAt) { player.pause(); stopAt = null; }
  if (!aligned) return;
  const t = player.currentTime;
  document.querySelectorAll('.w.playing').forEach((el) => el.classList.remove('playing'));
  const i = aligned.words.findIndex((w) => t >= w.start && t < w.end);
  if (i >= 0 && !player.paused) document.querySelector(`.w[data-i="${i}"]`)?.classList.add('playing');
});
$('timeline').onclick = (e) => {
  const t = e.target.closest('time');
  if (t) { play(+t.dataset.t, null); return; }
  const el = e.target.closest('.w'); if (!el) return;
  selected = +el.dataset.i;
  const w = aligned.words[selected];
  play(w.start, w.end);
  showWordEditor();
  renderTimeline();
};
function showWordEditor() {
  const w = aligned.words[selected];
  $('wordEditor').hidden = false;
  $('weWord').textContent = w.text;
  $('weWhy').textContent = `start from ${w.start_source || '?'}, end from ${w.end_source || '?'}` +
    (w.score != null ? ` · match ${(w.score * 100).toFixed(0)}%` : '') + (w.review?.length ? ` · ${w.review.join(', ')}` : '');
  $('weStart').value = w.start.toFixed(3); $('weEnd').value = w.end.toFixed(3);
}
$('wordEditor').onclick = async (e) => {
  const n = e.target.dataset.nudge;
  if (n) {
    const [edge, d] = n.split(':');
    const input = edge === 'start' ? $('weStart') : $('weEnd');
    input.value = (+input.value + +d).toFixed(3);
    const s = +$('weStart').value, en = +$('weEnd').value;
    edge === 'start' ? play(s, Math.min(en, s + 0.6)) : play(Math.max(s, en - 0.6), en);
  }
};
$('wePlay').onclick = () => play(+$('weStart').value, +$('weEnd').value);
$('wePlayStart').onclick = () => play(+$('weStart').value - 0.3, +$('weStart').value + 0.25);
$('wePlayEnd').onclick = () => play(+$('weEnd').value - 0.25, +$('weEnd').value + 0.3);
$('weSave').onclick = async () => {
  try {
    aligned = await api(`/api/songs/${current}/words`, json('PUT', [{ index: selected, start: +$('weStart').value, end: +$('weEnd').value }]));
    showWordEditor(); renderTimeline();
  } catch (err) { $('weWhy').innerHTML = `<span class="err">${esc(err.message)}</span>`; }
};

// ------------------------------------------------------------- phrasing
function renderPhrasing() {
  const g = song.generation || {};
  $('phrasingMethod').textContent = g.method || '';
  if (!phrasingDirty && document.activeElement !== $('genLyrics')) $('genLyrics').value = g.lyrics || '';
}
$('genLyrics').oninput = () => { phrasingDirty = true; };
$('savePhrasing').onclick = async () => {
  try {
    await api(`/api/songs/${current}/generation-lyrics`, json('PUT', { lyrics: $('genLyrics').value }));
    phrasingDirty = false; refresh();
  } catch (err) { alertInline(err.message); }
};
$('resetPhrasing').onclick = async () => {
  await api(`/api/songs/${current}/generation-lyrics`, json('PUT', { lyrics: null }));
  phrasingDirty = false; refresh();
};
function alertInline(message) { $('phrasingMethod').innerHTML = `<span class="err">${esc(message)}</span>`; }

// ------------------------------------------------------------- styles + takes
function renderStyles() {
  $('styleGrid').innerHTML = styles.map((s) => `<div class="style ${picked.has(s.id) ? 'on' : ''}" data-id="${s.id}" title="${esc(s.prompt)}">
    <b>${esc(s.label)}</b><small>${esc(s.family)}</small></div>`).join('');
  $('pickCount').textContent = `${picked.size} picked`;
}
$('styleGrid').onclick = (e) => {
  const el = e.target.closest('.style'); if (!el) return;
  picked.has(el.dataset.id) ? picked.delete(el.dataset.id) : picked.add(el.dataset.id);
  renderStyles();
};
$('make').onclick = async () => {
  if (phrasingDirty) await $('savePhrasing').onclick();
  $('make').disabled = true;
  try {
    await api(`/api/songs/${current}/covers`, json('POST', {
      styles: [...picked], custom_style: $('customStyle').value, vocal: $('vocal').value,
      melody: $('melodyMode').value, seed: $('seed').value === '' ? null : +$('seed').value,
    }));
    refresh();
  } catch (err) { $('takes').insertAdjacentHTML('afterbegin', `<li class="err">${esc(err.message)}</li>`); }
  $('make').disabled = false;
};

function renderTakes() {
  const takes = [...(song.takes || [])].reverse();
  const openAudio = new Set([...document.querySelectorAll('.take audio')].filter((a) => !a.paused).map((a) => a.dataset.id));
  if (openAudio.size) return; // never interrupt a take that is playing
  $('takes').innerHTML = takes.length ? takes.map((t) => {
    const acc = t.lyric_accuracy;
    const pct = acc ? Math.max(0, 100 - acc.wer * 100) : null;
    const cls = pct == null ? '' : pct >= 85 ? 'good' : pct >= 65 ? 'mid' : 'bad';
    return `<li class="take"><div class="take-head"><b>${esc(t.label)}</b>
      <span class="stage ${t.status === 'done' ? 'done' : t.status === 'failed' ? 'failed' : 'running'}">${t.status}</span>
      ${pct != null ? `<span class="acc ${cls}" title="Words heard in the take vs. the lyrics (Whisper on the isolated vocal)">lyrics heard: ${pct.toFixed(0)}%</span>` : ''}
      ${t.seconds ? `<span class="muted small">${Math.round(t.audio_seconds)}s song · ${Math.round(t.seconds)}s on GPU</span>` : ''}
      ${t.truncated && (t.truncated.abc || t.truncated.semantic) ? '<span class="err">hit generation limit</span>' : ''}
      <div class="spacer"></div>
      ${t.file ? `<a class="button ghost" href="/api/songs/${current}/files/${t.file}" download="${esc(song.title || 'cover')} (${esc(t.label)}).flac">FLAC</a>` : ''}
      ${t.lrc ? `<a class="button ghost" href="/api/songs/${current}/files/${t.lrc}" download="${esc(song.title || 'cover')} (${esc(t.label)}).lrc">LRC</a>` : ''}
      ${['done', 'failed', 'interrupted'].includes(t.status) ? `<button class="ghost" data-del="${t.id}">Remove</button>` : ''}</div>
      ${t.file ? `<audio controls preload="none" data-id="${t.id}" src="/api/songs/${current}/files/${t.preview || t.file}"></audio>` : ''}
      ${t.error ? `<div class="err">${esc(t.error)}</div>` : ''}
      <small class="muted">${esc(t.style_prompt)}</small></li>`;
  }).join('') : '<li class="muted">Pick styles and press “Make covers”. Each take is saved the moment it finishes.</li>';
}
$('takes').onclick = async (e) => {
  const b = e.target.closest('[data-del]'); if (!b) return;
  await api(`/api/songs/${current}/takes/${b.dataset.del}`, { method: 'DELETE' });
  refresh();
};

// ------------------------------------------------------------- boot
(async () => {
  styles = await api('/api/styles');
  renderStyles();
  await loadSongs();
  const first = $('songs').querySelector('li');
  if (first) openSong(first.dataset.id);
  const h = await api('/api/health').catch(() => ({ ok: false }));
  $('cloud').textContent = h.ok ? 'Modal connected' : 'Modal unreachable';
  $('cloud').className = `pill ${h.ok ? 'ok' : 'bad'}`;
  if (!h.ok) $('cloud').title = h.error || '';
})();
