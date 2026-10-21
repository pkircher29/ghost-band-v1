const $ = (id) => document.getElementById(id);

const dropzone = $("dropzone");
const fileInput = $("fileInput");
const sourcePlayer = $("sourcePlayer");
const sourceAudio = $("sourceAudio");
const sourceFilename = $("sourceFilename");
const styleGrid = $("styleGrid");
const runBtn = $("runBtn");
const progressBox = $("progress");
const progressPulse = $("progressPulse");
const progressElapsed = $("progressElapsed");
const progressStageLabel = $("progressStageLabel");
const resultsEmpty = $("resultsEmpty");
const styleResults = $("styleResults");
const toast = $("toast");
const statusDot = $("statusDot");
const statusText = $("statusText");
const vramGauge = $("vramGauge");
const vramNeedle = $("vramNeedle");
const vramPct = $("vramPct");
const vramLabel = $("vramLabel");
const transcribeBtn = $("transcribeBtn");
const lyricsPanel = $("lyricsPanel");
const lyricsText = $("lyricsText");
const stylesPanel = $("stylesPanel");
const lyricsProgress = $("lyricsProgress");
const lyricsPulse = $("lyricsPulse");
const lyricsElapsed = $("lyricsElapsed");
const lyricsStageLabel = $("lyricsStageLabel");

let selectedFile = null;
let sessionId = null;
let preparingSong = false;
let generatingCover = false;
let sessionReady = false;
let toastTimer = null;
const styleRowById = {};

function unlock(panel) {
  panel.classList.remove("is-locked");
}

function showToast(message) {
  toast.textContent = message;
  toast.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { toast.hidden = true; }, 6000);
}

function setBusy(btn, busyLabel, isBusy, idleLabel) {
  btn.disabled = isBusy;
  btn.textContent = isBusy ? busyLabel : idleLabel;
}

function formatElapsed(seconds) {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}

// ---- Source selection ----

function handleFile(file) {
  if (!file) return;
  if (preparingSong || generatingCover) { showToast("Wait for the current run to finish."); return; }
  sessionId = null;
  sessionReady = false;
  runBtn.disabled = true;
  $("vocalPreview").hidden = true;
  $("vocalAudio").removeAttribute("src");
  selectedFile = file;
  lyricsText.value = "";
  renderGenerationLyrics(null);
  if (sourceAudio.src.startsWith("blob:")) URL.revokeObjectURL(sourceAudio.src);
  sourceAudio.src = URL.createObjectURL(file);
  sourceFilename.textContent = file.name;
  sourcePlayer.hidden = false;
  transcribeBtn.disabled = false;
  transcribeBtn.textContent = "Prepare song";
  $("saveLyricsBtn").disabled = true;
  $("sessionHint").textContent = "This upload will create a new saved song.";
  $("whisperBtn").disabled = true;
  renderTiming(null);
  $("savedCovers").replaceChildren();
}

async function refreshSessions() {
  const res = await fetch("/api/sessions");
  if (!res.ok) throw new Error("Could not load saved songs");
  const songs = await res.json();
  const select = $("savedSongSelect");
  select.replaceChildren(new Option("Choose a saved song", ""));
  for (const song of songs) {
    select.add(new Option(`${song.source_name} — ${song.ready ? "ready" : "resume preparation"} · ${new Date(song.created_at * 1000).toLocaleString()}`, song.session_id));
  }
  if (sessionId) select.value = sessionId;
}

let persistedLyrics = "";

function showPreparedSong(data) {
  renderGenerationLyrics(data.generation_lyrics);
  sessionReady = data.ready;
  sessionId = data.session_id;
  persistedLyrics = data.lyrics || "";
  lyricsText.value = data.generation_lyrics?.lyrics || persistedLyrics;
  $("saveLyricsBtn").disabled = !data.ready;
  $("whisperBtn").disabled = !data.ready || !!data.active_job_id || !!data.active_batch_id;
  renderTiming(data.lyric_timing);
  $("vocalPreview").hidden = !data.ready;
  if (data.ready) {
    $("vocalAudio").src = data.vocals_url;
    $("melodySummary").textContent = `${data.melody.vocal_notes} vocal notes · ${data.melody.instrumental_notes} instrumental notes · direct SheetSage2 score from the full recording`;
    unlock(lyricsPanel);
    unlock(stylesPanel);
  }
  runBtn.disabled = !data.ready || !!data.active_batch_id;
  $("sessionHint").textContent = data.ready
    ? "Saved analysis loaded. Choose styles and make more covers without retranscribing."
    : (data.error || "Preparation is in progress. Resume to follow it.");
  $("savedCovers").replaceChildren();
  for (const run of data.generations || []) {
    for (const style of run.styles || []) {
      if (!style.filename) continue;
      const row = document.createElement("p");
      const link = document.createElement("a");
      link.href = `/completed/${encodeURIComponent(style.filename)}`;
      link.textContent = `${style.label} — ${new Date(run.created_at * 1000).toLocaleString()}`;
      link.target = "_blank";
      row.appendChild(link);
      $("savedCovers").appendChild(row);
    }
  }
}

function renderGenerationLyrics(prepared) {
  $("generationLyrics").textContent = prepared?.error || "Edit the lyrics, section labels, and line breaks directly above. Save lyrics keeps your version; Make covers also saves and uses your edits. Section-only edits preserve word timings.";
}

function renderTiming(timing) {
  $("tightenTimingBtn").disabled = !timing?.current;
  $("wordTimings").replaceChildren();
  $("timingSummary").textContent = timing ? timing.message : "No word timings saved yet. Use Whisper large-v3 to add them.";
  const savedLyrics = lyricsText.value;
  const expectedLyrics = persistedLyrics;
  for (const [index, word] of (timing?.words || []).entries()) {
    const group = document.createElement("span");
    group.style.cssText = "display:inline-flex;gap:3px;align-items:center";
    const button = document.createElement("button");
    button.type = "button";
    button.className = "btn";
    button.textContent = `${word.text} · ${word.start.toFixed(2)}–${word.end.toFixed(2)}s${word.alignment_status === 'review' ? ' · review' : ''}`;
    button.title = `Word confidence: ${Math.round(word.probability * 100)}%\n` + (word.note_indices.length
      ? word.note_indices.map(i => { const n = timing.notes[i]; return `MIDI ${n.pitch}: ${n.start.toFixed(2)}–${n.end.toFixed(2)}s`; }).join("\n")
      : "No overlapping vocal note detected");
    if (word.original_start !== undefined) {
      button.title += `\nOriginal: ${word.original_start.toFixed(2)}–${word.original_end.toFixed(2)}s\nAlignment score: ${word.alignment_score ?? 'unavailable'} · ${word.alignment_status}`;
    }
    if (word.text_edited) button.title += `\nCorrected text; confidence belongs to the original recognition: ${word.recognized_text}`;
    button.addEventListener("click", () => {
      sourceAudio.currentTime = word.start;
      sourceAudio.play().catch(e => showToast(e.message));
    });
    const edit = document.createElement("button");
    edit.type = "button";
    edit.className = "btn";
    edit.textContent = "Edit";
    edit.setAttribute("aria-label", `Edit word ${index + 1}: ${word.text}`);
    edit.addEventListener("click", () => {
      if (preparingSong || generatingCover) { showToast("Wait for the current run to finish."); return; }
      const input = document.createElement("input");
      input.value = word.text;
      input.setAttribute("aria-label", `Replacement for word ${index + 1}`);
      input.style.width = "130px";
      const save = document.createElement("button");
      save.type = "button";
      save.className = "btn";
      save.textContent = "Save word";
      const cancel = document.createElement("button");
      cancel.type = "button";
      cancel.className = "btn";
      cancel.textContent = "Cancel";
      cancel.onclick = () => group.replaceChildren(button, edit);
      save.onclick = async () => {
        const text = input.value.trim();
        if (!text || /\s/.test(text)) { showToast("Enter one word. Its timing will stay unchanged."); return; }
        const spans = [...savedLyrics.matchAll(/^[ \t]*\[[^\]\n]+\][ \t]*$|\S+/gm)]
          .filter(span => !/^[ \t]*\[/.test(span[0]));
        if (spans.length !== timing.words.length || lyricsText.value !== savedLyrics) {
          showToast("Save your lyric changes before editing a timed word."); return;
        }
        const span = spans[index];
        const lyrics = savedLyrics.slice(0, span.index) + text + savedLyrics.slice(span.index + span[0].length);
        save.disabled = true;
        preparingSong = true;
        lyricsText.disabled = true;
        try {
          const res = await fetch(`/api/sessions/${sessionId}/lyrics`, {
            method: "PUT", headers: {"Content-Type": "application/json"},
            body: JSON.stringify({lyrics, expected_lyrics: expectedLyrics}),
          });
          if (!res.ok) throw new Error(await res.text());
          lyricsText.value = lyrics;
          persistedLyrics = lyrics;
          const saved = await res.json();
          renderTiming(saved.lyric_timing);
          renderGenerationLyrics(saved.generation_lyrics);
          showToast("Word saved. Timing unchanged.");
        } catch (e) { showToast(e.message); save.disabled = false; }
        finally { preparingSong = false; lyricsText.disabled = false; }
      };
      input.onkeydown = e => {
        if (e.key === "Enter") { e.preventDefault(); save.click(); }
        if (e.key === "Escape") cancel.click();
      };
      group.replaceChildren(input, save, cancel);
      input.focus();
      input.select();
    });
    group.append(button, edit);
    $("wordTimings").appendChild(group);
  }
}

lyricsText.addEventListener("input", () => {
  $("generationLyrics").textContent = "Unsaved edits. Save lyrics or Make covers to keep this version.";
  $("tightenTimingBtn").disabled = true;
  $("wordTimings").replaceChildren();
  $("timingSummary").textContent = "Lyrics edited. Save to check whether the stored word timings still match.";
});

for (const action of ["whisperBtn", "tightenTimingBtn"]) $(action).addEventListener("click", async () => {
  if (!sessionId || preparingSong || generatingCover) return;
  preparingSong = true;
  $("whisperBtn").disabled = true;
  $("tightenTimingBtn").disabled = true;
  $("saveLyricsBtn").disabled = true;
  lyricsText.disabled = true;
  runBtn.disabled = true;
  try {
    $("timingSummary").textContent = action === 'tightenTimingBtn' ? 'Aligning word boundaries…' : "Transcribing with Whisper large-v3 and aligning words…";
    const route = action === 'tightenTimingBtn' ? 'tighten-timing' : 'transcribe-whisper';
    const res = await fetch(`/api/sessions/${sessionId}/${route}`, {method: "POST"});
    if (!res.ok) throw new Error(await res.text());
    const {job_id} = await res.json();
    await watchEvents(`/api/jobs/${job_id}/events`, data => {
      $("timingSummary").textContent = `${data.stage} · ${formatElapsed(data.elapsed_s)}`;
    }, () => {});
    const result = await fetch(`/api/jobs/${job_id}/result`);
    if (!result.ok) throw new Error(await result.text());
    showPreparedSong(await result.json());
  } catch (e) { showToast(e.message); $("timingSummary").textContent = "Timing update failed. Your previous lyrics and timings are retained.";
    $("tightenTimingBtn").disabled = !sessionReady;
  }
  finally {
    preparingSong = false;
    lyricsText.disabled = false;
    $("whisperBtn").disabled = !sessionReady;
    $("saveLyricsBtn").disabled = !sessionReady;
    runBtn.disabled = !sessionReady;
  }
});

$("refreshSessionsBtn").addEventListener("click", () => refreshSessions().catch(e => showToast(e.message)));
$("openSessionBtn").addEventListener("click", async () => {
  if (preparingSong || generatingCover) { showToast("Wait for the current run to finish."); return; }
  const id = $("savedSongSelect").value;
  if (!id) return;
  preparingSong = true;
  try {
    const res = await fetch(`/api/sessions/${id}`);
    if (!res.ok) throw new Error("Could not open this song");
    const data = await res.json();
    selectedFile = null;
    sourceAudio.src = data.source_url;
    sourceFilename.textContent = data.source_name;
    sourcePlayer.hidden = false;
    showPreparedSong(data);
    transcribeBtn.disabled = data.ready;
    transcribeBtn.textContent = data.ready ? "Song prepared" : "Resume preparation";
    const settings = data.generations?.at(-1)?.settings;
    if (settings) {
      $("cotSelect").value = settings.cot;
      $("seedInput").value = settings.seed;
      $("stepsInput").value = settings.num_inference_steps;
      $("flipKeyInput").checked = settings.flip_key;
      $("vocalSelect").value = settings.vocal;
      for (const input of styleGrid.querySelectorAll("input")) {
        input.checked = settings.styles.includes(input.value);
        input.closest("label").classList.toggle("is-checked", input.checked);
      }
      updateStyleLibrary();
    }
  } catch (err) { showToast(err.message); }
  finally { preparingSong = false; }
});

$("saveLyricsBtn").addEventListener("click", async () => {
  if (!sessionId) return;
  try {
    const res = await fetch(`/api/sessions/${sessionId}/lyrics`, {
      method: "PUT", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({lyrics: lyricsText.value, expected_lyrics: persistedLyrics}),
    });
    if (!res.ok) throw new Error(await res.text());
    const saved = await res.json();
    persistedLyrics = lyricsText.value;
    renderTiming(saved.lyric_timing);
    renderGenerationLyrics(saved.generation_lyrics);
    showToast("Lyrics saved.");
  } catch (err) { showToast(err.message); }
});
refreshSessions().catch(e => showToast(e.message));

dropzone.addEventListener("click", (e) => {
  if (e.target !== fileInput) fileInput.click();
});
dropzone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); }
});
fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  // A previously selected file must still trigger change after opening a saved song.
  fileInput.value = "";
  handleFile(file);
});

["dragenter", "dragover"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    e.stopPropagation();
    e.dataTransfer.dropEffect = "copy";
    dropzone.classList.add("is-dragover");
  })
);
["dragleave", "drop"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.remove("is-dragover");
  })
);
dropzone.addEventListener("drop", (e) => {
  e.stopPropagation();
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});

// ---- Style picker ----

let stylePresets = [];

function updateStyleLibrary() {
  const selected = new Set(selectedStyleIds());
  let visible = 0;
  for (const card of styleGrid.children) {
    const style = stylePresets.find(item => item.id === card.dataset.styleId);
    if (!style) continue;
    card.hidden = !StyleLibrary.matchesStyle(style, $("styleSearch").value, $("selectedOnly").checked, selected);
    if (!card.hidden) visible++;
  }
  $("styleSelectionSummary").textContent = `${visible} of ${stylePresets.length} styles shown · ${selected.size} selected for your batch`;
  $("styleEmpty").hidden = visible > 0 || stylePresets.length === 0;
  $("clearStyles").disabled = selected.size === 0;
  $("selectVisibleStyles").disabled = visible === 0;
}

$("styleSearch").addEventListener("input", updateStyleLibrary);
$("selectedOnly").addEventListener("change", updateStyleLibrary);
$("selectVisibleStyles").addEventListener("click", () => {
  for (const card of styleGrid.children) {
    if (card.hidden || !card.querySelector('input')) continue;
    card.querySelector('input').checked = true;
    card.querySelector('label').classList.add('is-checked');
  }
  updateStyleLibrary();
});
$("clearStyles").addEventListener("click", () => {
  for (const input of styleGrid.querySelectorAll('input')) {
    input.checked = false;
    input.closest('label').classList.remove('is-checked');
  }
  updateStyleLibrary();
});

async function loadStyles() {
  const previouslyChecked = new Set(selectedStyleIds());
  try {
    const res = await fetch("/api/styles/full");
    if (!res.ok) throw new Error('Could not load styles');
    const data = await res.json();
    stylePresets = data.styles;
    styleGrid.innerHTML = "";
    data.styles.forEach((style) => {
      const card = document.createElement("article");
      card.className = "style-card";
      card.dataset.styleId = style.id;
      const label = document.createElement("label");
      label.className = "style-option";
      label.dataset.styleId = style.id;
      const input = document.createElement("input");
      input.type = "checkbox";
      input.value = style.id;
      input.checked = previouslyChecked.has(style.id);
      label.classList.toggle("is-checked", input.checked);
      input.addEventListener("change", () => {
        label.classList.toggle("is-checked", input.checked);
        updateStyleLibrary();
      });
      label.appendChild(input);
      label.appendChild(document.createTextNode(style.label));
      card.appendChild(label);
      const description = document.createElement('p');
      description.className = 'style-description';
      description.textContent = StyleLibrary.describeStyle(style);
      card.appendChild(description);
      if (style.prompt) {
        const details = document.createElement('details');
        const summary = document.createElement('summary');
        summary.textContent = 'View sound recipe';
        const prompt = document.createElement('p');
        prompt.textContent = style.prompt;
        details.append(summary, prompt);
        card.appendChild(details);
      }
      styleGrid.appendChild(card);
    });
    updateStyleLibrary();
  } catch {
    styleGrid.innerHTML = '<span class="hint">Could not load style presets. Is the app server running?</span>';
  }
}
loadStyles();

function selectedStyleIds() {
  return Array.from(styleGrid.querySelectorAll("input:checked")).map((el) => el.value);
}

// ---- Style manager (human-editable canvas over styles.json) ----

const manageStylesBtn = $("manageStylesBtn");
const styleManager = $("styleManager");
const styleEditorList = $("styleEditorList");
const newStyleLabel = $("newStyleLabel");
const newStylePrompt = $("newStylePrompt");
const addStyleBtn = $("addStyleBtn");

function styleEditorStatus(card, message, isError) {
  let statusEl = card.querySelector(".style-editor-status");
  if (!statusEl) {
    statusEl = document.createElement("span");
    statusEl.className = "style-editor-status";
    card.appendChild(statusEl);
  }
  statusEl.textContent = message;
  statusEl.classList.toggle("is-error", Boolean(isError));
  if (!isError) setTimeout(() => { statusEl.textContent = ""; }, 2500);
}

async function loadStyleEditors() {
  try {
    const res = await fetch("/api/styles/full");
    const data = await res.json();
    styleEditorList.innerHTML = "";
    data.styles.forEach((style) => {
      const card = document.createElement("div");
      card.className = "style-editor-card";
      card.dataset.styleId = style.id;

      const idEl = document.createElement("span");
      idEl.className = "style-editor-id";
      idEl.textContent = style.id;
      card.appendChild(idEl);

      const labelField = document.createElement("label");
      labelField.className = "field";
      labelField.innerHTML = '<span class="field-label">Name</span>';
      const labelInput = document.createElement("input");
      labelInput.type = "text";
      labelInput.value = style.label;
      labelField.appendChild(labelInput);
      card.appendChild(labelField);

      const promptField = document.createElement("label");
      promptField.className = "field";
      promptField.innerHTML = '<span class="field-label">Prompt</span>';
      const promptInput = document.createElement("textarea");
      promptInput.rows = 2;
      promptInput.value = style.prompt;
      promptField.appendChild(promptInput);
      card.appendChild(promptField);

      const row = document.createElement("div");
      row.className = "row";

      const saveBtn = document.createElement("button");
      saveBtn.className = "btn btn-secondary";
      saveBtn.type = "button";
      saveBtn.textContent = "Save";
      saveBtn.addEventListener("click", async () => {
        try {
          const res = await fetch(`/api/styles/${encodeURIComponent(style.id)}`, {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ label: labelInput.value, prompt: promptInput.value }),
          });
          if (!res.ok) throw new Error((await res.json()).detail || "Save failed");
          styleEditorStatus(card, "Saved");
          loadStyles();
        } catch (err) {
          styleEditorStatus(card, err.message, true);
        }
      });
      row.appendChild(saveBtn);

      const deleteBtn = document.createElement("button");
      deleteBtn.className = "btn";
      deleteBtn.type = "button";
      deleteBtn.textContent = "Delete";
      deleteBtn.addEventListener("click", async () => {
        try {
          const res = await fetch(`/api/styles/${encodeURIComponent(style.id)}`, { method: "DELETE" });
          if (!res.ok) throw new Error((await res.json()).detail || "Delete failed");
          loadStyleEditors();
          loadStyles();
        } catch (err) {
          styleEditorStatus(card, err.message, true);
        }
      });
      row.appendChild(deleteBtn);

      card.appendChild(row);
      styleEditorList.appendChild(card);
    });
  } catch {
    styleEditorList.innerHTML = '<span class="hint">Could not load styles for editing.</span>';
  }
}

manageStylesBtn.addEventListener("click", () => {
  const opening = styleManager.hidden;
  styleManager.hidden = !opening;
  manageStylesBtn.textContent = opening ? "Hide style editor" : "Manage styles";
  if (opening) loadStyleEditors();
});

addStyleBtn.addEventListener("click", async () => {
  const label = newStyleLabel.value.trim();
  const prompt = newStylePrompt.value.trim();
  if (!label || !prompt) { showToast("Give the new style a name and a prompt."); return; }
  try {
    const res = await fetch("/api/styles", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ label, prompt }),
    });
    if (!res.ok) throw new Error((await res.json()).detail || "Could not add style.");
    newStyleLabel.value = "";
    newStylePrompt.value = "";
    loadStyleEditors();
    loadStyles();
  } catch (err) {
    showToast(err.message);
  }
});

// ---- Shared SSE progress watcher (single jobs and batches alike) ----

function watchEvents(eventsUrl, onProgress, onReconnecting) {
  return new Promise((resolve, reject) => {
    const source = new EventSource(eventsUrl);

    source.addEventListener("open", () => {
      if (onReconnecting) onReconnecting(false);
    });

    source.addEventListener("progress", (evt) => {
      onProgress(JSON.parse(evt.data));
    });

    source.addEventListener("final", (evt) => {
      const data = JSON.parse(evt.data);
      source.close();
      if (data.status === "error") {
        reject(new Error(data.error || "Cover run failed."));
      } else {
        resolve();
      }
    });

    // A dropped connection (a backgrounded tab throttling, a brief network
    // blip) is transient: the browser retries on its own as long as we
    // don't close() here. The batch keeps running server-side regardless of
    // whether anyone's watching, so just surface "reconnecting" and let it
    // recover instead of giving up on the first hiccup.
    source.onerror = () => {
      if (source.readyState === EventSource.CLOSED) {
        reject(new Error("Lost connection to the progress stream."));
      } else if (onReconnecting) {
        onReconnecting(true);
      }
    };
  });
}

function renderStyleResult(style) {
  let row = styleRowById[style.id];
  if (!row) {
    row = document.createElement("div");
    row.className = "style-result";
    row.innerHTML = `
      <span class="style-result-label">${style.label}</span>
      <span class="style-result-status"></span>
    `;
    styleResults.appendChild(row);
    styleRowById[style.id] = row;
  }

  row.className = `style-result is-${style.status}`;
  const statusEl = row.querySelector(".style-result-status");
  const existingAudio = row.querySelector("audio");
  const existingLink = row.querySelector("a");

  if (style.status === "queued") {
    statusEl.textContent = "queued";
  } else if (style.status === "running") {
    statusEl.textContent = "generating…";
  } else if (style.status === "error") {
    statusEl.textContent = style.error || "failed";
  } else if (style.status === "done" && style.filename) {
    statusEl.textContent = "";
    if (!existingAudio) {
      const audioEl = document.createElement("audio");
      audioEl.controls = true;
      audioEl.src = `/completed/${style.filename}`;
      row.appendChild(audioEl);
    }
    if (!existingLink) {
      const link = document.createElement("a");
      link.href = `/completed/${style.filename}`;
      link.textContent = style.filename;
      link.target = "_blank";
      row.appendChild(link);
    }
  }
}

// ---- Transcribe lyrics (review/edit before generating) ----

transcribeBtn.addEventListener("click", async () => {
  if ((!selectedFile && !sessionId) || generatingCover) return;

  preparingSong = true;
  runBtn.disabled = true;
  $("vocalPreview").hidden = true;
  setBusy(transcribeBtn, "Preparing…", true, "Prepare song");
  lyricsProgress.hidden = false;
  lyricsPulse.classList.remove("is-active");
  lyricsElapsed.textContent = "0:00";
  lyricsStageLabel.textContent = "Separating vocals (htdemucs_ft)…";

  const form = new FormData();
  if (selectedFile) form.append("file", selectedFile);

  try {
    const startRes = selectedFile
      ? await fetch("/api/covers/prepare", { method: "POST", body: form })
      : await fetch(`/api/sessions/${sessionId}/prepare`, { method: "POST" });
    if (!startRes.ok) {
      const detail = await startRes.text();
      throw new Error(detail || `Could not start (${startRes.status})`);
    }
    const { session_id, job_id } = await startRes.json();
    sessionId = session_id;
    selectedFile = null;

    await watchEvents(
      `/api/jobs/${job_id}/events`,
      (data) => {
        lyricsElapsed.textContent = formatElapsed(data.elapsed_s);
        lyricsStageLabel.textContent = data.stage;
        lyricsPulse.classList.toggle("is-active", data.status === "running");
      },
      (isReconnecting) => {
        if (isReconnecting) lyricsStageLabel.textContent = "Reconnecting… still transcribing";
      }
    );

    const res = await fetch(`/api/jobs/${job_id}/result`);
    if (!res.ok) {
      const detail = await res.text();
      throw new Error(detail || `Transcription failed (${res.status})`);
    }
    const data = await res.json();
    showPreparedSong(data);
    lyricsStageLabel.textContent = "Lyrics and both melody parts ready — review below";
  } catch (err) {
    lyricsStageLabel.textContent = "Song preparation failed";
    showToast(err.message || "Song preparation failed.");
  } finally {
    preparingSong = false;
    setBusy(transcribeBtn, "Preparing…", false, "Prepare song");
    transcribeBtn.disabled = sessionReady;
    transcribeBtn.textContent = sessionReady ? "Song prepared" : (sessionId ? "Resume preparation" : "Prepare song");
    lyricsPulse.classList.remove("is-active");
    refreshSessions().catch(e => showToast(e.message));
  }
});

// ---- Generate (uses the session from the transcribe step + edited lyrics) ----

runBtn.addEventListener("click", async () => {
  if (!sessionId) { showToast("Prepare the song first."); return; }
  if (preparingSong || generatingCover) return;
  const styleIds = selectedStyleIds();
  if (styleIds.length === 0) { showToast("Pick at least one style."); return; }

  generatingCover = true;
  setBusy(runBtn, "Running…", true, "Make covers");
  progressBox.hidden = false;
  progressPulse.classList.remove("is-active");
  progressElapsed.textContent = "0:00";
  progressStageLabel.textContent = "Starting…";
  resultsEmpty.hidden = true;
  styleResults.innerHTML = "";
  Object.keys(styleRowById).forEach((k) => delete styleRowById[k]);

  const form = new FormData();
  form.append("lyrics", lyricsText.value);
  form.append("lyrics_reviewed", "true");
  form.append("styles", JSON.stringify(styleIds));
  form.append("cot", $("cotSelect").value);
  form.append("seed", $("seedInput").value || "831001");
  form.append("num_inference_steps", $("stepsInput").value || "16");
  form.append("flip_key", $("flipKeyInput").checked ? "true" : "false");
  form.append("vocal", $("vocalSelect").value);

  try {
    const startRes = await fetch(`/api/covers/${sessionId}/generate`, { method: "POST", body: form });
    if (!startRes.ok) {
      const detail = await startRes.text();
      throw new Error(detail || `Could not start (${startRes.status})`);
    }
    const { batch_id } = await startRes.json();
    persistedLyrics = form.get("lyrics");
    $("vocalAudio").pause();

    await watchEvents(
      `/api/covers/${batch_id}/events`,
      (data) => {
        progressElapsed.textContent = formatElapsed(data.elapsed_s);
        progressPulse.classList.toggle("is-active", !!data.engine_active);
        progressStageLabel.textContent = data.stage;
        data.styles.forEach(renderStyleResult);
      },
      (isReconnecting) => {
        if (isReconnecting) {
          progressStageLabel.textContent += " (reconnecting… the run itself is still going)";
        }
      }
    );

    showToast("Cover run complete.");
  } catch (err) {
    showToast(err.message || "Cover run failed.");
  } finally {
    generatingCover = false;
    setBusy(runBtn, "Running…", false, "Make covers");
    progressPulse.classList.remove("is-active");
    refreshSessions().catch(e => showToast(e.message));
  }
});

// ---- Engine status ----

async function pollHealth() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    if (data.ok) {
      statusDot.className = "status-dot ok";
      const engine = [data.backend?.toUpperCase(), data.device].filter(Boolean).join(" · ");
      statusText.textContent = `${engine} · ${data.yue2_loaded ? "model ready" : "model loads on first use"}`;
    } else {
      statusDot.className = "status-dot bad";
      statusText.textContent = "engine unreachable";
    }
  } catch {
    statusDot.className = "status-dot bad";
    statusText.textContent = "engine unreachable";
  }
}

pollHealth();

// ---- VRAM gauge ----
// Needle sweeps -90deg (0% used, left, green) to +90deg (100% used, right,
// red) through 0deg (50%, top, yellow) -- matches the gradient arc, which
// always runs left(green) -> middle(yellow) -> right(red) regardless of the
// GPU: the scale is whatever memory.total nvidia-smi reports for the card
// actually installed, not a hardcoded VRAM size.
const VRAM_MAXED_PCT = 97;

async function pollVram() {
  try {
    const res = await fetch("/api/gpu/vram");
    const data = await res.json();
    if (!data.available) {
      vramGauge.hidden = true;
      return;
    }
    vramGauge.hidden = false;
    const pct = Math.max(0, Math.min(100, (data.used_mib / data.total_mib) * 100));
    const angle = (pct / 100) * 180 - 90;
    vramNeedle.style.transform = `rotate(${angle}deg)`;
    vramPct.textContent = `${Math.round(pct)}%`;
    vramLabel.textContent = `${data.used_mib.toLocaleString()} / ${data.total_mib.toLocaleString()} MiB`;
    vramGauge.title = data.name;
    vramGauge.classList.toggle("is-maxed", pct >= VRAM_MAXED_PCT);
  } catch {
    vramGauge.hidden = true;
  }
}

pollVram();
setInterval(pollVram, 1000);
setInterval(pollHealth, 15000);
