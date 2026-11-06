"""Durable, per-song files. Only explicit future deletion should remove them."""
from dataclasses import asdict, dataclass, field
import json
import logging
import os
from pathlib import Path
import time
import uuid

ROOT = Path(os.environ.get("SESSION_FILES_DIR", Path(__file__).resolve().parents[1] / "session-files"))


@dataclass
class Session:
    id: str
    source_path: str
    tmp_dir: str  # Kept as the working-directory field for existing callers.
    source_name: str
    vocals_path: str | None = None
    melody: dict | None = None
    lyrics: str | None = None
    transcription: dict | None = None
    ready: bool = False
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    generations: list = field(default_factory=list)
    active_job_id: str | None = None
    active_batch_id: str | None = None
    lyrics_reviewed: bool = False


def save(session: Session) -> None:
    folder = Path(session.tmp_dir)
    folder.mkdir(parents=True, exist_ok=True)
    data = asdict(session)
    data.pop("tmp_dir")
    data["source_path"] = Path(session.source_path).name
    data["vocals_path"] = "stems/vocals.wav" if session.vocals_path else None
    data["pipeline_version"] = 1
    temporary = folder / f".session-{uuid.uuid4().hex}.tmp"
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(folder / "session.json")


def load_all(root: Path | None = None) -> dict[str, Session]:
    root = ROOT if root is None else root
    root.mkdir(parents=True, exist_ok=True)
    result = {}
    for manifest in root.glob("*/session.json"):
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            data.pop("pipeline_version", None)
            source = Path(data["source_path"])
            if source.name != str(source):
                raise ValueError("Session source must be a filename")
            data["source_path"] = str(manifest.parent / source)
            data["tmp_dir"] = str(manifest.parent)
            if data.get("vocals_path"):
                data["vocals_path"] = str(manifest.parent / "stems" / "vocals.wav")
            data["active_job_id"] = None
            data["active_batch_id"] = None
            session = Session(**data)
            if not Path(session.source_path).is_file():
                raise ValueError("Original recording is missing")
            if not session.vocals_path or not Path(session.vocals_path).is_file():
                session.vocals_path = None
                session.ready = False
                session.melody = None
            if not session.ready and not session.error:
                session.error = "Preparation was interrupted. Resume to reuse completed steps."
            for generation in session.generations:
                if generation.get("status") == "running":
                    generation["status"] = "interrupted"
            result[session.id] = session
        except (OSError, ValueError, TypeError, KeyError):
            logging.getLogger("ghost_band").exception("Cannot read saved session %s", manifest)
    return result
