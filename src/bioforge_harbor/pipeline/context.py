"""RunContext: what every stage receives (backend, prompts, paths).

Keeps stages honest: a stage gets its prompt text from ``load_prompt`` (never
inlines a prompt string), records its report to ``artifact_path`` (so the audit
trail is on disk), and talks to a model only through ``backend``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..backends.base import AgentBackend
from ..util.io import write_json, read_json, sha256_text

REPO_ROOT = Path(__file__).resolve().parents[3]   # .../rsibio-paper2env
PROMPTS_DIR = REPO_ROOT / "prompts"


@dataclass
class RunContext:
    paper_id: str
    run_dir: Path
    backend: AgentBackend
    prompts_dir: Path = field(default_factory=lambda: PROMPTS_DIR)

    def __post_init__(self):
        self.run_dir = Path(self.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def stage_dir(self, stage: str) -> Path:
        d = self.run_dir / stage
        d.mkdir(parents=True, exist_ok=True)
        return d

    def load_prompt(self, name: str) -> str:
        p = Path(self.prompts_dir) / f"{name}.md"
        if not p.is_file():
            raise FileNotFoundError(f"prompt not found: {p}")
        return p.read_text(encoding="utf-8")

    def write_json(self, stage: str, filename: str, obj: Any) -> Path:
        path = self.stage_dir(stage) / filename
        write_json(path, obj)
        return path

    def read_json(self, stage: str, filename: str) -> Any:
        return read_json(self.stage_dir(stage) / filename)

    def has_artifact(self, stage: str, filename: str) -> bool:
        return (self.stage_dir(stage) / filename).is_file()


def digest(obj: Any) -> str:
    import json
    return sha256_text(json.dumps(obj, ensure_ascii=False, sort_keys=True))[:16]
