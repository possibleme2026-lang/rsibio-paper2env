"""Recover v2's per-step sampler checkpoint paths from the bridge log.

Each training step calls save_weights_for_sampler(name=f"pi-grpo-step{N}"),
and the bridge logs the path it was pointed at. The trainer never persisted
those paths (lost on kill), but the bridge log did. Reconstruct them so the
eval arm can still use a mid-run checkpoint.
"""
import re
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LOG = REPO / "out" / "rl_runs" / "pi_bridge" / "bridge_server.log"
OUT = REPO / "out" / "rl_runs" / "_archive" / "v2_step_samplers.json"

pat = re.compile(r"sampling from saved weights: (tinker://[\w:-]+/sampler_weights/(pi-grpo-step\d+))")
seen: dict[str, str] = {}
for line in LOG.read_text(encoding="utf-8", errors="replace").splitlines():
    m = pat.search(line)
    if m:
        path, name = m.group(1), m.group(2)
        seen.setdefault(name, path)

rows = [{"name": k, "path": seen[k]} for k in sorted(seen, key=lambda s: int(s.rsplit("step", 1)[1]))]
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(rows, indent=2), encoding="utf-8")
print(f"recovered {len(rows)} step sampler paths -> {OUT}")
for r in rows:
    print(" ", r["name"], "->", r["path"])
