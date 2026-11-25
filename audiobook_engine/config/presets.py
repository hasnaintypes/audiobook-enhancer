"""Loads config/presets.yaml's tier_max cap per preset -- the piece that
actually makes the API's preset field do something (gap.md Step 9
follow-up: presets were defined but never enforced by run_job before this).

config/presets.yaml lives at the repo root, not inside this package, since
it's config data the Colab bootstrap notebook's working directory (the
cloned repo root) already resolves naturally -- same convention as
config/requirements.lock. Reading it is a plain local file read, not
network/subprocess, so unlike the rest of this package's impure I/O
(jobstore/storage.py, runner/job_runner.py's run_job) this IS covered by
the unit suite, against the real committed file.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from audiobook_engine.analysis.strategy import EnhancementTier

DEFAULT_PRESETS_PATH = Path("config/presets.yaml")


def load_preset_tier_caps(path: Path = DEFAULT_PRESETS_PATH) -> dict[str, EnhancementTier | None]:
    """Return {preset_name: tier_max or None}. A preset with no tier_max
    entry in the YAML (currently "custom") is absent from the file, so
    resolve_max_tier below returns None (no cap) for it rather than this
    function inventing a default.
    """
    data = yaml.safe_load(path.read_text())
    caps: dict[str, EnhancementTier | None] = {}
    for name, config in data.items():
        tier_max = config.get("tier_max")
        caps[name] = EnhancementTier(tier_max) if tier_max is not None else None
    return caps


def resolve_max_tier(
    preset: str, caps: dict[str, EnhancementTier | None] | None = None
) -> EnhancementTier | None:
    """caps=None loads the real config/presets.yaml; pass an explicit dict
    in tests instead of hitting the filesystem. A preset absent from caps
    (e.g. "custom") resolves to None -- no cap, not an error, since an
    unenforced/uncapped preset is a valid, already-decided outcome here.
    """
    if caps is None:
        caps = load_preset_tier_caps()
    return caps.get(preset)
