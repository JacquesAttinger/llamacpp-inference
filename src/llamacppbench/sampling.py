# Last edited: 2026-10-16 07:44
"""Read the sampling settings from sampling.toml, plus any changes in sampling.local.toml."""

import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SETTINGS_PATH = REPO_ROOT / "sampling.toml"
LOCAL_PATH = REPO_ROOT / "sampling.local.toml"  # optional, not in git
# Each setting and its allowed range. top_k is a whole number, the rest are numbers from 0 to 1.
INT_SETTINGS = ("top_k",)
FRACTION_SETTINGS = ("top_p", "min_p", "typical_p")
NAMES = INT_SETTINGS + FRACTION_SETTINGS


def check_value(name: str, value, path: Path):
    """Return the value as the right type, or raise ValueError that names the file and the setting."""
    if name in INT_SETTINGS:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{path.name}: {name} must be a whole number, 0 or more (got {value!r})")
        return value
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
        raise ValueError(f"{path.name}: {name} must be a number from 0 to 1 (got {value!r})")
    return float(value)


def read_file(path: Path) -> dict:
    """Read one TOML file and check every setting in it."""
    with path.open("rb") as f:
        raw = tomllib.load(f)
    unknown = sorted(set(raw) - set(NAMES))
    if unknown:
        raise ValueError(f"{path.name}: unknown setting {', '.join(unknown)} (known: {', '.join(NAMES)})")
    return {name: check_value(name, value, path) for name, value in raw.items()}


def load_sampling(settings_path: Path = SETTINGS_PATH, local_path: Path = LOCAL_PATH) -> tuple[dict, dict]:
    """Return (settings, overrides).

    settings has all four values. overrides is the part that the local file changed to something new.
    """
    settings = read_file(settings_path)
    missing = [name for name in NAMES if name not in settings]
    if missing:
        raise ValueError(f"{settings_path.name}: missing setting {', '.join(missing)}")
    overrides = {}
    if local_path.exists():
        local = read_file(local_path)
        overrides = {name: value for name, value in local.items() if value != settings[name]}
        settings.update(local)
    return settings, overrides
