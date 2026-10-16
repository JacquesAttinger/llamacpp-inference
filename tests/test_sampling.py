# Last edited: 2026-10-16 07:44
"""Tests for how the sampling settings are read and checked."""

import pytest

from llamacppbench.sampling import LOCAL_PATH, SETTINGS_PATH, load_sampling

ALL = "top_k = 40\ntop_p = 0.95\nmin_p = 0.05\ntypical_p = 1.0\n"


def write(path, text):
    path.write_text(text)
    return path


def test_committed_file_has_all_four_settings():
    settings, overrides = load_sampling(SETTINGS_PATH, LOCAL_PATH.with_name("no_such_file.toml"))
    assert settings == {"top_k": 40, "top_p": 0.95, "min_p": 0.05, "typical_p": 1.0}
    assert overrides == {}


def test_local_file_changes_only_the_values_it_names(tmp_path):
    base = write(tmp_path / "sampling.toml", ALL)
    local = write(tmp_path / "sampling.local.toml", "top_k = 0\nmin_p = 0.05\n")
    settings, overrides = load_sampling(base, local)
    assert settings == {"top_k": 0, "top_p": 0.95, "min_p": 0.05, "typical_p": 1.0}
    assert overrides == {"top_k": 0}  # min_p matches the committed value, so it is not a change


def test_whole_number_is_accepted_for_a_fraction(tmp_path):
    base = write(tmp_path / "sampling.toml", ALL)
    local = write(tmp_path / "sampling.local.toml", "top_p = 1\n")
    settings, _ = load_sampling(base, local)
    assert settings["top_p"] == 1.0
    assert isinstance(settings["top_p"], float)


@pytest.mark.parametrize(
    "text",
    [
        "top_p = 1.5\n",
        "min_p = -0.1\n",
        "top_k = 2.5\n",
        "top_k = -1\n",
        'top_p = "high"\n',
        "top_k = true\n",
    ],
)
def test_bad_values_are_rejected(tmp_path, text):
    base = write(tmp_path / "sampling.toml", ALL)
    local = write(tmp_path / "sampling.local.toml", text)
    with pytest.raises(ValueError, match="sampling.local.toml"):
        load_sampling(base, local)


def test_unknown_setting_is_rejected(tmp_path):
    base = write(tmp_path / "sampling.toml", ALL)
    local = write(tmp_path / "sampling.local.toml", "top_q = 3\n")
    with pytest.raises(ValueError, match="unknown setting top_q"):
        load_sampling(base, local)


def test_committed_file_must_have_every_setting(tmp_path):
    base = write(tmp_path / "sampling.toml", "top_k = 40\n")
    with pytest.raises(ValueError, match="missing setting"):
        load_sampling(base, tmp_path / "none.toml")
