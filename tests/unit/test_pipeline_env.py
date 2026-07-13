"""Seed unit test: pipeline_env.load_env parsing contract.

Characterizes current .env parsing behavior so it survives the refactor.
"""

import os

from conftest import load_script

pipeline_env = load_script("pipeline_env.py")


def test_load_env_parses_and_strips_quotes(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text(
        "# a comment\n"
        "\n"
        "PLAIN=value\n"
        'QUOTED="quoted value"\n'
        "SINGLE='single'\n"
        "export EXPORTED=exported\n"
        "NO_EQUALS_LINE\n",
        encoding="utf-8",
    )
    for key in ("PLAIN", "QUOTED", "SINGLE", "EXPORTED"):
        monkeypatch.delenv(key, raising=False)

    returned = pipeline_env.load_env(env)

    assert returned == env
    assert os.environ["PLAIN"] == "value"
    assert os.environ["QUOTED"] == "quoted value"
    assert os.environ["SINGLE"] == "single"
    assert os.environ["EXPORTED"] == "exported"


def test_load_env_does_not_override_existing(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("ALREADY_SET=from_file\n", encoding="utf-8")
    monkeypatch.setenv("ALREADY_SET", "from_shell")

    pipeline_env.load_env(env)

    assert os.environ["ALREADY_SET"] == "from_shell"


def test_load_env_missing_file_returns_none(tmp_path):
    assert pipeline_env.load_env(tmp_path / "does-not-exist.env") is None
