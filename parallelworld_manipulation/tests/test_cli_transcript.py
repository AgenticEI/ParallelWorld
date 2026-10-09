from __future__ import annotations

import json

from parallelworld_manipulation.cli.main import _write_json_atomic


def test_write_json_atomic_replaces_an_existing_transcript(tmp_path) -> None:
    transcript = tmp_path / "transcript.json"
    transcript.write_text('{"attempt": 1}{"attempt": 2}', encoding="utf-8")

    _write_json_atomic(transcript, {"attempt": 3, "finish": {"_finish": True}})

    with transcript.open("r", encoding="utf-8") as file_obj:
        assert json.load(file_obj) == {
            "attempt": 3,
            "finish": {"_finish": True},
        }
    assert not (tmp_path / ".transcript.json.tmp").exists()
