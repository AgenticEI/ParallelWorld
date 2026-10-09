import threading
import time
from types import SimpleNamespace

from parallelworld_manipulation.planner import codex as module


def test_finish_timeout_returns_existing_result_without_total_timeout(tmp_path, monkeypatch):
    # The session runner is replaced below; constructor setup must not depend
    # on API credentials from the host running this timeout regression.
    monkeypatch.setenv("PARALLELWORLD_CODEX_AUTH", "chatgpt")
    interrupted = threading.Event()
    monkeypatch.setattr(module, "HttpMcpServer", lambda toolkit: SimpleNamespace(
        start=lambda: "http://127.0.0.1:1", stop=lambda: None,
    ))
    monkeypatch.setattr(module, "_interrupt", lambda state: interrupted.set())

    def stalled(self, prompt, output, raw, last, recorder, state, url, queue):
        output.write_text("existing transcript\n")
        raw.write_text("")
        recorder.finish_result = {"_finish": True, "status": "success"}
        recorder.finish_completed_at = time.monotonic() - 121
        interrupted.wait(5)

    monkeypatch.setattr(module.CodexPlanner, "_run_session", stalled)
    planner = module.CodexPlanner(
        output_dir=str(tmp_path), dashboard_events=SimpleNamespace(emit=lambda e: None),
        timeout_s=None, output_path=tmp_path / "run.txt",
    )
    result = planner.solve(system_prompt="", user_message="test", toolkit=None, max_turns=10)
    assert interrupted.is_set()
    assert result.finish_result["status"] == "success"
    assert result.error is None
    assert '"type": "finish_timeout"' in (tmp_path / "run.txt.stream.jsonl").read_text()
