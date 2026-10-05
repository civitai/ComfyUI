import pytest
import torch
from unittest.mock import Mock

from comfy.cli_args import args

# Must precede the import: comfy.model_management picks its device at import time, and a CUDA
# build with no driver raises there.
_original_cpu = args.cpu
if not torch.cuda.is_available():
    args.cpu = True
try:
    import main
finally:
    args.cpu = _original_cpu

class LoopEscape(Exception):
    pass


class Queue:
    def __init__(
        self,
        completion_error: RuntimeError | None = None,
        queued_after_first: int = 1,
        on_second_get=None,
    ) -> None:
        self.completion_error = completion_error
        # Prompts still waiting once the first is taken; tests opt in to an idle queue.
        self.queued = queued_after_first
        self.running = 0
        self.on_second_get = on_second_get
        self.get_calls = 0

    def get(self, timeout=None):
        self.get_calls += 1
        if self.get_calls > 1:
            if self.on_second_get is not None:
                self.on_second_get()
            raise LoopEscape("prompt worker requested a second item")
        self.running = 1
        return (0, "prompt-id", {}, {}, [], {}), 1

    def get_tasks_remaining(self) -> int:
        # Like PromptQueue: queued plus running, and the running prompt only leaves in task_done.
        return self.queued + self.running

    def task_done(self, *args, **kwargs) -> None:
        if self.completion_error is not None:
            raise self.completion_error
        self.running = 0

    def get_flags_with_completion(self):
        return {}, []


class Server:
    last_prompt_id = None
    client_id = None


class AssetManager:
    def __init__(self, resume_error: RuntimeError | None = None) -> None:
        self.paused = False
        self.resume_error = resume_error

    def pause_background_scan(self) -> None:
        self.paused = True

    def resume_background_scan(self) -> None:
        self.paused = False
        if self.resume_error is not None:
            raise self.resume_error


class Executor:
    def __init__(self, *args, **kwargs) -> None:
        self.history_result = {}
        self.success = True
        self.status_messages = []

    def execute(self, *args, **kwargs) -> None:
        return None


class ExecuteFailureExecutor(Executor):
    def execute(self, *args, **kwargs) -> None:
        raise RuntimeError("forced execute failure")


def test_prompt_worker_resumes_background_scan_when_execute_raises(monkeypatch) -> None:
    monkeypatch.setattr(main.execution, "PromptExecutor", ExecuteFailureExecutor)
    asset_manager = AssetManager()

    with pytest.raises(RuntimeError, match="^forced execute failure$"):
        main.prompt_worker(Queue(), Server(), asset_manager)

    assert asset_manager.paused is False


def test_prompt_worker_resumes_background_scan_when_completion_raises(monkeypatch) -> None:
    monkeypatch.setattr(main.execution, "PromptExecutor", Executor)
    asset_manager = AssetManager()

    with pytest.raises(RuntimeError, match="^forced completion failure$"):
        main.prompt_worker(
            Queue(completion_error=RuntimeError("forced completion failure")),
            Server(),
            asset_manager,
        )

    assert asset_manager.paused is False


def test_prompt_worker_preserves_execute_error_when_resume_raises(monkeypatch) -> None:
    monkeypatch.setattr(main.execution, "PromptExecutor", ExecuteFailureExecutor)
    asset_manager = AssetManager(resume_error=RuntimeError("forced resume failure"))

    with pytest.raises(RuntimeError, match="^forced execute failure$"):
        main.prompt_worker(Queue(), Server(), asset_manager)

    assert asset_manager.paused is False


def test_prompt_worker_resumes_scan_when_later_iteration_raises_before_gc(monkeypatch) -> None:
    monkeypatch.setattr(main.execution, "PromptExecutor", Executor)
    clock = iter((1.0, 2.0, 2.0))
    monkeypatch.setattr(main.time, "perf_counter", lambda: next(clock))
    asset_manager = AssetManager()
    queue = Queue()

    with pytest.raises(LoopEscape, match="^prompt worker requested a second item$"):
        main.prompt_worker(queue, Server(), asset_manager)

    assert queue.get_calls == 2
    assert asset_manager.paused is False


def _paused_when_the_worker_asks_for_the_next_prompt(monkeypatch, queued_after_first: int) -> list[bool]:
    monkeypatch.setattr(main.execution, "PromptExecutor", Executor)
    # The clock never reaches the 10 s GC interval, so only the idle check can resume the scan.
    monkeypatch.setattr(main.time, "perf_counter", lambda: 1.0)
    asset_manager = AssetManager()
    seen: list[bool] = []
    queue = Queue(queued_after_first=queued_after_first, on_second_get=lambda: seen.append(asset_manager.paused))

    with pytest.raises(LoopEscape, match="^prompt worker requested a second item$"):
        main.prompt_worker(queue, Server(), asset_manager)

    return seen


def test_prompt_worker_resumes_scan_once_the_queue_is_empty(monkeypatch) -> None:
    assert _paused_when_the_worker_asks_for_the_next_prompt(monkeypatch, queued_after_first=0) == [False]


def test_prompt_worker_keeps_scan_paused_while_prompts_are_queued(monkeypatch) -> None:
    assert _paused_when_the_worker_asks_for_the_next_prompt(monkeypatch, queued_after_first=1) == [True]


@pytest.mark.parametrize("failed_stage", [None, "unload", "reset", "gc", "empty_cache"])
def test_free_completion_follows_all_cleanup(monkeypatch, failed_stage):
    queue = main.execution.PromptQueue(Mock())
    completion = queue.set_flags({"unload_models": True, "free_memory": True})
    events = []

    def stage(name):
        def run(*args, **kwargs):
            assert not completion.done()
            events.append(name)
            if name == failed_stage:
                raise RuntimeError(name)
        return run

    class FreeExecutor(Executor):
        reset = stage("reset")

    original_get = queue.get
    calls = 0

    def get(timeout=None):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise LoopEscape()
        return original_get(timeout=0)

    monkeypatch.setattr(queue, "get", get)
    monkeypatch.setattr(main.execution, "PromptExecutor", FreeExecutor)
    monkeypatch.setattr(main.comfy.model_management, "unload_all_models", stage("unload"))
    monkeypatch.setattr(main.gc, "collect", stage("gc"))
    monkeypatch.setattr(main.comfy.model_management, "soft_empty_cache", stage("empty_cache"))
    monkeypatch.setattr(main.hook_breaker_ac10a0, "restore_functions", stage("restore"))
    # Explicit cleanup must not wait for the periodic ten-second GC interval.
    monkeypatch.setattr(main.time, "perf_counter", lambda: 1.0)

    if failed_stage:
        with pytest.raises(RuntimeError, match=f"^{failed_stage}$"):
            main.prompt_worker(queue, Server(), Mock())
        with pytest.raises(RuntimeError, match=f"^{failed_stage}$"):
            completion.result(timeout=0)
    else:
        with pytest.raises(LoopEscape):
            main.prompt_worker(queue, Server(), Mock())
        assert completion.result(timeout=0) is None
        assert events == ["unload", "reset", "gc", "empty_cache", "restore"]


def test_free_requested_during_prompt_completes_after_execution(monkeypatch):
    queue = main.execution.PromptQueue(Mock())
    queue.put((0, "prompt-id", {}, {}, [], {}))
    completion = None
    events = []

    class BusyExecutor(Executor):
        def execute(self, *args, **kwargs):
            nonlocal completion
            events.append("execute")
            completion = queue.set_flags({"free_memory": True})
            assert not completion.done()
            events.append("executed")

        def reset(self):
            events.append("reset")

    original_get = queue.get
    calls = 0

    def get(timeout=None):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise LoopEscape()
        return original_get(timeout=0)

    monkeypatch.setattr(queue, "get", get)
    monkeypatch.setattr(main.execution, "PromptExecutor", BusyExecutor)
    monkeypatch.setattr(main.comfy.model_management, "unload_all_models", lambda: events.append("unload"))
    monkeypatch.setattr(main.gc, "collect", lambda: events.append("gc"))
    monkeypatch.setattr(main.comfy.model_management, "soft_empty_cache", lambda: events.append("empty_cache"))
    monkeypatch.setattr(main.hook_breaker_ac10a0, "restore_functions", lambda: None)

    with pytest.raises(LoopEscape):
        main.prompt_worker(queue, Server(), Mock())
    assert events == ["execute", "executed", "unload", "reset", "gc", "empty_cache"]
    assert completion.result(timeout=0) is None
