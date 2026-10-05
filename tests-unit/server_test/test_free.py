import asyncio
from unittest.mock import Mock

import pytest
import pytest_asyncio
from aiohttp import web

from comfy.cli_args import args
import execution
from server import PromptServer


@pytest.fixture
def prompt_queue():
    return execution.PromptQueue(Mock())


@pytest_asyncio.fixture
async def free_client(aiohttp_client, monkeypatch, tmp_path):
    monkeypatch.setattr(args, "front_end_root", str(tmp_path))
    assets = Mock(enabled=False)
    server = PromptServer(asyncio.get_running_loop(), assets)
    app = web.Application()
    app.add_routes([route for route in server.routes if route.path == "/free"])
    return await aiohttp_client(app), server.prompt_queue


async def flags_queued(queue):
    async def wait():
        while not queue.get_flags(reset=False):
            await asyncio.sleep(0)
    await asyncio.wait_for(wait(), 2)


def test_pending_flag_does_not_wait_for_another_notification(prompt_queue):
    prompt_queue.set_flag("free_memory", True)
    prompt_queue.not_empty.wait = Mock(side_effect=AssertionError("missed pending flags"))
    assert prompt_queue.get(timeout=30) is None


@pytest.mark.asyncio
async def test_wait_request_does_not_acknowledge_pending_cleanup(free_client):
    client, queue = free_client
    request = asyncio.create_task(client.post("/free", json={"unload_models": True, "free_memory": True, "wait": True}))
    try:
        await flags_queued(queue)
        assert queue.get_flags(reset=False) == {"unload_models": True, "free_memory": True}
        assert not request.done()
        flags, completions = queue.get_flags_with_completion()
        assert len(completions) == 1
        completions[0].set_result(None)
        assert (await request).status == 200
    finally:
        request.cancel()
        await asyncio.gather(request, return_exceptions=True)


@pytest.mark.asyncio
async def test_non_waiting_request_keeps_existing_behavior(free_client):
    client, queue = free_client
    response = await client.post("/free", json={"unload_models": True, "free_memory": True})
    assert response.status == 200
    assert queue.get_flags(reset=False) == {"unload_models": True, "free_memory": True}


@pytest.mark.asyncio
async def test_wait_without_cleanup_is_immediate(free_client):
    client, queue = free_client
    response = await client.post("/free", json={"wait": True})
    assert response.status == 200
    assert queue.get_flags(reset=False) == {}
    assert queue.flag_completions == []


@pytest.mark.asyncio
async def test_timeout_does_not_cancel_other_requests_or_cleanup(free_client):
    client, queue = free_client
    timed_out = await client.post("/free", json={"free_memory": True, "wait": True, "timeout": 0.01})
    assert timed_out.status == 504
    other = asyncio.create_task(client.post("/free", json={"free_memory": True, "wait": True}))
    try:
        async def both_queued():
            while len(queue.flag_completions) != 2:
                await asyncio.sleep(0)
        await asyncio.wait_for(both_queued(), 2)
        flags, completions = queue.get_flags_with_completion()
        assert flags == {"free_memory": True}
        assert completions[0].cancelled()
        for completion in completions:
            if completion.set_running_or_notify_cancel():
                completion.set_result(None)
        assert (await other).status == 200
    finally:
        other.cancel()
        await asyncio.gather(other, return_exceptions=True)


@pytest.mark.asyncio
async def test_cleanup_failure_returns_error(free_client):
    client, queue = free_client
    request = asyncio.create_task(client.post("/free", json={"free_memory": True, "wait": True}))
    await flags_queued(queue)
    flags, completions = queue.get_flags_with_completion()
    completions[0].set_exception(RuntimeError("cleanup failed"))
    assert (await request).status == 500


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", [0, -1, 301, float("inf"), float("nan"), True, "30", None])
async def test_invalid_timeout_does_not_schedule_cleanup(free_client, timeout):
    client, queue = free_client
    response = await client.post("/free", json={"free_memory": True, "wait": True, "timeout": timeout})
    assert response.status == 400
    assert queue.get_flags(reset=False) == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("wait", [0, 1, "true", None])
async def test_invalid_wait_does_not_schedule_cleanup(free_client, wait):
    client, queue = free_client
    response = await client.post("/free", json={"free_memory": True, "wait": wait})
    assert response.status == 400
    assert queue.get_flags(reset=False) == {}


def test_empty_queue_times_out_without_cleanup(prompt_queue):
    assert prompt_queue.get(timeout=0) is None


def test_request_arriving_during_cleanup_waits_for_next_batch(prompt_queue):
    first = prompt_queue.set_flags({"unload_models": True, "free_memory": True})
    flags, first_batch = prompt_queue.get_flags_with_completion()
    second = prompt_queue.set_flags({"free_memory": True})
    for completion in first_batch:
        completion.set_result(None)
    assert first.done()
    assert not second.done()
    flags, second_batch = prompt_queue.get_flags_with_completion()
    assert flags == {"free_memory": True}
    assert second_batch == [second]


def test_pending_cleanup_takes_priority_over_next_prompt(prompt_queue):
    prompt_queue.put((0, "prompt", {}, {}, [], {}))
    prompt_queue.set_flags({"free_memory": True})
    assert prompt_queue.get(timeout=0) is None
    prompt_queue.get_flags_with_completion()
    assert prompt_queue.get(timeout=0)[0][1] == "prompt"
