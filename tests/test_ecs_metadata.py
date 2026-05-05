"""Unit tests for the ECS Task Metadata V4 emulator (services/ecs_metadata.py).

These exercise the registry + handler directly, without a running Docker
container or boto client, since the module is reachable only from inside an
ECS task in production.
"""

import asyncio
import json

import pytest

from ministack.services import ecs_metadata


def _call(method, path):
    status, _headers, body = asyncio.run(
        ecs_metadata.handle_request(method, path, {}, b"", {})
    )
    payload = json.loads(body) if body else None
    return status, payload


@pytest.fixture(autouse=True)
def _reset_registry():
    ecs_metadata.reset()
    yield
    ecs_metadata.reset()


def _register(token, task_arn, container_name, **container_overrides):
    container = {
        "DockerId": "",
        "Name": container_name,
        "Image": "alpine:latest",
        "Labels": {"com.amazonaws.ecs.task-arn": task_arn},
        "DesiredStatus": "RUNNING",
        "KnownStatus": "RUNNING",
        "Type": "NORMAL",
    }
    container.update(container_overrides)
    ecs_metadata.register_container(
        token,
        task_arn,
        task_payload={
            "Cluster": "arn:aws:ecs:us-east-1:000000000000:cluster/c",
            "TaskARN": task_arn,
            "Family": "fam",
            "Revision": "1",
            "DesiredStatus": "RUNNING",
            "KnownStatus": "RUNNING",
            "AvailabilityZone": "us-east-1a",
            "LaunchType": "FARGATE",
        },
        container_payload=container,
    )


def test_unknown_path_returns_404():
    status, body = _call("GET", "/v4/")
    assert status == 404
    assert body["message"] == "not found"


def test_unknown_token_returns_404():
    status, body = _call("GET", "/v4/abcdefgh/task")
    assert status == 404
    assert body["message"] == "unknown token"


def test_root_returns_container_payload():
    _register("tok-aaaaaaaa", "arn:task/1", "web", DockerId="d1")
    status, body = _call("GET", "/v4/tok-aaaaaaaa")
    assert status == 200
    assert body["Name"] == "web"
    assert body["DockerId"] == "d1"


def test_task_endpoint_lists_all_sibling_containers():
    _register("tok-aaaaaaaa", "arn:task/1", "web")
    _register("tok-bbbbbbbb", "arn:task/1", "sidecar")
    _register("tok-cccccccc", "arn:task/2", "other")

    status, body = _call("GET", "/v4/tok-aaaaaaaa/task")
    assert status == 200
    names = {c["Name"] for c in body["Containers"]}
    assert names == {"web", "sidecar"}

    # Task 2 sees only its own container.
    status, body = _call("GET", "/v4/tok-cccccccc/task")
    assert status == 200
    assert [c["Name"] for c in body["Containers"]] == ["other"]


def test_set_docker_id_mutates_container_payload():
    _register("tok-aaaaaaaa", "arn:task/1", "web")
    ecs_metadata.set_docker_id("tok-aaaaaaaa", "deadbeef")
    _, body = _call("GET", "/v4/tok-aaaaaaaa")
    assert body["DockerId"] == "deadbeef"


def test_set_docker_id_unknown_token_is_noop():
    # Should not raise; just does nothing.
    ecs_metadata.set_docker_id("tok-zzzzzzzz", "x")


def test_stats_endpoints_return_empty_object():
    _register("tok-aaaaaaaa", "arn:task/1", "web")
    for path in ("/v4/tok-aaaaaaaa/stats", "/v4/tok-aaaaaaaa/task/stats"):
        status, body = _call("GET", path)
        assert status == 200
        assert body == {}


def test_unregister_removes_only_that_container():
    _register("tok-aaaaaaaa", "arn:task/1", "web")
    _register("tok-bbbbbbbb", "arn:task/1", "sidecar")

    ecs_metadata.unregister_token("tok-aaaaaaaa")

    status, _ = _call("GET", "/v4/tok-aaaaaaaa")
    assert status == 404

    status, body = _call("GET", "/v4/tok-bbbbbbbb/task")
    assert status == 200
    assert [c["Name"] for c in body["Containers"]] == ["sidecar"]


def test_unregister_last_token_drops_task_entry():
    _register("tok-aaaaaaaa", "arn:task/1", "web")
    ecs_metadata.unregister_token("tok-aaaaaaaa")
    # Internal: the task arn entry should be cleaned up so the registry
    # doesn't grow unbounded across run/stop cycles.
    assert "arn:task/1" not in ecs_metadata._TASKS


def test_unregister_unknown_token_is_noop():
    ecs_metadata.unregister_token("tok-zzzzzzzz")  # must not raise


def test_reset_clears_all_state():
    _register("tok-aaaaaaaa", "arn:task/1", "web")
    _register("tok-bbbbbbbb", "arn:task/2", "other")
    ecs_metadata.reset()
    assert ecs_metadata._TASKS == {}
    assert ecs_metadata._TOKEN_TO_TASK == {}
    assert ecs_metadata._TOKEN_TO_CONTAINER == {}


def test_path_regex_rejects_short_token():
    # Token must be at least 8 chars per the regex.
    status, _ = _call("GET", "/v4/short/task")
    assert status == 404


def test_trailing_slash_on_root_is_tolerated():
    _register("tok-aaaaaaaa", "arn:task/1", "web")
    status, body = _call("GET", "/v4/tok-aaaaaaaa/")
    assert status == 200
    assert body["Name"] == "web"
