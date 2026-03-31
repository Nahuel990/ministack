"""
Unit tests for ECS container network detection logic.

Verifies that _run_task attaches spawned containers to the same Docker
network as the Ministack host container.
"""
import os
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _reset_ecs_state():
    """Reset ECS module-level state between tests."""
    from ministack.services import ecs
    ecs._clusters.clear()
    ecs._task_defs.clear()
    ecs._tasks.clear()
    ecs._services.clear()
    ecs._tags.clear()
    yield
    ecs._clusters.clear()
    ecs._task_defs.clear()
    ecs._tasks.clear()
    ecs._services.clear()
    ecs._tags.clear()


def _setup_cluster_and_taskdef(ecs_mod):
    """Register a cluster + task definition so _run_task has something to launch."""
    ecs_mod._clusters["default"] = {
        "clusterArn": "arn:aws:ecs:us-east-1:000000000000:cluster/default",
        "clusterName": "default",
        "status": "ACTIVE",
        "registeredContainerInstancesCount": 0,
        "runningTasksCount": 0,
        "pendingTasksCount": 0,
        "activeServicesCount": 0,
    }
    ecs_mod._task_defs["myapp"] = {
        "taskDefinitionArn": "arn:aws:ecs:us-east-1:000000000000:task-definition/myapp:1",
        "family": "myapp",
        "revision": 1,
        "status": "ACTIVE",
        "containerDefinitions": [
            {
                "name": "web",
                "image": "alpine:latest",
                "essential": True,
                "portMappings": [],
                "environment": [],
            }
        ],
    }


def _make_mock_docker(network_name=None, hostname_found=True):
    """Build a mock docker client.

    If network_name is set, the 'self' container reports that network.
    If hostname_found is False, containers.get() raises NotFound.
    """
    mock_client = MagicMock()

    if hostname_found and network_name:
        self_container = MagicMock()
        self_container.attrs = {
            "NetworkSettings": {
                "Networks": {network_name: {"NetworkID": "abc123"}}
            }
        }
        mock_client.containers.get.return_value = self_container
    else:
        mock_client.containers.get.side_effect = Exception("not found")

    # The container returned by containers.run()
    run_container = MagicMock()
    run_container.id = "deadbeef12345678"
    mock_client.containers.run.return_value = run_container

    return mock_client


def test_run_task_passes_detected_network():
    """When Ministack is on 'my-compose-net', ECS containers join that network."""
    from ministack.services import ecs

    _setup_cluster_and_taskdef(ecs)

    mock_docker = _make_mock_docker(network_name="my-compose-net")

    with patch.object(ecs, "_get_docker", return_value=mock_docker), \
         patch.dict(os.environ, {"HOSTNAME": "abc123host"}):
        resp = ecs._run_task({"taskDefinition": "myapp", "cluster": "default"})

    # Verify containers.get was called with the HOSTNAME
    mock_docker.containers.get.assert_called_with("abc123host")

    # Verify containers.run was called with network="my-compose-net"
    call_kwargs = mock_docker.containers.run.call_args
    assert call_kwargs.kwargs.get("network") == "my-compose-net", \
        f"Expected network='my-compose-net', got {call_kwargs.kwargs.get('network')}"


def test_run_task_falls_back_to_none_when_detection_fails():
    """When HOSTNAME lookup fails, network should be None (default bridge)."""
    from ministack.services import ecs

    _setup_cluster_and_taskdef(ecs)

    mock_docker = _make_mock_docker(hostname_found=False)

    with patch.object(ecs, "_get_docker", return_value=mock_docker), \
         patch.dict(os.environ, {"HOSTNAME": "nonexistent"}):
        resp = ecs._run_task({"taskDefinition": "myapp", "cluster": "default"})

    call_kwargs = mock_docker.containers.run.call_args
    assert call_kwargs.kwargs.get("network") is None, \
        f"Expected network=None, got {call_kwargs.kwargs.get('network')}"


def test_run_task_no_hostname_env():
    """When HOSTNAME is not set, detection should fail gracefully."""
    from ministack.services import ecs

    _setup_cluster_and_taskdef(ecs)

    mock_docker = _make_mock_docker(hostname_found=False)

    env = os.environ.copy()
    env.pop("HOSTNAME", None)

    with patch.object(ecs, "_get_docker", return_value=mock_docker), \
         patch.dict(os.environ, env, clear=True):
        resp = ecs._run_task({"taskDefinition": "myapp", "cluster": "default"})

    call_kwargs = mock_docker.containers.run.call_args
    assert call_kwargs.kwargs.get("network") is None


def test_run_task_multiple_networks_picks_first():
    """When the host container is on multiple networks, pick the first one."""
    from ministack.services import ecs

    _setup_cluster_and_taskdef(ecs)

    mock_client = MagicMock()
    self_container = MagicMock()
    # Python 3.7+ dicts are insertion-ordered
    self_container.attrs = {
        "NetworkSettings": {
            "Networks": {
                "primary-net": {"NetworkID": "aaa"},
                "secondary-net": {"NetworkID": "bbb"},
            }
        }
    }
    mock_client.containers.get.return_value = self_container

    run_container = MagicMock()
    run_container.id = "deadbeef12345678"
    mock_client.containers.run.return_value = run_container

    with patch.object(ecs, "_get_docker", return_value=mock_client), \
         patch.dict(os.environ, {"HOSTNAME": "myhost"}):
        resp = ecs._run_task({"taskDefinition": "myapp", "cluster": "default"})

    call_kwargs = mock_client.containers.run.call_args
    assert call_kwargs.kwargs.get("network") == "primary-net"
