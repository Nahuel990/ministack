"""
OpenSearch Service Emulator — AWS-compatible.
Supports: CreateDomain, DescribeDomain, DescribeDomains, DeleteDomain,
          ListDomainNames, UpdateDomainConfig, DescribeDomainConfig,
          DescribeDomainChangeProgress, ListVersions, GetCompatibleVersions,
          AddTags, ListTags, RemoveTags.
REST-JSON control plane on /2021-01-01/* (boto3 ``opensearch`` client; SigV4
credential scope ``es``). Data plane is in-memory by default (stub endpoint
<domain>.ministack.local:9200); set OPENSEARCH_DATAPLANE=1 to spawn one
``opensearchproject/opensearch`` container per domain (labeled
com.docker.compose.project=ministack, bound at
localhost:<OPENSEARCH_BASE_PORT+N>, torn down on DeleteDomain).
When MiniStack runs inside Docker, the dataplane joins MiniStack's own
network (auto-detected from ``HOSTNAME``) or ``ministack-opensearch-net`` as
a fallback, and ``DescribeDomain`` reports ``<container_name>:9200`` so
sibling containers (e.g. Lambda) resolve the cluster by DNS.
Additionally set OPENSEARCH_DASHBOARDS=1 to spawn a per-domain
``opensearchproject/opensearch-dashboards`` sidecar that points at the
matching cluster over the same network,
published on localhost:<OPENSEARCH_DASHBOARDS_BASE_PORT+N> and surfaced
through ``DescribeDomain.DomainStatus.DashboardEndpoint``.
Env: OPENSEARCH_DATAPLANE, OPENSEARCH_BASE_PORT (14571), OPENSEARCH_IMAGE,
OPENSEARCH_DASHBOARDS, OPENSEARCH_DASHBOARDS_BASE_PORT (15601),
OPENSEARCH_DASHBOARDS_IMAGE, MINISTACK_OPENSEARCH_ENDPOINT (override),
DOCKER_NETWORK / LAMBDA_DOCKER_NETWORK (optional explicit network).
"""

import copy
import json
import logging
import os
import time
from urllib.parse import unquote

from ministack.core.docker_network import get_ministack_network
from ministack.core.responses import (
    AccountScopedDict,
    error_response_json,
    get_account_id,
    get_region,
    json_response,
    new_uuid,
    now_iso,
)


def _now_epoch() -> float:
    """Return the current time as unix epoch seconds (float).

    The OpenSearch REST-JSON service models ``OptionStatus.CreationDate`` /
    ``UpdateDate`` and ``ChangeProgressStatus.StartTime`` with
    ``@timestampFormat("unixTimestamp")`` — the SDK (and Terraform's AWS
    provider, which uses aws-sdk-go-v2) rejects ISO-8601 strings with
    ``expected ... to be a JSON Number``. Returning epoch seconds matches
    the real service and unblocks ``terraform apply``.
    """
    return time.time()

logger = logging.getLogger("opensearch")

_DEFAULT_ENGINE_VERSION = "OpenSearch_2.15"
_SUPPORTED_VERSIONS = [
    "OpenSearch_2.15", "OpenSearch_2.13", "OpenSearch_2.11", "OpenSearch_2.9",
    "OpenSearch_2.7", "OpenSearch_2.5", "OpenSearch_2.3",
    "OpenSearch_1.3", "OpenSearch_1.2", "OpenSearch_1.1", "OpenSearch_1.0",
    "Elasticsearch_7.10", "Elasticsearch_7.9",
]

_domains = AccountScopedDict()
_tags = AccountScopedDict()

_OPENSEARCH_BASE_PORT = int(os.environ.get("OPENSEARCH_BASE_PORT", "14571"))
_OPENSEARCH_DASHBOARDS_BASE_PORT = int(os.environ.get("OPENSEARCH_DASHBOARDS_BASE_PORT", "15601"))
_port_counter = [_OPENSEARCH_BASE_PORT]
_dashboards_port_counter = [_OPENSEARCH_DASHBOARDS_BASE_PORT]
_docker = None
_NETWORK_NAME = "ministack-opensearch-net"


def _dataplane_enabled() -> bool:
    return os.environ.get("OPENSEARCH_DATAPLANE", "0").lower() in ("1", "true", "yes", "on")


def _dashboards_enabled() -> bool:
    """Dashboards require the real dataplane; silently off otherwise."""
    return _dataplane_enabled() and os.environ.get(
        "OPENSEARCH_DASHBOARDS", "0"
    ).lower() in ("1", "true", "yes", "on")


def _get_docker():
    """Lazily connect to the Docker daemon via the mounted socket."""
    global _docker
    if _docker is None:
        try:
            import docker
            _docker = docker.from_env()
        except Exception as e:  # pragma: no cover - best-effort
            logger.warning("OpenSearch: Docker unavailable (%s); falling back to stub endpoint", e)
    return _docker


def _ensure_ministack_network(client) -> str | None:
    """Create (or look up) the shared bridge network that lets the
    Dashboards container resolve its OpenSearch peer by container name.
    Returns the network name on success, ``None`` if Docker is unhappy."""
    try:
        client.networks.get(_NETWORK_NAME)
        return _NETWORK_NAME
    except Exception:
        pass
    try:
        client.networks.create(
            _NETWORK_NAME,
            driver="bridge",
            labels={"com.docker.compose.project": "ministack"},
        )
        return _NETWORK_NAME
    except Exception as e:  # pragma: no cover - best-effort
        logger.warning("OpenSearch: failed to create network %s: %s", _NETWORK_NAME, e)
        return None


def _stub_endpoint(domain_name: str) -> str:
    """Deterministic in-memory endpoint used when no container is spawned."""
    return f"{domain_name}.ministack.local:9200"


def _default_endpoint_for(domain_name: str) -> str:
    """Return a per-domain endpoint when no Docker container is available.

    Respects an explicit ``MINISTACK_OPENSEARCH_ENDPOINT`` override (handy
    if the user is running a custom OpenSearch elsewhere), otherwise uses
    the deterministic ``<domain>.ministack.local:9200`` stub.
    """
    explicit = os.environ.get("MINISTACK_OPENSEARCH_ENDPOINT")
    if explicit:
        return explicit
    return _stub_endpoint(domain_name)


def _spawn_opensearch_container(domain_name: str) -> tuple[str | None, str]:
    """Start a real OpenSearch container for this domain.

    Returns ``(container_id, endpoint)``. When Docker is unavailable or
    ``OPENSEARCH_DATAPLANE`` is off we return ``(None, stub_endpoint)`` so
    the management plane keeps working without a real cluster.
    """
    if not _dataplane_enabled():
        return None, _default_endpoint_for(domain_name)

    client = _get_docker()
    if client is None:
        return None, _default_endpoint_for(domain_name)

    host_port = _port_counter[0]
    _port_counter[0] += 1
    image = os.environ.get("OPENSEARCH_IMAGE", "opensearchproject/opensearch:2.15.0")
    container_name = f"ministack-opensearch-{domain_name}"
    network = get_ministack_network(client) or _ensure_ministack_network(client)
    try:
        container = client.containers.run(
            image,
            detach=True,
            ports={"9200/tcp": host_port},
            name=container_name,
            hostname=container_name,
            network=network,
            labels={
                "ministack": "opensearch",
                "ministack.domain": domain_name,
                "com.docker.compose.project": "ministack",
                "com.docker.compose.service": f"opensearch-{domain_name}",
            },
            environment={
                "discovery.type": "single-node",
                "DISABLE_SECURITY_PLUGIN": "true",
                "DISABLE_INSTALL_DEMO_CONFIG": "true",
                "OPENSEARCH_JAVA_OPTS": "-Xms512m -Xmx512m",
            },
        )
        logger.info(
            "OpenSearch: started container %s for domain %s on port %s",
            container.short_id, domain_name, host_port,
        )
        if network:
            return container.id, f"{container_name}:9200"
        return container.id, f"localhost:{host_port}"
    except Exception as e:
        logger.warning(
            "OpenSearch: failed to start container for %s: %s (falling back to stub)",
            domain_name, e,
        )
        return None, _default_endpoint_for(domain_name)


def _spawn_opensearch_dashboards(domain_name: str) -> tuple[str | None, str | None]:
    """Start an OpenSearch Dashboards sidecar attached to the same Docker
    network as the dataplane cluster (MiniStack-detected network or
    ``ministack-opensearch-net``).

    Returns ``(container_id, dashboard_endpoint)`` or ``(None, None)`` when
    OPENSEARCH_DASHBOARDS is off, Docker is unavailable, or spawning fails.
    Dashboards needs ~30-60 s to become ready; we publish the endpoint
    immediately and the caller (tests / human) polls ``/api/status``.
    """
    if not _dashboards_enabled():
        return None, None
    client = _get_docker()
    if client is None:
        return None, None
    network = get_ministack_network(client) or _ensure_ministack_network(client)
    if not network:
        return None, None

    host_port = _dashboards_port_counter[0]
    _dashboards_port_counter[0] += 1
    image = os.environ.get(
        "OPENSEARCH_DASHBOARDS_IMAGE",
        "opensearchproject/opensearch-dashboards:2.15.0",
    )
    cluster_name = f"ministack-opensearch-{domain_name}"
    dash_name = f"ministack-opensearch-dashboards-{domain_name}"
    try:
        container = client.containers.run(
            image,
            detach=True,
            ports={"5601/tcp": host_port},
            name=dash_name,
            hostname=dash_name,
            network=network,
            labels={
                "ministack": "opensearch-dashboards",
                "ministack.domain": domain_name,
                "com.docker.compose.project": "ministack",
                "com.docker.compose.service": f"opensearch-dashboards-{domain_name}",
            },
            environment={
                # opensearchproject/opensearch-dashboards picks these up via
                # its docker-entrypoint; we point it at the cluster by its
                # network-resolvable container name.
                "OPENSEARCH_HOSTS": f'["http://{cluster_name}:9200"]',
                "DISABLE_SECURITY_DASHBOARDS_PLUGIN": "true",
            },
        )
        logger.info(
            "OpenSearch: started Dashboards container %s for domain %s on port %s",
            container.short_id, domain_name, host_port,
        )
        return container.id, f"localhost:{host_port}"
    except Exception as e:
        logger.warning(
            "OpenSearch: failed to start Dashboards container for %s: %s",
            domain_name, e,
        )
        return None, None


def _stop_opensearch_container(domain: dict) -> None:
    for attr, label in (
        ("_docker_dashboards_container_id", "Dashboards"),
        ("_docker_container_id", "cluster"),
    ):
        container_id = domain.get(attr)
        if not container_id:
            continue
        client = _get_docker()
        if client is None:
            return
        try:
            c = client.containers.get(container_id)
            c.stop(timeout=5)
            c.remove(v=True)
            logger.info(
                "OpenSearch: removed %s container for domain %s",
                label, domain.get("DomainName"),
            )
        except Exception as e:  # pragma: no cover - best-effort cleanup
            logger.warning(
                "OpenSearch: failed to remove %s container %s for %s: %s",
                label, container_id, domain.get("DomainName"), e,
            )


def _endpoint_for(domain: dict) -> str:
    ep = domain.get("Endpoint")
    if ep:
        return ep
    return _default_endpoint_for(domain.get("DomainName", ""))


def _arn(domain_name: str) -> str:
    return f"arn:aws:es:{get_region()}:{get_account_id()}:domain/{domain_name}"


def _domain_id(domain_name: str) -> str:
    return f"{get_account_id()}/{domain_name}"


def _default_cluster_config(user_cfg: dict | None) -> dict:
    cfg = {
        "InstanceType": "r6g.large.search",
        "InstanceCount": 1,
        "DedicatedMasterEnabled": False,
        "ZoneAwarenessEnabled": False,
        "WarmEnabled": False,
        "ColdStorageOptions": {"Enabled": False},
        "MultiAZWithStandbyEnabled": False,
    }
    if user_cfg:
        cfg.update(user_cfg)
    return cfg


def _default_ebs(user_cfg: dict | None) -> dict:
    cfg = {"EBSEnabled": True, "VolumeType": "gp3", "VolumeSize": 10}
    if user_cfg:
        cfg.update(user_cfg)
    return cfg


def _default_endpoint_options(user_cfg: dict | None) -> dict:
    cfg = {
        "EnforceHTTPS": False,
        "TLSSecurityPolicy": "Policy-Min-TLS-1-2-2019-07",
        "CustomEndpointEnabled": False,
    }
    if user_cfg:
        cfg.update(user_cfg)
    return cfg


def _default_encryption_at_rest(user_cfg: dict | None) -> dict:
    cfg = {"Enabled": False}
    if user_cfg:
        cfg.update(user_cfg)
    return cfg


def _default_node_to_node_encryption(user_cfg: dict | None) -> dict:
    cfg = {"Enabled": False}
    if user_cfg:
        cfg.update(user_cfg)
    return cfg


def _domain_status(domain: dict, *, created: bool = True, deleted: bool = False) -> dict:
    """Assemble a DomainStatus payload in the exact shape boto3 decodes.

    The real service includes a handful of lifecycle booleans
    (``Created``, ``Deleted``, ``Processing``, ``UpgradeProcessing``). We
    report a stable "available" shape — ``Created=True``, everything else
    ``False`` — which is what the AWS provider and opensearch-py both
    expect for a healthy domain.

    For non-VPC domains we emit ``Endpoint`` alone and OMIT ``Endpoints``
    (the AWS provider explicitly rejects a domain that has both
    ``VpcOptions == nil`` and a present-but-empty ``Endpoints`` with
    ``expected to have null Endpoints value``). VPC domains get
    ``Endpoints`` and no top-level ``Endpoint``.
    """
    endpoint = _endpoint_for(domain)
    status = {
        "DomainId": domain["DomainId"],
        "DomainName": domain["DomainName"],
        "ARN": domain["ARN"],
        "Created": created,
        "Deleted": deleted,
        "Processing": False,
        "UpgradeProcessing": False,
        "EngineVersion": domain.get("EngineVersion", _DEFAULT_ENGINE_VERSION),
        "ClusterConfig": domain.get("ClusterConfig", _default_cluster_config(None)),
        "EBSOptions": domain.get("EBSOptions", _default_ebs(None)),
        "AccessPolicies": domain.get("AccessPolicies", ""),
        "IPAddressType": domain.get("IPAddressType", "ipv4"),
        "SnapshotOptions": domain.get("SnapshotOptions", {"AutomatedSnapshotStartHour": 0}),
        "CognitoOptions": domain.get("CognitoOptions", {"Enabled": False}),
        "EncryptionAtRestOptions": domain.get(
            "EncryptionAtRestOptions", _default_encryption_at_rest(None)
        ),
        "NodeToNodeEncryptionOptions": domain.get(
            "NodeToNodeEncryptionOptions", _default_node_to_node_encryption(None)
        ),
        "AdvancedOptions": domain.get("AdvancedOptions", {}),
        "LogPublishingOptions": domain.get("LogPublishingOptions", {}),
        "ServiceSoftwareOptions": {
            "CurrentVersion": domain.get("EngineVersion", _DEFAULT_ENGINE_VERSION),
            "NewVersion": "",
            "UpdateAvailable": False,
            "Cancellable": False,
            "UpdateStatus": "COMPLETED",
            "Description": "",
            "AutomatedUpdateDate": 0,
            "OptionalDeployment": False,
        },
        "DomainEndpointOptions": domain.get(
            "DomainEndpointOptions", _default_endpoint_options(None)
        ),
        "AdvancedSecurityOptions": domain.get(
            "AdvancedSecurityOptions",
            {"Enabled": False, "InternalUserDatabaseEnabled": False},
        ),
        "AutoTuneOptions": domain.get(
            "AutoTuneOptions", {"State": "DISABLED", "UseOffPeakWindow": False}
        ),
        "DomainProcessingStatus": "Active",
    }
    vpc = domain.get("VPCOptions")
    if vpc:
        status["VPCOptions"] = vpc
        status["Endpoints"] = {"vpc": endpoint}
    else:
        status["Endpoint"] = endpoint
    # DashboardEndpoint is only present when OPENSEARCH_DASHBOARDS=1 spawned
    # a sidecar; omitting the key on the null case matches real AWS (which
    # also omits it for domains with no enabled dashboard plugin).
    dash = domain.get("DashboardEndpoint")
    if dash:
        status["DashboardEndpoint"] = dash
    return status


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------

def _create_domain(body: dict):
    name = body.get("DomainName")
    if not name:
        return error_response_json("ValidationException", "DomainName is required", 400)
    if not isinstance(name, str) or len(name) < 3 or len(name) > 28:
        return error_response_json(
            "ValidationException",
            "DomainName must be between 3 and 28 characters",
            400,
        )
    if name in _domains:
        return error_response_json(
            "ResourceAlreadyExistsException",
            f"Domain {name} already exists",
            409,
        )

    container_id, endpoint = _spawn_opensearch_container(name)
    dashboards_container_id, dashboard_endpoint = _spawn_opensearch_dashboards(name)

    domain = {
        "DomainId": _domain_id(name),
        "DomainName": name,
        "ARN": _arn(name),
        "Endpoint": endpoint,
        "DashboardEndpoint": dashboard_endpoint,
        "_docker_container_id": container_id,
        "_docker_dashboards_container_id": dashboards_container_id,
        "EngineVersion": body.get("EngineVersion", _DEFAULT_ENGINE_VERSION),
        "ClusterConfig": _default_cluster_config(body.get("ClusterConfig")),
        "EBSOptions": _default_ebs(body.get("EBSOptions")),
        "AccessPolicies": body.get("AccessPolicies", ""),
        "IPAddressType": body.get("IPAddressType", "ipv4"),
        "SnapshotOptions": body.get("SnapshotOptions") or {"AutomatedSnapshotStartHour": 0},
        "VPCOptions": body.get("VPCOptions"),
        "CognitoOptions": body.get("CognitoOptions") or {"Enabled": False},
        "EncryptionAtRestOptions": _default_encryption_at_rest(
            body.get("EncryptionAtRestOptions")
        ),
        "NodeToNodeEncryptionOptions": _default_node_to_node_encryption(
            body.get("NodeToNodeEncryptionOptions")
        ),
        "AdvancedOptions": body.get("AdvancedOptions") or {},
        "LogPublishingOptions": body.get("LogPublishingOptions") or {},
        "DomainEndpointOptions": _default_endpoint_options(
            body.get("DomainEndpointOptions")
        ),
        "AdvancedSecurityOptions": body.get("AdvancedSecurityOptions")
        or {"Enabled": False, "InternalUserDatabaseEnabled": False},
        "AutoTuneOptions": body.get("AutoTuneOptions")
        or {"State": "DISABLED", "UseOffPeakWindow": False},
        "CreatedAt": now_iso(),
        "CreatedAtEpoch": _now_epoch(),
    }
    _domains[name] = domain

    tag_list = body.get("TagList") or []
    if tag_list:
        _tags[_arn(name)] = [
            {"Key": t.get("Key", ""), "Value": t.get("Value", "")} for t in tag_list
        ]

    logger.info("OpenSearch: created domain %s", name)
    return json_response({"DomainStatus": _domain_status(domain)})


def _describe_domain(name: str):
    domain = _domains.get(name)
    if not domain:
        return error_response_json(
            "ResourceNotFoundException", f"Domain not found: {name}", 404
        )
    return json_response({"DomainStatus": _domain_status(domain)})


def _describe_domains(body: dict):
    names = body.get("DomainNames") or []
    statuses = []
    for n in names:
        d = _domains.get(n)
        if d:
            statuses.append(_domain_status(d))
    return json_response({"DomainStatusList": statuses})


def _delete_domain(name: str):
    domain = _domains.pop(name, None)
    if not domain:
        return error_response_json(
            "ResourceNotFoundException", f"Domain not found: {name}", 404
        )
    _stop_opensearch_container(domain)
    _tags.pop(_arn(name), None)
    logger.info("OpenSearch: deleted domain %s", name)
    return json_response({"DomainStatus": _domain_status(domain, deleted=True)})


def _list_domain_names(engine_type: str | None):
    items = []
    for name, d in _domains.items():
        ev = d.get("EngineVersion", _DEFAULT_ENGINE_VERSION)
        et = "Elasticsearch" if ev.startswith("Elasticsearch_") else "OpenSearch"
        if engine_type and engine_type != et:
            continue
        items.append({"DomainName": name, "EngineType": et})
    return json_response({"DomainNames": items})


def _update_domain_config(name: str, body: dict):
    domain = _domains.get(name)
    if not domain:
        return error_response_json(
            "ResourceNotFoundException", f"Domain not found: {name}", 404
        )

    _UPDATABLE = (
        "ClusterConfig", "EBSOptions", "SnapshotOptions", "VPCOptions",
        "CognitoOptions", "AdvancedOptions", "AccessPolicies",
        "LogPublishingOptions", "DomainEndpointOptions",
        "AdvancedSecurityOptions", "NodeToNodeEncryptionOptions",
        "EncryptionAtRestOptions", "AutoTuneOptions", "IPAddressType",
    )
    for key in _UPDATABLE:
        if key in body and body[key] is not None:
            # Merge dict-valued fields so partial updates behave like real AWS.
            existing = domain.get(key)
            incoming = body[key]
            if isinstance(existing, dict) and isinstance(incoming, dict):
                merged = dict(existing)
                merged.update(incoming)
                domain[key] = merged
            else:
                domain[key] = incoming

    logger.info("OpenSearch: updated domain %s", name)
    return json_response({"DomainConfig": _domain_config_view(domain)})


def _describe_domain_config(name: str):
    domain = _domains.get(name)
    if not domain:
        return error_response_json(
            "ResourceNotFoundException", f"Domain not found: {name}", 404
        )
    return json_response({"DomainConfig": _domain_config_view(domain)})


def _domain_config_view(domain: dict) -> dict:
    """Wrap each config field in the ``{Options, Status}`` shape boto3 expects.

    ``OptionStatus.CreationDate`` and ``UpdateDate`` are unix-epoch
    timestamps (``@timestampFormat("unixTimestamp")`` in the smithy model),
    not ISO strings — the Terraform AWS provider rejects strings with
    ``expected UpdateTimestamp to be a JSON Number``.
    """
    def wrap(value):
        return {
            "Options": value,
            "Status": {
                "CreationDate": domain.get("CreatedAtEpoch", _now_epoch()),
                "UpdateDate": _now_epoch(),
                "UpdateVersion": 1,
                "State": "Active",
                "PendingDeletion": False,
            },
        }

    return {
        "EngineVersion": wrap(domain.get("EngineVersion", _DEFAULT_ENGINE_VERSION)),
        "ClusterConfig": wrap(domain.get("ClusterConfig", _default_cluster_config(None))),
        "EBSOptions": wrap(domain.get("EBSOptions", _default_ebs(None))),
        "AccessPolicies": wrap(domain.get("AccessPolicies", "")),
        "IPAddressType": wrap(domain.get("IPAddressType", "ipv4")),
        "SnapshotOptions": wrap(domain.get("SnapshotOptions", {"AutomatedSnapshotStartHour": 0})),
        "VPCOptions": wrap(domain.get("VPCOptions") or {}),
        "CognitoOptions": wrap(domain.get("CognitoOptions") or {"Enabled": False}),
        "EncryptionAtRestOptions": wrap(domain.get(
            "EncryptionAtRestOptions", _default_encryption_at_rest(None)
        )),
        "NodeToNodeEncryptionOptions": wrap(domain.get(
            "NodeToNodeEncryptionOptions", _default_node_to_node_encryption(None)
        )),
        "AdvancedOptions": wrap(domain.get("AdvancedOptions") or {}),
        "LogPublishingOptions": wrap(domain.get("LogPublishingOptions") or {}),
        "DomainEndpointOptions": wrap(domain.get(
            "DomainEndpointOptions", _default_endpoint_options(None)
        )),
        "AdvancedSecurityOptions": wrap(domain.get(
            "AdvancedSecurityOptions",
            {"Enabled": False, "InternalUserDatabaseEnabled": False},
        )),
        "AutoTuneOptions": wrap(domain.get(
            "AutoTuneOptions", {"State": "DISABLED", "UseOffPeakWindow": False}
        )),
    }


def _describe_change_progress(name: str):
    if name not in _domains:
        return error_response_json(
            "ResourceNotFoundException", f"Domain not found: {name}", 404
        )
    return json_response({
        "ChangeProgressStatus": {
            "ChangeId": new_uuid(),
            "StartTime": _now_epoch(),
            "Status": "COMPLETED",
            "PendingProperties": [],
            "CompletedProperties": [],
            "TotalNumberOfStages": 0,
            "ChangeProgressStages": [],
            "ConfigChangeStatus": "Completed",
        }
    })


def _list_versions():
    return json_response({"Versions": list(_SUPPORTED_VERSIONS)})


def _get_compatible_versions(domain_name: str | None):
    if domain_name:
        if domain_name not in _domains:
            return error_response_json(
                "ResourceNotFoundException",
                f"Domain not found: {domain_name}",
                404,
            )
        src = _domains[domain_name].get("EngineVersion", _DEFAULT_ENGINE_VERSION)
        return json_response({
            "CompatibleVersions": [
                {"SourceVersion": src, "TargetVersions": list(_SUPPORTED_VERSIONS)},
            ],
        })
    return json_response({
        "CompatibleVersions": [
            {"SourceVersion": v, "TargetVersions": list(_SUPPORTED_VERSIONS)}
            for v in _SUPPORTED_VERSIONS
        ],
    })


# ---------------------------------------------------------------------------
# Tagging
# ---------------------------------------------------------------------------

def _add_tags(body: dict):
    arn = body.get("ARN")
    if not arn:
        return error_response_json("ValidationException", "ARN is required", 400)
    if not _arn_exists(arn):
        return error_response_json(
            "ResourceNotFoundException", f"Resource not found: {arn}", 404
        )
    incoming = body.get("TagList") or []
    existing = _tags.get(arn, [])
    by_key = {t["Key"]: t for t in existing}
    for t in incoming:
        k = t.get("Key", "")
        by_key[k] = {"Key": k, "Value": t.get("Value", "")}
    _tags[arn] = list(by_key.values())
    return json_response({})


def _list_tags(arn: str):
    if not arn:
        return error_response_json("ValidationException", "ARN is required", 400)
    if not _arn_exists(arn):
        return error_response_json(
            "ResourceNotFoundException", f"Resource not found: {arn}", 404
        )
    return json_response({"TagList": copy.deepcopy(_tags.get(arn, []))})


def _remove_tags(body: dict):
    arn = body.get("ARN")
    if not arn:
        return error_response_json("ValidationException", "ARN is required", 400)
    if not _arn_exists(arn):
        return error_response_json(
            "ResourceNotFoundException", f"Resource not found: {arn}", 404
        )
    keys = set(body.get("TagKeys") or [])
    remaining = [t for t in _tags.get(arn, []) if t.get("Key") not in keys]
    _tags[arn] = remaining
    return json_response({})


def _arn_exists(arn: str) -> bool:
    try:
        name = arn.split(":domain/")[1]
    except (IndexError, AttributeError):
        return False
    return name in _domains


# ---------------------------------------------------------------------------
# Request dispatch
# ---------------------------------------------------------------------------

async def handle_request(method, path, headers, body, query_params):
    try:
        data = json.loads(body) if body else {}
    except json.JSONDecodeError:
        return error_response_json("SerializationException", "Invalid JSON", 400)

    if not path.startswith("/2021-01-01"):
        return error_response_json(
            "InvalidAction", f"Unsupported OpenSearch path: {path}", 400
        )

    sub = path[len("/2021-01-01"):]

    # /domain  →  ListDomainNames
    if sub == "/domain" and method == "GET":
        et = query_params.get("engineType", [""])[0] or query_params.get("engineType") or ""
        if isinstance(et, list):
            et = et[0] if et else ""
        return _list_domain_names(et or None)

    # /opensearch/domain  →  CreateDomain
    if sub == "/opensearch/domain" and method == "POST":
        return _create_domain(data)

    # /opensearch/domain-info  →  DescribeDomains
    if sub == "/opensearch/domain-info" and method == "POST":
        return _describe_domains(data)

    # /opensearch/versions  →  ListVersions
    if sub == "/opensearch/versions" and method == "GET":
        return _list_versions()

    # /opensearch/compatibleVersions  →  GetCompatibleVersions
    if sub == "/opensearch/compatibleVersions" and method == "GET":
        dn = query_params.get("domainName", [""])
        dn = dn[0] if isinstance(dn, list) else dn
        return _get_compatible_versions(dn or None)

    # /opensearch/domain/{name}[/config|/progress]
    if sub.startswith("/opensearch/domain/"):
        tail = sub[len("/opensearch/domain/"):]
        parts = tail.split("/", 1)
        name = unquote(parts[0])
        rest = parts[1] if len(parts) > 1 else ""
        if not rest:
            if method == "GET":
                return _describe_domain(name)
            if method == "DELETE":
                return _delete_domain(name)
        elif rest == "config":
            if method == "GET":
                return _describe_domain_config(name)
            if method == "POST":
                return _update_domain_config(name, data)
        elif rest == "progress":
            if method == "GET":
                return _describe_change_progress(name)

    # /tags  and  /tags/?arn=...  and  /tags-removal
    if sub == "/tags" and method == "POST":
        return _add_tags(data)
    if sub in ("/tags/", "/tags") and method == "GET":
        arn = query_params.get("arn", [""])
        arn = arn[0] if isinstance(arn, list) else arn
        return _list_tags(arn or "")
    if sub == "/tags-removal" and method == "POST":
        return _remove_tags(data)

    return error_response_json(
        "InvalidAction",
        f"Unsupported OpenSearch operation: {method} {path}",
        400,
    )


# ---------------------------------------------------------------------------
# State lifecycle
# ---------------------------------------------------------------------------

def get_state():
    return {
        "domains": copy.deepcopy(_domains),
        "tags": copy.deepcopy(_tags),
    }


def restore_state(data):
    if not data:
        return
    for name, dom in data.get("domains", {}).items():
        # Runtime-only attrs: containers don't survive a MiniStack restart,
        # so drop the cached ids and endpoint. Callers will get the stub
        # endpoint until they re-create the domain.
        dom.pop("_docker_container_id", None)
        dom.pop("_docker_dashboards_container_id", None)
        dom.pop("DashboardEndpoint", None)
        _domains[name] = dom
    _tags.update(data.get("tags", {}))


def load_persisted_state(data):
    restore_state(data)


def reset():
    for domain in list(_domains.values()):
        _stop_opensearch_container(domain)
    _domains.clear()
    _tags.clear()
    _port_counter[0] = _OPENSEARCH_BASE_PORT
    _dashboards_port_counter[0] = _OPENSEARCH_DASHBOARDS_BASE_PORT
