"""
Integration tests for MiniStack's Amazon OpenSearch Service emulator.

Covers the management plane (domain CRUD, config, tags, versions) via boto3's
``opensearch`` client. Data-plane tests (index/search against the real
opensearchproject container) are intentionally opt-in behind the
``OPENSEARCH_DATAPLANE=1`` env var so the main suite stays fast and doesn't
depend on a heavy JVM sidecar.
"""

import os

import pytest
from botocore.exceptions import ClientError

_NAME_COUNTER = {"n": 0}


def _unique_name(prefix: str = "testdomain") -> str:
    # Domain names are 3..28 chars; keep the suffix short.
    _NAME_COUNTER["n"] += 1
    suffix = f"{os.getpid() % 1000:03d}{_NAME_COUNTER['n']:03d}"
    return f"{prefix}-{suffix}"[:28]


def _create_domain(opensearch, name: str, **overrides):
    kwargs = {
        "DomainName": name,
        "EngineVersion": "OpenSearch_2.15",
        "ClusterConfig": {"InstanceType": "r6g.large.search", "InstanceCount": 1},
        "EBSOptions": {"EBSEnabled": True, "VolumeType": "gp3", "VolumeSize": 10},
    }
    kwargs.update(overrides)
    return opensearch.create_domain(**kwargs)


# ---------------------------------------------------------------------------
# Domain CRUD
# ---------------------------------------------------------------------------

def test_create_and_describe_domain(opensearch):
    name = _unique_name()
    resp = _create_domain(opensearch, name)
    status = resp["DomainStatus"]
    assert status["DomainName"] == name
    assert status["Created"] is True
    assert status["Deleted"] is False
    assert status["ARN"].startswith("arn:aws:es:")
    assert status["ARN"].endswith(f":domain/{name}")
    assert status["EngineVersion"] == "OpenSearch_2.15"
    assert status["Endpoint"]
    assert status["ClusterConfig"]["InstanceCount"] == 1

    described = opensearch.describe_domain(DomainName=name)["DomainStatus"]
    assert described["ARN"] == status["ARN"]
    assert described["DomainName"] == name
    opensearch.delete_domain(DomainName=name)


def test_describe_unknown_domain_raises(opensearch):
    with pytest.raises(ClientError) as exc:
        opensearch.describe_domain(DomainName="does-not-exist-xyz")
    assert exc.value.response["Error"]["Code"] == "ResourceNotFoundException"


def test_create_duplicate_domain_raises(opensearch):
    name = _unique_name("dup")
    _create_domain(opensearch, name)
    try:
        with pytest.raises(ClientError) as exc:
            _create_domain(opensearch, name)
        assert exc.value.response["Error"]["Code"] == "ResourceAlreadyExistsException"
    finally:
        opensearch.delete_domain(DomainName=name)


def test_delete_domain_returns_status_and_removes(opensearch):
    name = _unique_name("del")
    _create_domain(opensearch, name)
    resp = opensearch.delete_domain(DomainName=name)
    assert resp["DomainStatus"]["Deleted"] is True
    with pytest.raises(ClientError):
        opensearch.describe_domain(DomainName=name)


def test_delete_unknown_domain_raises(opensearch):
    with pytest.raises(ClientError) as exc:
        opensearch.delete_domain(DomainName="missing-domain-xyz")
    assert exc.value.response["Error"]["Code"] == "ResourceNotFoundException"


# ---------------------------------------------------------------------------
# List / batch describe
# ---------------------------------------------------------------------------

def test_list_domain_names(opensearch):
    names = [_unique_name("ls") for _ in range(3)]
    for n in names:
        _create_domain(opensearch, n)
    try:
        resp = opensearch.list_domain_names()
        returned = {d["DomainName"] for d in resp["DomainNames"]}
        assert set(names).issubset(returned)
        for d in resp["DomainNames"]:
            assert d["EngineType"] in ("OpenSearch", "Elasticsearch")
    finally:
        for n in names:
            opensearch.delete_domain(DomainName=n)


def test_list_domain_names_filtered_by_engine_type(opensearch):
    opensearch_name = _unique_name("os")
    es_name = _unique_name("es")
    _create_domain(opensearch, opensearch_name, EngineVersion="OpenSearch_2.15")
    _create_domain(opensearch, es_name, EngineVersion="Elasticsearch_7.10")
    try:
        os_only = opensearch.list_domain_names(EngineType="OpenSearch")
        assert opensearch_name in {d["DomainName"] for d in os_only["DomainNames"]}
        assert es_name not in {d["DomainName"] for d in os_only["DomainNames"]}

        es_only = opensearch.list_domain_names(EngineType="Elasticsearch")
        assert es_name in {d["DomainName"] for d in es_only["DomainNames"]}
        assert opensearch_name not in {d["DomainName"] for d in es_only["DomainNames"]}
    finally:
        opensearch.delete_domain(DomainName=opensearch_name)
        opensearch.delete_domain(DomainName=es_name)


def test_describe_domains_batch(opensearch):
    names = [_unique_name("batch") for _ in range(2)]
    for n in names:
        _create_domain(opensearch, n)
    try:
        resp = opensearch.describe_domains(DomainNames=names + ["missing-zzz"])
        returned = {d["DomainName"] for d in resp["DomainStatusList"]}
        assert set(names) == returned
        assert "missing-zzz" not in returned
    finally:
        for n in names:
            opensearch.delete_domain(DomainName=n)


# ---------------------------------------------------------------------------
# Domain config
# ---------------------------------------------------------------------------

def test_describe_domain_config_wraps_options_and_status(opensearch):
    name = _unique_name("cfg")
    _create_domain(opensearch, name)
    try:
        cfg = opensearch.describe_domain_config(DomainName=name)["DomainConfig"]
        for key in ("EngineVersion", "ClusterConfig", "EBSOptions",
                    "AccessPolicies", "SnapshotOptions", "DomainEndpointOptions"):
            assert "Options" in cfg[key]
            assert "Status" in cfg[key]
            assert cfg[key]["Status"]["State"] == "Active"
    finally:
        opensearch.delete_domain(DomainName=name)


def test_update_domain_config_changes_cluster_and_ebs(opensearch):
    name = _unique_name("upd")
    _create_domain(opensearch, name)
    try:
        opensearch.update_domain_config(
            DomainName=name,
            ClusterConfig={"InstanceType": "r6g.xlarge.search", "InstanceCount": 3},
            EBSOptions={"EBSEnabled": True, "VolumeType": "gp3", "VolumeSize": 20},
        )
        status = opensearch.describe_domain(DomainName=name)["DomainStatus"]
        assert status["ClusterConfig"]["InstanceType"] == "r6g.xlarge.search"
        assert status["ClusterConfig"]["InstanceCount"] == 3
        assert status["EBSOptions"]["VolumeSize"] == 20
    finally:
        opensearch.delete_domain(DomainName=name)


def test_update_unknown_domain_raises(opensearch):
    with pytest.raises(ClientError) as exc:
        opensearch.update_domain_config(
            DomainName="missing-xyz",
            ClusterConfig={"InstanceType": "r6g.large.search", "InstanceCount": 1},
        )
    assert exc.value.response["Error"]["Code"] == "ResourceNotFoundException"


def test_describe_change_progress(opensearch):
    name = _unique_name("prog")
    _create_domain(opensearch, name)
    try:
        resp = opensearch.describe_domain_change_progress(DomainName=name)
        assert resp["ChangeProgressStatus"]["Status"] == "COMPLETED"
        assert "ChangeId" in resp["ChangeProgressStatus"]
    finally:
        opensearch.delete_domain(DomainName=name)


# ---------------------------------------------------------------------------
# Versions
# ---------------------------------------------------------------------------

def test_list_versions_returns_known_engines(opensearch):
    resp = opensearch.list_versions()
    versions = resp["Versions"]
    assert any(v.startswith("OpenSearch_") for v in versions)
    assert any(v.startswith("Elasticsearch_") for v in versions)


def test_get_compatible_versions_without_domain(opensearch):
    resp = opensearch.get_compatible_versions()
    assert len(resp["CompatibleVersions"]) > 0
    assert "TargetVersions" in resp["CompatibleVersions"][0]


def test_get_compatible_versions_for_domain(opensearch):
    name = _unique_name("ver")
    _create_domain(opensearch, name, EngineVersion="OpenSearch_2.11")
    try:
        resp = opensearch.get_compatible_versions(DomainName=name)
        pairs = resp["CompatibleVersions"]
        assert any(p["SourceVersion"] == "OpenSearch_2.11" for p in pairs)
    finally:
        opensearch.delete_domain(DomainName=name)


# ---------------------------------------------------------------------------
# Tags
# ---------------------------------------------------------------------------

def test_tags_add_list_remove(opensearch):
    name = _unique_name("tag")
    created = _create_domain(opensearch, name)
    arn = created["DomainStatus"]["ARN"]
    try:
        opensearch.add_tags(
            ARN=arn,
            TagList=[
                {"Key": "env", "Value": "test"},
                {"Key": "team", "Value": "platform"},
            ],
        )
        listed = opensearch.list_tags(ARN=arn)["TagList"]
        tags_map = {t["Key"]: t["Value"] for t in listed}
        assert tags_map == {"env": "test", "team": "platform"}

        opensearch.add_tags(ARN=arn, TagList=[{"Key": "env", "Value": "prod"}])
        tags_map = {t["Key"]: t["Value"] for t in opensearch.list_tags(ARN=arn)["TagList"]}
        assert tags_map["env"] == "prod"
        assert tags_map["team"] == "platform"

        opensearch.remove_tags(ARN=arn, TagKeys=["team"])
        tags_map = {t["Key"]: t["Value"] for t in opensearch.list_tags(ARN=arn)["TagList"]}
        assert tags_map == {"env": "prod"}
    finally:
        opensearch.delete_domain(DomainName=name)


def test_create_domain_with_tag_list(opensearch):
    name = _unique_name("itag")
    resp = _create_domain(
        opensearch, name,
        TagList=[{"Key": "owner", "Value": "tf"}],
    )
    arn = resp["DomainStatus"]["ARN"]
    try:
        tags_map = {t["Key"]: t["Value"] for t in opensearch.list_tags(ARN=arn)["TagList"]}
        assert tags_map == {"owner": "tf"}
    finally:
        opensearch.delete_domain(DomainName=name)


def test_list_tags_unknown_arn_raises(opensearch):
    with pytest.raises(ClientError) as exc:
        opensearch.list_tags(ARN="arn:aws:es:us-east-1:000000000000:domain/missing-zzz")
    assert exc.value.response["Error"]["Code"] == "ResourceNotFoundException"


# ---------------------------------------------------------------------------
# Terraform compatibility
# ---------------------------------------------------------------------------

def test_terraform_aws_opensearch_domain_has_required_attributes(opensearch):
    """The Terraform AWS provider's aws_opensearch_domain resource reads a
    specific set of attributes from CreateDomain / DescribeDomain. Verify
    the full wire shape so ``terraform apply`` gets exactly what it expects.
    """
    name = _unique_name("tf")
    created = _create_domain(
        opensearch,
        name,
        ClusterConfig={
            "InstanceType": "r6g.large.search",
            "InstanceCount": 2,
            "DedicatedMasterEnabled": False,
            "ZoneAwarenessEnabled": False,
            "WarmEnabled": False,
        },
        EBSOptions={
            "EBSEnabled": True,
            "VolumeType": "gp3",
            "VolumeSize": 10,
        },
        EncryptionAtRestOptions={"Enabled": False},
        NodeToNodeEncryptionOptions={"Enabled": False},
        DomainEndpointOptions={
            "EnforceHTTPS": True,
            "TLSSecurityPolicy": "Policy-Min-TLS-1-2-2019-07",
        },
        AdvancedSecurityOptions={
            "Enabled": False,
            "InternalUserDatabaseEnabled": False,
        },
    )
    try:
        status = created["DomainStatus"]
        for key in (
            "DomainId", "DomainName", "ARN", "Created", "Endpoint",
            "EngineVersion", "ClusterConfig", "EBSOptions",
            "SnapshotOptions", "EncryptionAtRestOptions",
            "NodeToNodeEncryptionOptions", "DomainEndpointOptions",
            "AdvancedSecurityOptions", "AdvancedOptions", "AccessPolicies",
        ):
            assert key in status, f"missing {key} on DomainStatus"

        assert status["DomainEndpointOptions"]["EnforceHTTPS"] is True
        assert status["ClusterConfig"]["InstanceCount"] == 2
        assert status["Endpoint"]  # non-empty; real sidecar address

        # describe_domain round-trips the same shape (Terraform polls this
        # during refresh).
        roundtrip = opensearch.describe_domain(DomainName=name)["DomainStatus"]
        assert roundtrip["DomainEndpointOptions"]["EnforceHTTPS"] is True
        assert roundtrip["ARN"] == status["ARN"]
        assert roundtrip["Endpoint"] == status["Endpoint"]
    finally:
        opensearch.delete_domain(DomainName=name)


def test_terraform_aws_opensearch_domain_list_domain_names_engine_type(opensearch):
    """``data.aws_opensearch_domain`` + Terraform's domain listing use the
    ``EngineType`` field per entry."""
    name = _unique_name("tfls")
    _create_domain(opensearch, name)
    try:
        listed = opensearch.list_domain_names()["DomainNames"]
        match = next((d for d in listed if d["DomainName"] == name), None)
        assert match is not None
        assert match["EngineType"] == "OpenSearch"
    finally:
        opensearch.delete_domain(DomainName=name)


# ---------------------------------------------------------------------------
# Optional: real data plane against the opensearchproject sidecar
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    os.environ.get("OPENSEARCH_DATAPLANE") != "1",
    reason="Set OPENSEARCH_DATAPLANE=1 (both locally and on the MiniStack "
           "container) to run the per-domain data-plane smoke test. MiniStack "
           "then spawns a dedicated opensearchproject/opensearch container "
           "for each CreateDomain call; the endpoint returned is "
           "localhost:<OPENSEARCH_BASE_PORT+N>.",
)
def test_dataplane_index_and_search_against_sidecar(opensearch):
    """End-to-end smoke test: CreateDomain, wait for the spawned OpenSearch
    container to accept HTTP traffic, then hit the returned Endpoint
    directly with the real OpenSearch REST API and verify a round-trip."""
    import json as _json
    import time as _time
    import urllib.request

    name = _unique_name("dp")
    created = _create_domain(opensearch, name)
    endpoint = created["DomainStatus"]["Endpoint"]
    base = f"http://{endpoint}"

    # The opensearchproject/opensearch container takes ~30-60s to boot; poll
    # /_cluster/health until it responds before exercising index/search.
    deadline = _time.time() + 120
    last_err = None
    while _time.time() < deadline:
        try:
            r = urllib.request.urlopen(f"{base}/_cluster/health", timeout=3)
            if r.status == 200:
                break
        except Exception as e:
            last_err = e
        _time.sleep(2)
    else:
        opensearch.delete_domain(DomainName=name)
        pytest.fail(f"OpenSearch container at {endpoint} did not become ready: {last_err}")

    index = f"test-index-{_NAME_COUNTER['n']}"
    try:
        urllib.request.urlopen(
            urllib.request.Request(
                f"{base}/{index}/_doc/1?refresh=true",
                data=_json.dumps({"hello": "world"}).encode(),
                headers={"Content-Type": "application/json"},
                method="PUT",
            ),
            timeout=10,
        )
        got = urllib.request.urlopen(
            urllib.request.Request(
                f"{base}/{index}/_search",
                data=_json.dumps({"query": {"match_all": {}}}).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            ),
            timeout=10,
        )
        payload = _json.loads(got.read())
        hits = payload["hits"]["hits"]
        assert any(h["_source"].get("hello") == "world" for h in hits)
    finally:
        try:
            urllib.request.urlopen(
                urllib.request.Request(f"{base}/{index}", method="DELETE"),
                timeout=5,
            )
        except Exception:
            pass
        opensearch.delete_domain(DomainName=name)


# ---------------------------------------------------------------------------
# Dashboards sidecar
# ---------------------------------------------------------------------------

def test_dashboard_endpoint_absent_by_default(opensearch):
    """Without OPENSEARCH_DASHBOARDS=1 the DashboardEndpoint key must be
    absent from DescribeDomain — the AWS provider treats a present-but-null
    DashboardEndpoint as 'dashboard disabled' and we want to stay consistent
    with real AWS where the field is simply omitted."""
    name = _unique_name("nodash")
    try:
        _create_domain(opensearch, name)
        described = opensearch.describe_domain(DomainName=name)["DomainStatus"]
        assert "DashboardEndpoint" not in described
    finally:
        opensearch.delete_domain(DomainName=name)


@pytest.mark.skipif(
    os.environ.get("OPENSEARCH_DATAPLANE") != "1"
    or os.environ.get("OPENSEARCH_DASHBOARDS") != "1",
    reason="Set OPENSEARCH_DATAPLANE=1 AND OPENSEARCH_DASHBOARDS=1 (on the "
           "MiniStack container) to exercise the per-domain Dashboards "
           "sidecar. MiniStack then spawns a matching "
           "opensearchproject/opensearch-dashboards container wired to the "
           "cluster over ministack-opensearch-net; the endpoint returned is "
           "localhost:<OPENSEARCH_DASHBOARDS_BASE_PORT+N>.",
)
def test_dashboards_container_reachable(opensearch):
    """Poll `<DashboardEndpoint>/api/status` until the Dashboards server
    answers, then tear the domain down and verify cleanup."""
    import time as _time
    import urllib.request

    name = _unique_name("dash")
    created = _create_domain(opensearch, name)
    status = created["DomainStatus"]
    dash = status.get("DashboardEndpoint")
    assert dash, f"DashboardEndpoint not populated for domain {name}: {status}"
    assert dash.startswith("localhost:"), dash

    base = f"http://{dash}"
    # Dashboards boot is slower than the cluster — generous budget.
    deadline = _time.time() + 240
    last_err = None
    try:
        while _time.time() < deadline:
            try:
                r = urllib.request.urlopen(f"{base}/api/status", timeout=3)
                if r.status == 200:
                    break
            except Exception as e:
                last_err = e
            _time.sleep(3)
        else:
            pytest.fail(f"Dashboards at {dash} did not become ready: {last_err}")
    finally:
        opensearch.delete_domain(DomainName=name)
