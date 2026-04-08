import io
import json
import os
import time
import zipfile
from urllib.parse import urlparse
import pytest
from botocore.exceptions import ClientError
import uuid as _uuid_mod

def test_elasticache_create(ec):
    ec.create_cache_cluster(
        CacheClusterId="test-redis",
        Engine="redis",
        CacheNodeType="cache.t3.micro",
        NumCacheNodes=1,
    )
    resp = ec.describe_cache_clusters(CacheClusterId="test-redis")
    clusters = resp["CacheClusters"]
    assert len(clusters) == 1
    assert clusters[0]["CacheClusterId"] == "test-redis"
    assert clusters[0]["Engine"] == "redis"

def test_elasticache_replication_group(ec):
    ec.create_replication_group(
        ReplicationGroupId="test-rg",
        ReplicationGroupDescription="Test replication group",
        CacheNodeType="cache.t3.micro",
    )
    resp = ec.describe_replication_groups(ReplicationGroupId="test-rg")
    assert resp["ReplicationGroups"][0]["ReplicationGroupId"] == "test-rg"

def test_elasticache_engines(ec):
    resp = ec.describe_cache_engine_versions(Engine="redis")
    assert len(resp["CacheEngineVersions"]) > 0

def test_elasticache_modify_subnet_group(ec):
    ec.create_cache_subnet_group(
        CacheSubnetGroupName="test-mod-ecsg",
        CacheSubnetGroupDescription="Test EC SG",
        SubnetIds=["subnet-aaa"],
    )
    ec.modify_cache_subnet_group(
        CacheSubnetGroupName="test-mod-ecsg",
        CacheSubnetGroupDescription="Updated EC SG",
        SubnetIds=["subnet-bbb"],
    )
    resp = ec.describe_cache_subnet_groups(CacheSubnetGroupName="test-mod-ecsg")
    assert resp["CacheSubnetGroups"][0]["CacheSubnetGroupDescription"] == "Updated EC SG"

def test_elasticache_user_crud(ec):
    ec.create_user(
        UserId="test-user-1",
        UserName="test-user-1",
        Engine="redis",
        AccessString="on ~* +@all",
        NoPasswordRequired=True,
    )
    resp = ec.describe_users(UserId="test-user-1")
    assert len(resp["Users"]) >= 1
    assert resp["Users"][0]["UserId"] == "test-user-1"
    ec.modify_user(UserId="test-user-1", AccessString="on ~keys:* +get")
    ec.delete_user(UserId="test-user-1")

def test_elasticache_user_group_crud(ec):
    ec.create_user(
        UserId="ug-usr-1",
        UserName="ug-usr-1",
        Engine="redis",
        AccessString="on ~* +@all",
        NoPasswordRequired=True,
    )
    ec.create_user_group(UserGroupId="test-ug-1", Engine="redis", UserIds=["ug-usr-1"])
    resp = ec.describe_user_groups(UserGroupId="test-ug-1")
    assert len(resp["UserGroups"]) >= 1
    assert resp["UserGroups"][0]["UserGroupId"] == "test-ug-1"
    ec.delete_user_group(UserGroupId="test-ug-1")
    ec.delete_user(UserId="ug-usr-1")

def test_elasticache_reset_clears_param_groups():
    """ElastiCache reset clears _param_group_params and resets port counter."""
    from ministack.services import elasticache as _ec
    _ec._param_group_params["test-group"] = {"param1": "val1"}
    _ec._port_counter[0] = 99999
    _ec.reset()
    assert not _ec._param_group_params
    assert _ec._port_counter[0] == _ec.BASE_PORT

def test_elasticache_parameter_group_crud(ec):
    """CreateCacheParameterGroup / DescribeCacheParameterGroups / DeleteCacheParameterGroup."""
    ec.create_cache_parameter_group(
        CacheParameterGroupName="test-pg-v39",
        CacheParameterGroupFamily="redis7",
        Description="Test param group",
    )
    desc = ec.describe_cache_parameter_groups(CacheParameterGroupName="test-pg-v39")
    groups = desc["CacheParameterGroups"]
    assert len(groups) == 1
    assert groups[0]["CacheParameterGroupName"] == "test-pg-v39"
    assert groups[0]["CacheParameterGroupFamily"] == "redis7"
    ec.delete_cache_parameter_group(CacheParameterGroupName="test-pg-v39")

def test_elasticache_snapshot_crud(ec):
    """CreateSnapshot / DescribeSnapshots / DeleteSnapshot."""
    ec.create_cache_cluster(
        CacheClusterId="snap-cluster-v39",
        Engine="redis",
        CacheNodeType="cache.t3.micro",
        NumCacheNodes=1,
    )
    ec.create_snapshot(SnapshotName="test-snap-v39", CacheClusterId="snap-cluster-v39")
    desc = ec.describe_snapshots(SnapshotName="test-snap-v39")
    assert len(desc["Snapshots"]) == 1
    assert desc["Snapshots"][0]["SnapshotName"] == "test-snap-v39"
    ec.delete_snapshot(SnapshotName="test-snap-v39")

def test_elasticache_tags(ec):
    """AddTagsToResource / ListTagsForResource / RemoveTagsFromResource."""
    ec.create_cache_cluster(
        CacheClusterId="tag-cluster-v39",
        Engine="redis",
        CacheNodeType="cache.t3.micro",
        NumCacheNodes=1,
    )
    arn = "arn:aws:elasticache:us-east-1:000000000000:cluster:tag-cluster-v39"
    ec.add_tags_to_resource(
        ResourceName=arn,
        Tags=[{"Key": "env", "Value": "test"}, {"Key": "team", "Value": "platform"}],
    )
    tags = ec.list_tags_for_resource(ResourceName=arn)
    tag_map = {t["Key"]: t["Value"] for t in tags["TagList"]}
    assert tag_map["env"] == "test"
    assert tag_map["team"] == "platform"
    ec.remove_tags_from_resource(ResourceName=arn, TagKeys=["team"])
    tags = ec.list_tags_for_resource(ResourceName=arn)
    tag_keys = [t["Key"] for t in tags["TagList"]]
    assert "env" in tag_keys
    assert "team" not in tag_keys

# Migrated from test_ec.py
def test_elasticache_create_cluster_v2(ec):
    resp = ec.create_cache_cluster(
        CacheClusterId="ec-cc-v2",
        Engine="redis",
        CacheNodeType="cache.t3.micro",
        NumCacheNodes=1,
    )
    c = resp["CacheCluster"]
    assert c["CacheClusterId"] == "ec-cc-v2"
    assert c["Engine"] == "redis"
    assert c["CacheClusterStatus"] == "available"
    assert len(c["CacheNodes"]) == 1

def test_elasticache_describe_clusters_v2(ec):
    ec.create_cache_cluster(
        CacheClusterId="ec-dc-v2a",
        Engine="redis",
        CacheNodeType="cache.t3.micro",
        NumCacheNodes=1,
    )
    ec.create_cache_cluster(
        CacheClusterId="ec-dc-v2b",
        Engine="memcached",
        CacheNodeType="cache.t3.micro",
        NumCacheNodes=1,
    )
    resp = ec.describe_cache_clusters()
    ids = [c["CacheClusterId"] for c in resp["CacheClusters"]]
    assert "ec-dc-v2a" in ids
    assert "ec-dc-v2b" in ids

    resp2 = ec.describe_cache_clusters(CacheClusterId="ec-dc-v2b")
    assert resp2["CacheClusters"][0]["Engine"] == "memcached"

def test_elasticache_replication_group_v2(ec):
    resp = ec.create_replication_group(
        ReplicationGroupId="ec-rg-v2",
        ReplicationGroupDescription="Test RG v2",
        Engine="redis",
        CacheNodeType="cache.t3.micro",
        NumNodeGroups=1,
        ReplicasPerNodeGroup=1,
    )
    rg = resp["ReplicationGroup"]
    assert rg["ReplicationGroupId"] == "ec-rg-v2"
    assert rg["Status"] == "available"
    assert len(rg["NodeGroups"]) == 1

    desc = ec.describe_replication_groups(ReplicationGroupId="ec-rg-v2")
    assert desc["ReplicationGroups"][0]["ReplicationGroupId"] == "ec-rg-v2"

def test_elasticache_engine_versions_v2(ec):
    redis = ec.describe_cache_engine_versions(Engine="redis")
    assert len(redis["CacheEngineVersions"]) > 0
    assert all(v["Engine"] == "redis" for v in redis["CacheEngineVersions"])

    mc = ec.describe_cache_engine_versions(Engine="memcached")
    assert len(mc["CacheEngineVersions"]) > 0

def test_elasticache_tags_v2(ec):
    ec.create_cache_cluster(
        CacheClusterId="ec-tag-v2",
        Engine="redis",
        CacheNodeType="cache.t3.micro",
        NumCacheNodes=1,
    )
    arn = ec.describe_cache_clusters(CacheClusterId="ec-tag-v2")["CacheClusters"][0]["ARN"]

    ec.add_tags_to_resource(
        ResourceName=arn,
        Tags=[
            {"Key": "env", "Value": "prod"},
            {"Key": "tier", "Value": "cache"},
        ],
    )
    tags = ec.list_tags_for_resource(ResourceName=arn)["TagList"]
    tag_map = {t["Key"]: t["Value"] for t in tags}
    assert tag_map["env"] == "prod"
    assert tag_map["tier"] == "cache"

    ec.remove_tags_from_resource(ResourceName=arn, TagKeys=["env"])
    tags2 = ec.list_tags_for_resource(ResourceName=arn)["TagList"]
    assert not any(t["Key"] == "env" for t in tags2)
    assert any(t["Key"] == "tier" for t in tags2)

def test_elasticache_snapshot_v2(ec):
    ec.create_cache_cluster(
        CacheClusterId="ec-snap-v2",
        Engine="redis",
        CacheNodeType="cache.t3.micro",
        NumCacheNodes=1,
    )
    resp = ec.create_snapshot(SnapshotName="ec-snap-v2-s1", CacheClusterId="ec-snap-v2")
    assert resp["Snapshot"]["SnapshotName"] == "ec-snap-v2-s1"
    assert resp["Snapshot"]["SnapshotStatus"] == "available"

    desc = ec.describe_snapshots(SnapshotName="ec-snap-v2-s1")
    assert len(desc["Snapshots"]) == 1
    assert desc["Snapshots"][0]["SnapshotName"] == "ec-snap-v2-s1"

def test_elasticache_describe_cache_parameters(ec):
    """DescribeCacheParameters returns parameters for a parameter group."""
    ec.create_cache_parameter_group(
        CacheParameterGroupName="qa-ec-params",
        CacheParameterGroupFamily="redis7.0",
        Description="test",
    )
    resp = ec.describe_cache_parameters(CacheParameterGroupName="qa-ec-params")
    assert "Parameters" in resp
    assert len(resp["Parameters"]) > 0

def test_elasticache_modify_cache_parameter_group(ec):
    """ModifyCacheParameterGroup updates parameter values."""
    ec.create_cache_parameter_group(
        CacheParameterGroupName="qa-ec-modify-params",
        CacheParameterGroupFamily="redis7.0",
        Description="test",
    )
    ec.modify_cache_parameter_group(
        CacheParameterGroupName="qa-ec-modify-params",
        ParameterNameValues=[{"ParameterName": "maxmemory-policy", "ParameterValue": "allkeys-lru"}],
    )
    params = ec.describe_cache_parameters(CacheParameterGroupName="qa-ec-modify-params")["Parameters"]
    maxmem = next((p for p in params if p["ParameterName"] == "maxmemory-policy"), None)
    assert maxmem is not None
    assert maxmem["ParameterValue"] == "allkeys-lru"

