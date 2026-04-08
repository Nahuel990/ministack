import io
import json
import os
import time
import zipfile
from urllib.parse import urlparse
import pytest
from botocore.exceptions import ClientError
import uuid as _uuid_mod

_CF_DIST_CONFIG = {
    "CallerReference": "cf-test-ref-1",
    "Origins": {
        "Quantity": 1,
        "Items": [
            {
                "Id": "myS3Origin",
                "DomainName": "mybucket.s3.amazonaws.com",
                "S3OriginConfig": {"OriginAccessIdentity": ""},
            }
        ],
    },
    "DefaultCacheBehavior": {
        "TargetOriginId": "myS3Origin",
        "ViewerProtocolPolicy": "redirect-to-https",
        "ForwardedValues": {
            "QueryString": False,
            "Cookies": {"Forward": "none"},
        },
        "MinTTL": 0,
    },
    "Comment": "test distribution",
    "Enabled": True,
}

def test_cloudfront_create_distribution(cloudfront):
    resp = cloudfront.create_distribution(DistributionConfig=_CF_DIST_CONFIG)
    dist = resp["Distribution"]
    assert dist["Id"]
    assert dist["DomainName"].endswith(".cloudfront.net")
    assert dist["Status"] == "Deployed"
    assert resp["ResponseMetadata"]["HTTPStatusCode"] == 201

def test_cloudfront_list_distributions(cloudfront):
    cfg_a = {**_CF_DIST_CONFIG, "CallerReference": "cf-list-a", "Comment": "list-a"}
    cfg_b = {**_CF_DIST_CONFIG, "CallerReference": "cf-list-b", "Comment": "list-b"}
    cloudfront.create_distribution(DistributionConfig=cfg_a)
    cloudfront.create_distribution(DistributionConfig=cfg_b)
    resp = cloudfront.list_distributions()
    dist_list = resp["DistributionList"]
    ids = [d["Id"] for d in dist_list.get("Items", [])]
    assert len(ids) >= 2

def test_cloudfront_get_distribution(cloudfront):
    cfg = {**_CF_DIST_CONFIG, "CallerReference": "cf-get-1", "Comment": "get-test"}
    create_resp = cloudfront.create_distribution(DistributionConfig=cfg)
    dist_id = create_resp["Distribution"]["Id"]

    resp = cloudfront.get_distribution(Id=dist_id)
    dist = resp["Distribution"]
    assert dist["Id"] == dist_id
    assert dist["DomainName"] == f"{dist_id}.cloudfront.net"
    assert dist["Status"] == "Deployed"

def test_cloudfront_get_distribution_config(cloudfront):
    cfg = {**_CF_DIST_CONFIG, "CallerReference": "cf-getcfg-1", "Comment": "getcfg-test"}
    create_resp = cloudfront.create_distribution(DistributionConfig=cfg)
    dist_id = create_resp["Distribution"]["Id"]
    etag = create_resp["ETag"]

    resp = cloudfront.get_distribution_config(Id=dist_id)
    assert resp["ETag"] == etag
    assert resp["DistributionConfig"]["Comment"] == "getcfg-test"

def test_cloudfront_update_distribution(cloudfront):
    cfg = {**_CF_DIST_CONFIG, "CallerReference": "cf-upd-1", "Comment": "before-update"}
    create_resp = cloudfront.create_distribution(DistributionConfig=cfg)
    dist_id = create_resp["Distribution"]["Id"]
    etag = create_resp["ETag"]

    updated_cfg = {**cfg, "CallerReference": "cf-upd-1", "Comment": "after-update"}
    upd_resp = cloudfront.update_distribution(DistributionConfig=updated_cfg, Id=dist_id, IfMatch=etag)
    assert upd_resp["Distribution"]["Id"] == dist_id
    assert upd_resp["ETag"] != etag  # new ETag issued

    get_resp = cloudfront.get_distribution_config(Id=dist_id)
    assert get_resp["DistributionConfig"]["Comment"] == "after-update"

def test_cloudfront_update_distribution_etag_mismatch(cloudfront):
    cfg = {**_CF_DIST_CONFIG, "CallerReference": "cf-etag-mismatch", "Comment": "mismatch-test"}
    create_resp = cloudfront.create_distribution(DistributionConfig=cfg)
    dist_id = create_resp["Distribution"]["Id"]

    with pytest.raises(ClientError) as exc:
        cloudfront.update_distribution(
            DistributionConfig=cfg, Id=dist_id, IfMatch="wrong-etag-value"
        )
    assert exc.value.response["Error"]["Code"] == "PreconditionFailed"

def test_cloudfront_delete_distribution(cloudfront):
    cfg = {**_CF_DIST_CONFIG, "CallerReference": "cf-del-1", "Comment": "delete-test", "Enabled": True}
    create_resp = cloudfront.create_distribution(DistributionConfig=cfg)
    dist_id = create_resp["Distribution"]["Id"]
    etag = create_resp["ETag"]

    # Must disable before deleting
    disabled_cfg = {**cfg, "Enabled": False}
    upd_resp = cloudfront.update_distribution(DistributionConfig=disabled_cfg, Id=dist_id, IfMatch=etag)
    new_etag = upd_resp["ETag"]

    cloudfront.delete_distribution(Id=dist_id, IfMatch=new_etag)

    with pytest.raises(ClientError) as exc:
        cloudfront.get_distribution(Id=dist_id)
    assert exc.value.response["Error"]["Code"] == "NoSuchDistribution"

def test_cloudfront_delete_enabled_distribution(cloudfront):
    cfg = {**_CF_DIST_CONFIG, "CallerReference": "cf-del-enabled", "Comment": "del-enabled-test", "Enabled": True}
    create_resp = cloudfront.create_distribution(DistributionConfig=cfg)
    dist_id = create_resp["Distribution"]["Id"]
    etag = create_resp["ETag"]

    with pytest.raises(ClientError) as exc:
        cloudfront.delete_distribution(Id=dist_id, IfMatch=etag)
    assert exc.value.response["Error"]["Code"] == "DistributionNotDisabled"

def test_cloudfront_get_nonexistent(cloudfront):
    with pytest.raises(ClientError) as exc:
        cloudfront.get_distribution(Id="ENONEXISTENT1234")
    assert exc.value.response["Error"]["Code"] == "NoSuchDistribution"

def test_cloudfront_create_invalidation(cloudfront):
    cfg = {**_CF_DIST_CONFIG, "CallerReference": "cf-inv-1", "Comment": "inv-test"}
    create_resp = cloudfront.create_distribution(DistributionConfig=cfg)
    dist_id = create_resp["Distribution"]["Id"]

    inv_resp = cloudfront.create_invalidation(
        DistributionId=dist_id,
        InvalidationBatch={
            "Paths": {"Quantity": 2, "Items": ["/index.html", "/static/*"]},
            "CallerReference": "inv-ref-1",
        },
    )
    inv = inv_resp["Invalidation"]
    assert inv["Id"]
    assert inv["Status"] == "Completed"
    assert inv_resp["ResponseMetadata"]["HTTPStatusCode"] == 201

def test_cloudfront_list_invalidations(cloudfront):
    cfg = {**_CF_DIST_CONFIG, "CallerReference": "cf-listinv-1", "Comment": "listinv-test"}
    create_resp = cloudfront.create_distribution(DistributionConfig=cfg)
    dist_id = create_resp["Distribution"]["Id"]

    cloudfront.create_invalidation(
        DistributionId=dist_id,
        InvalidationBatch={"Paths": {"Quantity": 1, "Items": ["/a"]}, "CallerReference": "inv-list-a"},
    )
    cloudfront.create_invalidation(
        DistributionId=dist_id,
        InvalidationBatch={"Paths": {"Quantity": 1, "Items": ["/b"]}, "CallerReference": "inv-list-b"},
    )

    resp = cloudfront.list_invalidations(DistributionId=dist_id)
    inv_list = resp["InvalidationList"]
    assert inv_list["Quantity"] == 2
    assert len(inv_list["Items"]) == 2

def test_cloudfront_get_invalidation(cloudfront):
    cfg = {**_CF_DIST_CONFIG, "CallerReference": "cf-getinv-1", "Comment": "getinv-test"}
    create_resp = cloudfront.create_distribution(DistributionConfig=cfg)
    dist_id = create_resp["Distribution"]["Id"]

    inv_resp = cloudfront.create_invalidation(
        DistributionId=dist_id,
        InvalidationBatch={
            "Paths": {"Quantity": 1, "Items": ["/getinv-path"]},
            "CallerReference": "inv-get-ref",
        },
    )
    inv_id = inv_resp["Invalidation"]["Id"]

    get_resp = cloudfront.get_invalidation(DistributionId=dist_id, Id=inv_id)
    inv = get_resp["Invalidation"]
    assert inv["Id"] == inv_id
    assert inv["Status"] == "Completed"
    assert "/getinv-path" in inv["InvalidationBatch"]["Paths"]["Items"]

def test_cloudfront_tags(cloudfront):
    """TagResource / ListTagsForResource / UntagResource for CloudFront distributions."""
    resp = cloudfront.create_distribution(
        DistributionConfig={
            "CallerReference": "tag-test-v42",
            "Origins": {"Items": [{"Id": "o1", "DomainName": "example.com",
                                   "S3OriginConfig": {"OriginAccessIdentity": ""}}], "Quantity": 1},
            "DefaultCacheBehavior": {
                "TargetOriginId": "o1", "ViewerProtocolPolicy": "allow-all",
                "ForwardedValues": {"QueryString": False, "Cookies": {"Forward": "none"}},
                "MinTTL": 0,
            },
            "Comment": "tag test", "Enabled": True,
        }
    )
    dist_arn = resp["Distribution"]["ARN"]

    cloudfront.tag_resource(
        Resource=dist_arn,
        Tags={"Items": [
            {"Key": "env", "Value": "test"},
            {"Key": "team", "Value": "platform"},
        ]},
    )

    tags = cloudfront.list_tags_for_resource(Resource=dist_arn)
    tag_map = {t["Key"]: t["Value"] for t in tags["Tags"]["Items"]}
    assert tag_map["env"] == "test"
    assert tag_map["team"] == "platform"

    cloudfront.untag_resource(
        Resource=dist_arn,
        TagKeys={"Items": ["team"]},
    )

    tags = cloudfront.list_tags_for_resource(Resource=dist_arn)
    tag_keys = [t["Key"] for t in tags["Tags"]["Items"]]
    assert "env" in tag_keys
    assert "team" not in tag_keys
