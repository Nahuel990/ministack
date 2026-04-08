"""
Tests for multi-tenancy: dynamic Account ID derived from AWS_ACCESS_KEY_ID.

When the access key is a 12-digit number, MiniStack uses it as the Account ID
in all ARN generation. Non-numeric keys (like "test") fall back to the default
000000000000.
"""

import os

import boto3
import pytest
from botocore.config import Config

ENDPOINT = os.environ.get("MINISTACK_ENDPOINT", "http://localhost:4566")
REGION = "us-east-1"


def _client(service, access_key="test"):
    """Create a boto3 client with a specific access key."""
    return boto3.client(
        service,
        endpoint_url=ENDPOINT,
        aws_access_key_id=access_key,
        aws_secret_access_key="test",
        region_name=REGION,
        config=Config(region_name=REGION, retries={"max_attempts": 0}),
    )


# ── STS GetCallerIdentity ─────────────────────────────────

def test_default_account_id():
    """Non-numeric access key falls back to 000000000000."""
    sts = _client("sts", access_key="test")
    resp = sts.get_caller_identity()
    assert resp["Account"] == "000000000000"


def test_12_digit_access_key_becomes_account_id():
    """A 12-digit numeric access key is used as the Account ID."""
    sts = _client("sts", access_key="123456789012")
    resp = sts.get_caller_identity()
    assert resp["Account"] == "123456789012"


def test_different_12_digit_keys_get_different_accounts():
    """Two different 12-digit keys produce different account IDs."""
    sts_a = _client("sts", access_key="111111111111")
    sts_b = _client("sts", access_key="222222222222")
    assert sts_a.get_caller_identity()["Account"] == "111111111111"
    assert sts_b.get_caller_identity()["Account"] == "222222222222"


def test_non_12_digit_numeric_falls_back():
    """A numeric key that isn't exactly 12 digits uses the default."""
    sts = _client("sts", access_key="12345")
    resp = sts.get_caller_identity()
    assert resp["Account"] == "000000000000"


# ── S3: ARN isolation ─────────────────────────────────────

def test_sqs_queue_arn_uses_dynamic_account():
    """SQS queue ARN reflects the 12-digit access key as account ID."""
    sqs = _client("sqs", access_key="048408301323")
    q = sqs.create_queue(QueueName="mt-test-queue")
    try:
        attrs = sqs.get_queue_attributes(
            QueueUrl=q["QueueUrl"], AttributeNames=["QueueArn"]
        )
        arn = attrs["Attributes"]["QueueArn"]
        assert "048408301323" in arn, f"Expected account 048408301323 in ARN: {arn}"
    finally:
        sqs.delete_queue(QueueUrl=q["QueueUrl"])


def test_sqs_queues_isolated_by_account():
    """Queues created with different account keys are separate namespaces."""
    sqs_a = _client("sqs", access_key="111111111111")
    sqs_b = _client("sqs", access_key="222222222222")

    q_a = sqs_a.create_queue(QueueName="isolation-test")
    try:
        q_b = sqs_b.create_queue(QueueName="isolation-test")
        try:
            # Both should get their own queue with their own account in the ARN
            attrs_a = sqs_a.get_queue_attributes(
                QueueUrl=q_a["QueueUrl"], AttributeNames=["QueueArn"]
            )
            attrs_b = sqs_b.get_queue_attributes(
                QueueUrl=q_b["QueueUrl"], AttributeNames=["QueueArn"]
            )
            assert "111111111111" in attrs_a["Attributes"]["QueueArn"]
            assert "222222222222" in attrs_b["Attributes"]["QueueArn"]
        finally:
            sqs_b.delete_queue(QueueUrl=q_b["QueueUrl"])
    finally:
        sqs_a.delete_queue(QueueUrl=q_a["QueueUrl"])


# ── Lambda: ARN uses dynamic account ──────────────────────

def test_lambda_function_arn_uses_dynamic_account():
    """Lambda function ARN reflects the 12-digit access key."""
    lam = _client("lambda", access_key="999888777666")
    try:
        lam.create_function(
            FunctionName="mt-func",
            Runtime="python3.12",
            Role="arn:aws:iam::999888777666:role/test",
            Handler="index.handler",
            Code={"ZipFile": b"fake"},
        )
        resp = lam.get_function(FunctionName="mt-func")
        arn = resp["Configuration"]["FunctionArn"]
        assert "999888777666" in arn, f"Expected account in ARN: {arn}"
    finally:
        try:
            lam.delete_function(FunctionName="mt-func")
        except Exception:
            pass


# ── SSM: ARN uses dynamic account ────────────────────────

def test_ssm_parameter_arn_uses_dynamic_account():
    """SSM parameter ARN reflects the 12-digit access key."""
    ssm = _client("ssm", access_key="048408301323")
    ssm.put_parameter(
        Name="/mt-test/param1",
        Value="hello",
        Type="String",
    )
    try:
        resp = ssm.get_parameter(Name="/mt-test/param1")
        arn = resp["Parameter"]["ARN"]
        assert "048408301323" in arn, f"Expected account in ARN: {arn}"
    finally:
        ssm.delete_parameter(Name="/mt-test/param1")
