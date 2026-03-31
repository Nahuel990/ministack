"""Hypothesis-based integration tests for the CloudFormation service.

These tests run against a LIVE ministack server via boto3. They generate
random CloudFormation templates and verify deployment invariants.
"""

import json
import os
import string
import time
import urllib.request

import boto3
import pytest
from botocore.config import Config
from botocore.exceptions import ClientError
from hypothesis import given, settings, assume, HealthCheck
import hypothesis.strategies as st

from ministack.services.cloudformation import _parse_template

ENDPOINT = os.environ.get("MINISTACK_ENDPOINT", "http://localhost:4566")

_kwargs = dict(
    endpoint_url=ENDPOINT,
    aws_access_key_id="test",
    aws_secret_access_key="test",
    region_name="us-east-1",
    config=Config(region_name="us-east-1", retries={"max_attempts": 0}),
)


def _make_client(service):
    return boto3.client(service, **_kwargs)


def _wait_stack(cfn, name, timeout=25):
    deadline = time.time() + timeout
    while time.time() < deadline:
        stacks = cfn.describe_stacks(StackName=name)["Stacks"]
        status = stacks[0]["StackStatus"]
        if not status.endswith("_IN_PROGRESS"):
            return stacks[0]
        time.sleep(0.5)
    raise TimeoutError(f"Stack {name} stuck at {status}")


def _reset_server():
    req = urllib.request.Request(
        f"{ENDPOINT}/_ministack/reset", data=b"", method="POST"
    )
    try:
        urllib.request.urlopen(req, timeout=5)
    except Exception:
        pass


@pytest.fixture(autouse=True)
def reset_between_tests():
    _reset_server()


# ---------------------------------------------------------------------------
# Resource types and minimal properties
# ---------------------------------------------------------------------------

RESOURCE_TYPES = [
    "AWS::S3::Bucket",
    "AWS::SQS::Queue",
    "AWS::SNS::Topic",
    "AWS::DynamoDB::Table",
    "AWS::Lambda::Function",
    "AWS::IAM::Role",
    "AWS::SSM::Parameter",
    "AWS::Logs::LogGroup",
    "AWS::Events::Rule",
]

TYPES_WITH_ARN = {
    "AWS::S3::Bucket",
    "AWS::SQS::Queue",
    "AWS::SNS::Topic",
    "AWS::DynamoDB::Table",
    "AWS::Lambda::Function",
    "AWS::IAM::Role",
    "AWS::Logs::LogGroup",
    "AWS::Events::Rule",
}


def _props_for_type(rtype, name):
    if rtype == "AWS::S3::Bucket":
        return {"BucketName": name}
    elif rtype == "AWS::SQS::Queue":
        return {"QueueName": name}
    elif rtype == "AWS::SNS::Topic":
        return {"TopicName": name}
    elif rtype == "AWS::DynamoDB::Table":
        return {
            "TableName": name,
            "KeySchema": [{"AttributeName": "id", "KeyType": "HASH"}],
            "AttributeDefinitions": [
                {"AttributeName": "id", "AttributeType": "S"}
            ],
        }
    elif rtype == "AWS::Lambda::Function":
        return {
            "FunctionName": name,
            "Runtime": "python3.12",
            "Handler": "index.handler",
            "Role": "arn:aws:iam::000000000000:role/fake",
            "Code": {"ZipFile": "def handler(e,c): pass"},
        }
    elif rtype == "AWS::IAM::Role":
        return {
            "RoleName": name,
            "AssumeRolePolicyDocument": {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Principal": {"Service": "lambda.amazonaws.com"},
                        "Action": "sts:AssumeRole",
                    }
                ],
            },
        }
    elif rtype == "AWS::SSM::Parameter":
        return {"Name": f"/{name}", "Type": "String", "Value": "test"}
    elif rtype == "AWS::Logs::LogGroup":
        return {"LogGroupName": f"/{name}"}
    elif rtype == "AWS::Events::Rule":
        return {
            "Name": name,
            "ScheduleExpression": "rate(1 hour)",
            "State": "ENABLED",
        }
    return {}


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------


@st.composite
def random_cfn_template(draw):
    slug = draw(
        st.text(alphabet=string.ascii_lowercase, min_size=4, max_size=6)
    )
    count = draw(st.integers(min_value=1, max_value=6))
    resources = {}
    for i in range(count):
        rtype = draw(st.sampled_from(RESOURCE_TYPES))
        name = f"hyp-{slug}-{i}"
        resources[f"Res{i}"] = {
            "Type": rtype,
            "Properties": _props_for_type(rtype, name),
        }
    template = {
        "AWSTemplateFormatVersion": "2010-09-09",
        "Resources": resources,
    }
    return (f"hyp-{slug}", template)


@st.composite
def random_cfn_template_with_outputs(draw):
    stack_name, template = draw(random_cfn_template())
    outputs = {}
    for logical_id, res in template["Resources"].items():
        idx = logical_id.replace("Res", "")
        outputs[f"Out{idx}Ref"] = {"Value": {"Ref": logical_id}}
        if res["Type"] in TYPES_WITH_ARN:
            outputs[f"Out{idx}Arn"] = {
                "Value": {"Fn::GetAtt": [logical_id, "Arn"]}
            }
    template["Outputs"] = outputs
    return (stack_name, template)


@st.composite
def broken_cfn_template(draw):
    kind = draw(st.sampled_from(["bad_type", "not_json"]))
    slug = draw(
        st.text(alphabet=string.ascii_lowercase, min_size=4, max_size=6)
    )
    stack_name = f"hyp-brk-{slug}"
    if kind == "bad_type":
        template = {
            "AWSTemplateFormatVersion": "2010-09-09",
            "Resources": {
                "Bad0": {
                    "Type": "AWS::Fake::DoesNotExist",
                    "Properties": {},
                }
            },
        }
        return (kind, stack_name, json.dumps(template))
    else:
        raw = draw(
            st.text(
                alphabet=string.ascii_letters + string.digits,
                min_size=5,
                max_size=30,
            )
        )
        return (kind, stack_name, raw)


# ---------------------------------------------------------------------------
# Snapshot helper for cleanup verification
# ---------------------------------------------------------------------------


def _snapshot():
    s3 = _make_client("s3")
    sqs = _make_client("sqs")
    sns = _make_client("sns")
    ddb = _make_client("dynamodb")
    lam = _make_client("lambda")
    iam = _make_client("iam")
    ssm = _make_client("ssm")
    logs = _make_client("logs")
    events = _make_client("events")
    return {
        "s3": len(s3.list_buckets()["Buckets"]),
        "sqs": len(sqs.list_queues().get("QueueUrls", [])),
        "sns": len(sns.list_topics()["Topics"]),
        "dynamodb": len(ddb.list_tables()["TableNames"]),
        "lambda": len(lam.list_functions()["Functions"]),
        "iam_roles": len(
            [
                r
                for r in iam.list_roles()["Roles"]
                if r["RoleName"].startswith("hyp-")
            ]
        ),
        "ssm": len(ssm.describe_parameters()["Parameters"]),
        "logs": len(logs.describe_log_groups()["logGroups"]),
        "events": len(events.list_rules()["Rules"]),
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@given(random_cfn_template())
@settings(max_examples=100)
def test_generator_produces_valid_templates(data):
    """Generator produces structurally valid templates (no server)."""
    stack_name, template = data
    assert "Resources" in template
    assert len(template["Resources"]) >= 1
    for res in template["Resources"].values():
        assert res["Type"] in RESOURCE_TYPES
    parsed = _parse_template(json.dumps(template))
    assert "Resources" in parsed


@given(random_cfn_template())
@settings(
    max_examples=20,
    deadline=45000,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_all_resources_tracked_in_stack(data):
    """Every template resource appears in describe_stack_resources."""
    cfn = _make_client("cloudformation")
    stack_name, template = data
    cfn.create_stack(StackName=stack_name, TemplateBody=json.dumps(template))
    stack = _wait_stack(cfn, stack_name)
    assume(stack["StackStatus"] == "CREATE_COMPLETE")
    resources = cfn.describe_stack_resources(StackName=stack_name)[
        "StackResources"
    ]
    template_ids = set(template["Resources"].keys())
    stack_ids = {r["LogicalResourceId"] for r in resources}
    assert template_ids == stack_ids
    cfn.delete_stack(StackName=stack_name)
    _wait_stack(cfn, stack_name)


@given(random_cfn_template())
@settings(
    max_examples=15,
    deadline=60000,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_delete_cleans_up_all_resources(data):
    """Deleting a stack leaves service state unchanged."""
    cfn = _make_client("cloudformation")
    stack_name, template = data
    before = _snapshot()
    cfn.create_stack(StackName=stack_name, TemplateBody=json.dumps(template))
    stack = _wait_stack(cfn, stack_name)
    assume(stack["StackStatus"] == "CREATE_COMPLETE")
    cfn.delete_stack(StackName=stack_name)
    _wait_stack(cfn, stack_name)
    after = _snapshot()
    assert before == after, (
        f"Resource leak detected: before={before}, after={after}"
    )


@given(random_cfn_template_with_outputs())
@settings(
    max_examples=20,
    deadline=45000,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_outputs_resolve_to_nonempty(data):
    """All declared Outputs resolve to non-empty values."""
    cfn = _make_client("cloudformation")
    stack_name, template = data
    cfn.create_stack(StackName=stack_name, TemplateBody=json.dumps(template))
    stack = _wait_stack(cfn, stack_name)
    assume(stack["StackStatus"] == "CREATE_COMPLETE")
    outputs = {o["OutputKey"]: o["OutputValue"] for o in stack.get("Outputs", [])}
    expected_keys = set(template["Outputs"].keys())
    assert expected_keys == set(outputs.keys()), (
        f"Missing outputs: {expected_keys - set(outputs.keys())}"
    )
    for key, value in outputs.items():
        assert value, f"Output {key} resolved to empty"
        assert "${" not in value, f"Output {key} contains unresolved placeholder: {value}"
        if key.endswith("Arn"):
            assert value.startswith("arn:aws:"), (
                f"Output {key} should be an ARN, got: {value}"
            )
    cfn.delete_stack(StackName=stack_name)
    _wait_stack(cfn, stack_name)


@given(random_cfn_template())
@settings(
    max_examples=20,
    deadline=45000,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_stack_events_cover_all_resources(data):
    """Every template resource has a CREATE_COMPLETE event."""
    cfn = _make_client("cloudformation")
    stack_name, template = data
    cfn.create_stack(StackName=stack_name, TemplateBody=json.dumps(template))
    stack = _wait_stack(cfn, stack_name)
    assume(stack["StackStatus"] == "CREATE_COMPLETE")
    events = cfn.describe_stack_events(StackName=stack_name)["StackEvents"]
    create_complete_ids = {
        e["LogicalResourceId"]
        for e in events
        if e.get("ResourceStatus") == "CREATE_COMPLETE"
    }
    template_ids = set(template["Resources"].keys())
    assert template_ids.issubset(create_complete_ids), (
        f"Resources missing CREATE_COMPLETE event: {template_ids - create_complete_ids}"
    )
    stack_name_in_events = {e.get("StackName") for e in events}
    assert stack_name in stack_name_in_events
    cfn.delete_stack(StackName=stack_name)
    _wait_stack(cfn, stack_name)


@given(broken_cfn_template())
@settings(
    max_examples=20,
    deadline=45000,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_invalid_templates_return_structured_errors(data):
    """Invalid templates produce errors, not hangs."""
    kind, stack_name, body = data
    cfn = _make_client("cloudformation")
    if kind == "not_json":
        with pytest.raises(ClientError):
            cfn.create_stack(StackName=stack_name, TemplateBody=body)
    elif kind == "bad_type":
        cfn.create_stack(StackName=stack_name, TemplateBody=body)
        stack = _wait_stack(cfn, stack_name)
        assert stack["StackStatus"] in ("CREATE_FAILED", "ROLLBACK_COMPLETE"), (
            f"Expected terminal failure status, got: {stack['StackStatus']}"
        )
