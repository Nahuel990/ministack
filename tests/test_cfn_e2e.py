"""
End-to-end test: deploy a CloudFormation stack and use every resource it creates.

Verifies that CFN-provisioned resources are fully functional — not just metadata
stubs but real working S3 buckets, SQS queues, SNS topics, Lambda functions, etc.

Requires a running ministack server on MINISTACK_ENDPOINT (default localhost:4566).
"""

import json
import os
import time

import boto3
import pytest
from botocore.config import Config
from botocore.exceptions import ClientError

ENDPOINT = os.environ.get("MINISTACK_ENDPOINT", "http://localhost:4566")
STACK_NAME = "e2e-test"

_kwargs = dict(
    endpoint_url=ENDPOINT,
    aws_access_key_id="test",
    aws_secret_access_key="test",
    region_name="us-east-1",
    config=Config(region_name="us-east-1", retries={"max_attempts": 0}),
)


def _client(service):
    return boto3.client(service, **_kwargs)


def _wait_stack(cfn, name, timeout=15):
    deadline = time.time() + timeout
    while time.time() < deadline:
        stacks = cfn.describe_stacks(StackName=name)["Stacks"]
        status = stacks[0]["StackStatus"]
        if not status.endswith("_IN_PROGRESS"):
            return stacks[0]
        time.sleep(0.5)
    raise TimeoutError(f"Stack {name} stuck at {status}")


TEMPLATE = """
AWSTemplateFormatVersion: '2010-09-09'
Description: E2E test stack — verifies CFN resources are functional

Parameters:
  Env:
    Type: String
    Default: e2etest

Resources:
  Bucket:
    Type: AWS::S3::Bucket
    Properties:
      BucketName: !Sub "${AWS::StackName}-${Env}-assets"

  Queue:
    Type: AWS::SQS::Queue
    Properties:
      QueueName: !Sub "${AWS::StackName}-${Env}-events"
      VisibilityTimeout: 120

  Topic:
    Type: AWS::SNS::Topic
    Properties:
      TopicName: !Sub "${AWS::StackName}-${Env}-alerts"

  Role:
    Type: AWS::IAM::Role
    Properties:
      RoleName: !Sub "${AWS::StackName}-${Env}-role"
      AssumeRolePolicyDocument:
        Version: '2012-10-17'
        Statement:
          - Effect: Allow
            Principal:
              Service: lambda.amazonaws.com
            Action: sts:AssumeRole

  Processor:
    Type: AWS::Lambda::Function
    Properties:
      FunctionName: !Sub "${AWS::StackName}-${Env}-processor"
      Runtime: python3.12
      Handler: index.handler
      Role: !GetAtt Role.Arn
      Code:
        ZipFile: |
          def handler(event, context):
              return {"statusCode": 200}

  QueueUrlParam:
    Type: AWS::SSM::Parameter
    Properties:
      Name: !Sub "/${AWS::StackName}/${Env}/queue-url"
      Type: String
      Value: !Ref Queue

Outputs:
  BucketName:
    Value: !Ref Bucket
    Export:
      Name: !Sub "${AWS::StackName}-bucket"
  QueueUrl:
    Value: !Ref Queue
  TopicArn:
    Value: !Ref Topic
  ProcessorArn:
    Value: !GetAtt Processor.Arn
  RoleArn:
    Value: !GetAtt Role.Arn
"""


@pytest.fixture(scope="module")
def stack():
    """Deploy the stack once for all tests in this module."""
    cfn = _client("cloudformation")

    # Clean up from previous run
    try:
        cfn.delete_stack(StackName=STACK_NAME)
        _wait_stack(cfn, STACK_NAME)
    except ClientError:
        pass

    cfn.create_stack(StackName=STACK_NAME, TemplateBody=TEMPLATE)
    s = _wait_stack(cfn, STACK_NAME)
    assert s["StackStatus"] == "CREATE_COMPLETE", f"Stack failed: {s.get('StackStatusReason')}"

    outputs = {o["OutputKey"]: o["OutputValue"] for o in s.get("Outputs", [])}
    yield outputs

    cfn.delete_stack(StackName=STACK_NAME)
    _wait_stack(cfn, STACK_NAME)


# ── S3 ──

def test_s3_put_and_get(stack):
    s3 = _client("s3")
    bucket = stack["BucketName"]
    body = json.dumps({"id": "001", "total": 99.99})

    s3.put_object(Bucket=bucket, Key="orders/order-001.json", Body=body.encode())
    obj = s3.get_object(Bucket=bucket, Key="orders/order-001.json")
    data = json.loads(obj["Body"].read())

    assert data["id"] == "001"
    assert data["total"] == 99.99


def test_s3_list_objects(stack):
    s3 = _client("s3")
    bucket = stack["BucketName"]

    s3.put_object(Bucket=bucket, Key="docs/readme.txt", Body=b"hello")
    listing = s3.list_objects_v2(Bucket=bucket)

    assert listing["KeyCount"] >= 1
    keys = [o["Key"] for o in listing["Contents"]]
    assert "docs/readme.txt" in keys


# ── SQS ──

def test_sqs_send_receive_delete(stack):
    sqs = _client("sqs")
    url = stack["QueueUrl"]

    sqs.send_message(QueueUrl=url, MessageBody=json.dumps({"event": "order.created"}))
    sqs.send_message(QueueUrl=url, MessageBody=json.dumps({"event": "order.shipped"}))

    msgs = sqs.receive_message(QueueUrl=url, MaxNumberOfMessages=10, WaitTimeSeconds=1)
    received = msgs.get("Messages", [])
    assert len(received) == 2

    events = sorted(json.loads(m["Body"])["event"] for m in received)
    assert events == ["order.created", "order.shipped"]

    for m in received:
        sqs.delete_message(QueueUrl=url, ReceiptHandle=m["ReceiptHandle"])

    empty = sqs.receive_message(QueueUrl=url, MaxNumberOfMessages=10, WaitTimeSeconds=1)
    assert len(empty.get("Messages", [])) == 0


# ── SNS ──

def test_sns_publish(stack):
    sns = _client("sns")
    topic_arn = stack["TopicArn"]

    resp = sns.publish(TopicArn=topic_arn, Subject="Test Alert",
                       Message=json.dumps({"alert": "test", "severity": "low"}))
    assert "MessageId" in resp


# ── SSM ──

def test_ssm_read_cfn_param(stack):
    ssm = _client("ssm")
    param = ssm.get_parameter(Name=f"/{STACK_NAME}/e2etest/queue-url")["Parameter"]

    assert param["Value"] == stack["QueueUrl"]


def test_ssm_write_and_read(stack):
    ssm = _client("ssm")
    ssm.put_parameter(Name=f"/{STACK_NAME}/e2etest/flags", Type="String",
                      Value=json.dumps({"dark_mode": True}))

    flags = json.loads(ssm.get_parameter(Name=f"/{STACK_NAME}/e2etest/flags")["Parameter"]["Value"])
    assert flags["dark_mode"] is True


# ── Lambda ──

def test_lambda_invoke(stack):
    lam = _client("lambda")
    resp = lam.invoke(FunctionName=f"{STACK_NAME}-e2etest-processor",
                      Payload=json.dumps({"action": "test"}).encode())
    assert resp["StatusCode"] == 200


# ── IAM wiring ──

def test_lambda_role_matches_iam_role(stack):
    lam = _client("lambda")
    iam = _client("iam")

    fn = lam.get_function(FunctionName=f"{STACK_NAME}-e2etest-processor")["Configuration"]
    role = iam.get_role(RoleName=f"{STACK_NAME}-e2etest-role")["Role"]

    assert fn["Role"] == role["Arn"]


# ── Cross-service pipeline ──

def test_e2e_pipeline(stack):
    """S3 upload → SQS queue → read back from S3 → SNS alert."""
    s3 = _client("s3")
    sqs = _client("sqs")
    sns = _client("sns")
    bucket = stack["BucketName"]
    url = stack["QueueUrl"]
    topic_arn = stack["TopicArn"]

    # Upload orders to S3
    for i in range(3):
        order = {"id": f"pipe-{i}", "item": f"widget-{i}", "qty": (i + 1) * 5}
        s3.put_object(Bucket=bucket, Key=f"pipeline/order-{i}.json",
                      Body=json.dumps(order).encode())

    # Queue processing events
    for i in range(3):
        sqs.send_message(QueueUrl=url,
                         MessageBody=json.dumps({"event": "process", "key": f"pipeline/order-{i}.json"}))

    # Consume queue, read each order from S3
    msgs = sqs.receive_message(QueueUrl=url, MaxNumberOfMessages=10, WaitTimeSeconds=1)
    assert len(msgs.get("Messages", [])) == 3

    total_qty = 0
    for m in msgs["Messages"]:
        body = json.loads(m["Body"])
        obj = s3.get_object(Bucket=bucket, Key=body["key"])
        order = json.loads(obj["Body"].read())
        total_qty += order["qty"]
        sqs.delete_message(QueueUrl=url, ReceiptHandle=m["ReceiptHandle"])

    assert total_qty == 5 + 10 + 15  # 30

    # Send completion alert
    resp = sns.publish(TopicArn=topic_arn, Subject="Pipeline Done",
                       Message=json.dumps({"processed": 3, "total_qty": total_qty}))
    assert "MessageId" in resp


# ── Exports ──

def test_cfn_exports_available(stack):
    cfn = _client("cloudformation")
    exports = cfn.list_exports()["Exports"]
    names = {e["Name"]: e["Value"] for e in exports}

    assert f"{STACK_NAME}-bucket" in names
    assert names[f"{STACK_NAME}-bucket"] == stack["BucketName"]
