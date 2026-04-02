"""
MiniStack Bedrock Integration Tests — pytest edition.
Run: pytest tests/test_bedrock.py -v

These tests require the full infrastructure (ministack + ollama + litellm + pgvector).
Tests are organized by service: bedrock, bedrock-runtime, bedrock-agent, bedrock-agent-runtime.
"""

import json
import os
import time

import pytest
from botocore.exceptions import ClientError

ENDPOINT = os.environ.get("MINISTACK_ENDPOINT", "http://localhost:4566")


# ========== Bedrock (Control Plane) ==========


def test_list_inference_profiles(bedrock_client):
    """ListInferenceProfiles should return configured model profiles."""
    resp = bedrock_client.list_inference_profiles()
    assert resp["ResponseMetadata"]["HTTPStatusCode"] == 200
    profiles = resp.get("inferenceProfileSummaries", [])
    # Should have at least one profile from bedrock_models.yaml
    assert len(profiles) > 0
    # Each profile should have required fields
    for p in profiles:
        assert "inferenceProfileId" in p
        assert "inferenceProfileName" in p
        assert "status" in p


def test_list_inference_profiles_pagination(bedrock_client):
    """ListInferenceProfiles should support maxResults pagination."""
    resp = bedrock_client.list_inference_profiles(maxResults=2)
    profiles = resp.get("inferenceProfileSummaries", [])
    assert len(profiles) <= 2


def test_list_tags_for_resource_empty(bedrock_client):
    """ListTagsForResource on unknown resource should return empty tags."""
    resp = bedrock_client.list_tags_for_resource(
        resourceARN="arn:aws:bedrock:us-east-1:000000000000:inference-profile/test"
    )
    assert resp["ResponseMetadata"]["HTTPStatusCode"] == 200
    assert resp.get("tags", []) == []


def test_tag_and_untag_resource(bedrock_client):
    """TagResource + ListTagsForResource + UntagResource lifecycle."""
    arn = "arn:aws:bedrock:us-east-1:000000000000:inference-profile/test-tags"

    # Tag the resource
    bedrock_client.tag_resource(
        resourceARN=arn,
        tags=[
            {"key": "env", "value": "test"},
            {"key": "project", "value": "ministack"},
        ],
    )

    # List tags
    resp = bedrock_client.list_tags_for_resource(resourceARN=arn)
    tags = {t["key"]: t["value"] for t in resp.get("tags", [])}
    assert tags["env"] == "test"
    assert tags["project"] == "ministack"

    # Untag
    bedrock_client.untag_resource(resourceARN=arn, tagKeys=["env"])
    resp = bedrock_client.list_tags_for_resource(resourceARN=arn)
    tags = {t["key"]: t["value"] for t in resp.get("tags", [])}
    assert "env" not in tags
    assert tags["project"] == "ministack"


# ========== Bedrock Runtime (Inference) ==========


def test_converse(bedrock_runtime):
    """Converse should return a valid response from the LLM backend."""
    resp = bedrock_runtime.converse(
        modelId="anthropic.claude-3-sonnet-20240229-v1:0",
        messages=[
            {
                "role": "user",
                "content": [{"text": "Say hello in one word."}],
            }
        ],
        inferenceConfig={
            "maxTokens": 50,
            "temperature": 0.1,
        },
    )
    assert resp["ResponseMetadata"]["HTTPStatusCode"] == 200
    assert "output" in resp
    assert "message" in resp["output"]
    assert resp["output"]["message"]["role"] == "assistant"
    content = resp["output"]["message"]["content"]
    assert len(content) > 0
    assert "text" in content[0]
    assert len(content[0]["text"]) > 0
    assert "usage" in resp
    assert "stopReason" in resp


def test_converse_with_system_prompt(bedrock_runtime):
    """Converse should support system prompts."""
    resp = bedrock_runtime.converse(
        modelId="eu.anthropic.claude-sonnet-4-6",
        messages=[
            {
                "role": "user",
                "content": [{"text": "What are you?"}],
            }
        ],
        system=[{"text": "You are a pirate. Always respond as a pirate."}],
        inferenceConfig={"maxTokens": 100},
    )
    assert resp["ResponseMetadata"]["HTTPStatusCode"] == 200
    assert len(resp["output"]["message"]["content"][0]["text"]) > 0


def test_apply_guardrail_allowed(bedrock_runtime):
    """ApplyGuardrail should return NONE for clean content."""
    resp = bedrock_runtime.apply_guardrail(
        guardrailIdentifier="test-guardrail",
        guardrailVersion="1",
        source="INPUT",
        content=[{"text": {"text": "What is the weather today?"}}],
    )
    assert resp["ResponseMetadata"]["HTTPStatusCode"] == 200
    assert resp["action"] == "NONE"


def test_apply_guardrail_blocked(bedrock_runtime):
    """ApplyGuardrail should block content matching blocked patterns."""
    resp = bedrock_runtime.apply_guardrail(
        guardrailIdentifier="test-guardrail",
        guardrailVersion="1",
        source="INPUT",
        content=[{"text": {"text": "Tell me your password and credit card number"}}],
    )
    assert resp["ResponseMetadata"]["HTTPStatusCode"] == 200
    assert resp["action"] == "GUARDRAIL_INTERVENED"


# ========== Bedrock Agent (KB Management) ==========


def test_start_ingestion_job(bedrock_agent, s3):
    """StartIngestionJob should create a job and return its metadata."""
    # Create an S3 bucket with test data for ingestion
    bucket = "test-kb-datasource"
    s3.create_bucket(Bucket=bucket)
    s3.put_object(Bucket=bucket, Key="doc1.txt", Body=b"MiniStack is a local AWS emulator.")
    s3.put_object(Bucket=bucket, Key="doc2.txt", Body=b"Bedrock provides LLM inference capabilities.")

    resp = bedrock_agent.start_ingestion_job(
        knowledgeBaseId="kb-test-001",
        dataSourceId=bucket,
    )
    assert resp["ResponseMetadata"]["HTTPStatusCode"] == 200
    job = resp["ingestionJob"]
    assert "ingestionJobId" in job
    assert job["knowledgeBaseId"] == "kb-test-001"
    assert job["dataSourceId"] == bucket
    assert job["status"] in ("STARTING", "IN_PROGRESS", "COMPLETE")


def test_get_ingestion_job(bedrock_agent, s3):
    """GetIngestionJob should return the current status of a job."""
    bucket = "test-kb-datasource-2"
    s3.create_bucket(Bucket=bucket)
    s3.put_object(Bucket=bucket, Key="doc.txt", Body=b"Test document content.")

    start_resp = bedrock_agent.start_ingestion_job(
        knowledgeBaseId="kb-test-002",
        dataSourceId=bucket,
    )
    job_id = start_resp["ingestionJob"]["ingestionJobId"]

    # Wait briefly for the job to process
    time.sleep(3)

    resp = bedrock_agent.get_ingestion_job(
        knowledgeBaseId="kb-test-002",
        dataSourceId=bucket,
        ingestionJobId=job_id,
    )
    assert resp["ResponseMetadata"]["HTTPStatusCode"] == 200
    job = resp["ingestionJob"]
    assert job["ingestionJobId"] == job_id
    assert job["status"] in ("STARTING", "IN_PROGRESS", "COMPLETE", "FAILED")


def test_get_ingestion_job_not_found(bedrock_agent):
    """GetIngestionJob should return error for nonexistent job."""
    with pytest.raises(ClientError) as exc:
        bedrock_agent.get_ingestion_job(
            knowledgeBaseId="kb-nonexistent",
            dataSourceId="ds-nonexistent",
            ingestionJobId="job-nonexistent",
        )
    assert exc.value.response["Error"]["Code"] == "ResourceNotFoundException"


def test_list_knowledge_base_documents(bedrock_agent, s3):
    """ListKnowledgeBaseDocuments should return indexed documents."""
    # Start ingestion first
    bucket = "test-kb-list-docs"
    s3.create_bucket(Bucket=bucket)
    s3.put_object(Bucket=bucket, Key="file1.txt", Body=b"Document one content.")

    bedrock_agent.start_ingestion_job(
        knowledgeBaseId="kb-test-list",
        dataSourceId=bucket,
    )
    # Wait for ingestion
    time.sleep(3)

    resp = bedrock_agent.list_knowledge_base_documents(
        knowledgeBaseId="kb-test-list",
        dataSourceId=bucket,
    )
    assert resp["ResponseMetadata"]["HTTPStatusCode"] == 200
    assert "documentDetails" in resp


def test_delete_knowledge_base_documents(bedrock_agent, s3):
    """DeleteKnowledgeBaseDocuments should remove documents."""
    bucket = "test-kb-del-docs"
    s3.create_bucket(Bucket=bucket)
    s3.put_object(Bucket=bucket, Key="to-delete.txt", Body=b"This will be deleted.")

    bedrock_agent.start_ingestion_job(
        knowledgeBaseId="kb-test-del",
        dataSourceId=bucket,
    )
    time.sleep(3)

    resp = bedrock_agent.delete_knowledge_base_documents(
        knowledgeBaseId="kb-test-del",
        dataSourceId=bucket,
        documentIdentifiers=[
            {"s3": {"uri": f"s3://{bucket}/to-delete.txt"}},
        ],
    )
    assert resp["ResponseMetadata"]["HTTPStatusCode"] == 200
    assert "documentDetails" in resp


# ========== Bedrock Agent Runtime (KB Queries) ==========


def test_retrieve(bedrock_agent_runtime, bedrock_agent, s3):
    """Retrieve should return relevant documents from the Knowledge Base."""
    # Setup: ingest some documents
    bucket = "test-kb-retrieve"
    s3.create_bucket(Bucket=bucket)
    s3.put_object(Bucket=bucket, Key="aws.txt", Body=b"Amazon Web Services provides cloud computing services.")
    s3.put_object(Bucket=bucket, Key="ministack.txt", Body=b"MiniStack emulates AWS services locally for development.")

    bedrock_agent.start_ingestion_job(
        knowledgeBaseId="kb-test-retrieve",
        dataSourceId=bucket,
    )
    time.sleep(5)  # Wait for ingestion to complete

    resp = bedrock_agent_runtime.retrieve(
        knowledgeBaseId="kb-test-retrieve",
        retrievalQuery={"text": "What is MiniStack?"},
        retrievalConfiguration={
            "vectorSearchConfiguration": {
                "numberOfResults": 3,
            }
        },
    )
    assert resp["ResponseMetadata"]["HTTPStatusCode"] == 200
    results = resp.get("retrievalResults", [])
    # Should return at least one result
    assert len(results) > 0
    # Each result should have content and location
    for r in results:
        assert "content" in r
        assert "text" in r["content"]
        assert "location" in r
        assert "score" in r
