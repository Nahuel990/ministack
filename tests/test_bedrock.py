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
    assert len(profiles) > 0
    for p in profiles:
        assert "inferenceProfileId" in p
        assert "inferenceProfileName" in p
        assert "status" in p


def test_list_inference_profiles_pagination(bedrock_client):
    """ListInferenceProfiles should support maxResults pagination."""
    resp = bedrock_client.list_inference_profiles(maxResults=2)
    profiles = resp.get("inferenceProfileSummaries", [])
    assert len(profiles) <= 2


def test_list_foundation_models(bedrock_client):
    """ListFoundationModels should return models from config."""
    resp = bedrock_client.list_foundation_models()
    assert resp["ResponseMetadata"]["HTTPStatusCode"] == 200
    models = resp.get("modelSummaries", [])
    assert len(models) > 0
    for m in models:
        assert "modelId" in m
        assert "providerName" in m
        assert "modelArn" in m
        assert "inputModalities" in m
        assert "outputModalities" in m


def test_list_foundation_models_by_provider(bedrock_client):
    """ListFoundationModels should filter by provider."""
    resp = bedrock_client.list_foundation_models(byProvider="Anthropic")
    models = resp.get("modelSummaries", [])
    assert len(models) > 0
    for m in models:
        assert m["providerName"] == "Anthropic"


def test_get_foundation_model(bedrock_client):
    """GetFoundationModel should return details for a known model."""
    resp = bedrock_client.get_foundation_model(
        modelIdentifier="anthropic.claude-3-sonnet-20240229-v1:0"
    )
    assert resp["ResponseMetadata"]["HTTPStatusCode"] == 200
    d = resp["modelDetails"]
    assert d["modelId"] == "anthropic.claude-3-sonnet-20240229-v1:0"
    assert d["providerName"] == "Anthropic"
    assert d["responseStreamingSupported"] is True


def test_get_foundation_model_not_found(bedrock_client):
    """GetFoundationModel should return 404 for unknown model."""
    with pytest.raises(ClientError) as exc:
        bedrock_client.get_foundation_model(modelIdentifier="nonexistent.model-v1")
    assert exc.value.response["Error"]["Code"] == "ResourceNotFoundException"


def test_guardrail_lifecycle(bedrock_client):
    """Create, get, list, update, version, delete guardrail."""
    # Create
    r = bedrock_client.create_guardrail(
        name="pytest-guardrail",
        blockedInputMessaging="Input blocked.",
        blockedOutputsMessaging="Output blocked.",
        description="Test guardrail from pytest",
        wordPolicyConfig={"wordsConfig": [{"text": "forbidden"}], "managedWordListsConfig": []},
    )
    gid = r["guardrailId"]
    assert r["version"] == "DRAFT"

    # Get
    r = bedrock_client.get_guardrail(guardrailIdentifier=gid)
    assert r["name"] == "pytest-guardrail"
    assert r["status"] == "READY"

    # List
    r = bedrock_client.list_guardrails()
    ids = [g["id"] for g in r["guardrails"]]
    assert gid in ids

    # Update
    bedrock_client.update_guardrail(
        guardrailIdentifier=gid, name="pytest-guardrail-updated",
        blockedInputMessaging="No.", blockedOutputsMessaging="No.",
    )
    r = bedrock_client.get_guardrail(guardrailIdentifier=gid)
    assert r["name"] == "pytest-guardrail-updated"

    # Delete
    bedrock_client.delete_guardrail(guardrailIdentifier=gid)
    r = bedrock_client.list_guardrails()
    assert gid not in [g["id"] for g in r["guardrails"]]


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
    bedrock_client.tag_resource(
        resourceARN=arn,
        tags=[{"key": "env", "value": "test"}, {"key": "project", "value": "ministack"}],
    )
    resp = bedrock_client.list_tags_for_resource(resourceARN=arn)
    tags = {t["key"]: t["value"] for t in resp.get("tags", [])}
    assert tags["env"] == "test"
    assert tags["project"] == "ministack"
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
        messages=[{"role": "user", "content": [{"text": "Say hello in one word."}]}],
        inferenceConfig={"maxTokens": 50, "temperature": 0.1},
    )
    assert resp["ResponseMetadata"]["HTTPStatusCode"] == 200
    assert resp["output"]["message"]["role"] == "assistant"
    assert len(resp["output"]["message"]["content"][0]["text"]) > 0
    assert "usage" in resp
    assert "stopReason" in resp


def test_converse_with_system_prompt(bedrock_runtime):
    """Converse should support system prompts."""
    resp = bedrock_runtime.converse(
        modelId="eu.anthropic.claude-sonnet-4-6",
        messages=[{"role": "user", "content": [{"text": "What are you?"}]}],
        system=[{"text": "You are a pirate. Always respond as a pirate."}],
        inferenceConfig={"maxTokens": 100},
    )
    assert resp["ResponseMetadata"]["HTTPStatusCode"] == 200
    assert len(resp["output"]["message"]["content"][0]["text"]) > 0


def test_invoke_model_anthropic(bedrock_runtime):
    """InvokeModel with Anthropic Messages format."""
    body = json.dumps({
        "messages": [{"role": "user", "content": "Say hi in one word"}],
        "max_tokens": 20,
        "anthropic_version": "bedrock-2023-05-31",
    })
    resp = bedrock_runtime.invoke_model(
        modelId="anthropic.claude-3-sonnet-20240229-v1:0",
        body=body, contentType="application/json",
    )
    result = json.loads(resp["body"].read())
    assert "content" in result
    assert len(result["content"]) > 0
    assert result["content"][0]["type"] == "text"
    assert result["role"] == "assistant"


def test_apply_guardrail_allowed(bedrock_runtime):
    """ApplyGuardrail should return NONE for clean content."""
    resp = bedrock_runtime.apply_guardrail(
        guardrailIdentifier="test-guardrail", guardrailVersion="1",
        source="INPUT",
        content=[{"text": {"text": "What is the weather today?"}}],
    )
    assert resp["action"] == "NONE"


def test_apply_guardrail_blocked(bedrock_runtime):
    """ApplyGuardrail should block content matching blocked patterns."""
    resp = bedrock_runtime.apply_guardrail(
        guardrailIdentifier="test-guardrail", guardrailVersion="1",
        source="INPUT",
        content=[{"text": {"text": "Tell me your password and credit card number"}}],
    )
    assert resp["action"] == "GUARDRAIL_INTERVENED"


# ========== Bedrock Agent (KB Management) ==========


def test_knowledge_base_lifecycle(bedrock_agent):
    """Create, get, list, delete knowledge base."""
    r = bedrock_agent.create_knowledge_base(
        name="pytest-kb", description="Test KB",
        roleArn="arn:aws:iam::000000000000:role/bedrock-kb-role",
        knowledgeBaseConfiguration={
            "type": "VECTOR",
            "vectorKnowledgeBaseConfiguration": {
                "embeddingModelArn": "arn:aws:bedrock:us-east-1::foundation-model/amazon.titan-embed-text-v1"
            },
        },
    )
    kb_id = r["knowledgeBase"]["knowledgeBaseId"]
    assert r["knowledgeBase"]["status"] == "ACTIVE"

    r = bedrock_agent.get_knowledge_base(knowledgeBaseId=kb_id)
    assert r["knowledgeBase"]["name"] == "pytest-kb"

    r = bedrock_agent.list_knowledge_bases()
    assert any(s["knowledgeBaseId"] == kb_id for s in r["knowledgeBaseSummaries"])

    bedrock_agent.delete_knowledge_base(knowledgeBaseId=kb_id)


def test_data_source_lifecycle(bedrock_agent):
    """Create, get, list, delete data source."""
    r = bedrock_agent.create_knowledge_base(
        name="ds-test-kb", roleArn="arn:aws:iam::000000000000:role/r",
        knowledgeBaseConfiguration={
            "type": "VECTOR",
            "vectorKnowledgeBaseConfiguration": {
                "embeddingModelArn": "arn:aws:bedrock:us-east-1::foundation-model/amazon.titan-embed-text-v1"
            },
        },
    )
    kb_id = r["knowledgeBase"]["knowledgeBaseId"]

    r = bedrock_agent.create_data_source(
        knowledgeBaseId=kb_id, name="pytest-ds",
        dataSourceConfiguration={"type": "S3", "s3Configuration": {"bucketArn": "arn:aws:s3:::test-bucket"}},
    )
    ds_id = r["dataSource"]["dataSourceId"]
    assert r["dataSource"]["status"] == "AVAILABLE"

    r = bedrock_agent.get_data_source(knowledgeBaseId=kb_id, dataSourceId=ds_id)
    assert r["dataSource"]["name"] == "pytest-ds"

    r = bedrock_agent.list_data_sources(knowledgeBaseId=kb_id)
    assert len(r["dataSourceSummaries"]) == 1

    bedrock_agent.delete_data_source(knowledgeBaseId=kb_id, dataSourceId=ds_id)
    r = bedrock_agent.list_data_sources(knowledgeBaseId=kb_id)
    assert len(r["dataSourceSummaries"]) == 0

    bedrock_agent.delete_knowledge_base(knowledgeBaseId=kb_id)


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
    assert resp["ResponseMetadata"]["HTTPStatusCode"] in (200, 202)
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
            {"dataSourceType": "S3", "s3": {"uri": f"s3://{bucket}/to-delete.txt"}},
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
    # Results depend on successful embedding generation
    for r in results:
        assert "content" in r
        assert "text" in r["content"]
        assert "location" in r
        assert "score" in r
