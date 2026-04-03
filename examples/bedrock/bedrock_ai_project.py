#!/usr/bin/env python3
"""
AI/ML demo project with AWS Bedrock — works on both MiniStack and real AWS.

Simulates a complete RAG pipeline:
  1. Discover available foundation models
  2. Create a security guardrail
  3. Upload documents to S3
  4. Create a Knowledge Base + Data Source
  5. Ingest documents (S3 -> pgvector embeddings)
  6. Semantic search (Retrieve)
  7. RAG — Retrieve & Generate
  8. Direct conversation (Converse + InvokeModel)
  9. Token counting
  10. Guardrail testing (allowed + blocked content)
  11. Cleanup

Usage:
    # Against MiniStack (default)
    python examples/bedrock/bedrock_ai_project.py

    # Against real AWS
    python examples/bedrock/bedrock_ai_project.py --aws

    # With a specific model
    python examples/bedrock/bedrock_ai_project.py --model eu.anthropic.claude-sonnet-4-6

Requirements:
    pip install boto3
    docker compose up -d   (for MiniStack)
"""

import argparse
import io
import json
import sys
import time
import uuid

import boto3
from botocore.config import Config

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

MINISTACK_ENDPOINT = "http://localhost:4566"
DEFAULT_REGION = "us-east-1"
DEFAULT_MODEL = "anthropic.claude-3-sonnet-20240229-v1:0"

# Knowledge base documents for our fictional project
KNOWLEDGE_DOCS = {
    "docs/architecture.txt": (
        "Our application uses a microservices architecture with 3 main components: "
        "an API Gateway (FastAPI), a processing service (Celery + Redis), and a PostgreSQL "
        "database with pgvector for semantic search. Everything is deployed on AWS ECS Fargate."
    ),
    "docs/deployment.txt": (
        "Deployment is managed via Terraform. The dev/staging/prod environments are separated by "
        "distinct AWS accounts. CI/CD uses GitHub Actions with lint, test, Docker build, ECR push, "
        "and ECS deploy stages. Rollback is automatic if the health check fails."
    ),
    "docs/security.txt": (
        "Security is enforced by AWS WAF on the API Gateway, Bedrock guardrails to filter "
        "sensitive content (PII, passwords), and KMS encryption for secrets. "
        "Authentication uses Cognito with mandatory MFA for admins."
    ),
    "docs/monitoring.txt": (
        "Monitoring relies on CloudWatch Metrics and Logs, with alarms on P99 latency, "
        "5xx error rate, and CPU/memory utilization of ECS containers. "
        "A Grafana dashboard aggregates business metrics (requests/sec, LLM cost, tokens consumed)."
    ),
    "docs/llm_usage.txt": (
        "The application uses Amazon Bedrock for LLM inference. The primary model is "
        "Claude 3 Sonnet for response generation. Embeddings are computed with "
        "Amazon Titan Embed Text v1. A Redis caching layer reduces API calls by 40%."
    ),
}


def make_client(service, endpoint=None, region=DEFAULT_REGION):
    """Create a boto3 client configured for MiniStack or AWS."""
    kwargs = dict(
        region_name=region,
        config=Config(
            retries={"max_attempts": 2},
            read_timeout=300,
            connect_timeout=10,
        ),
    )
    if endpoint:
        kwargs["endpoint_url"] = endpoint
        kwargs["aws_access_key_id"] = "test"
        kwargs["aws_secret_access_key"] = "test"
    return boto3.client(service, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline steps
# ─────────────────────────────────────────────────────────────────────────────

def step_1_discover_models(bedrock):
    """Step 1 — Discover available Bedrock models."""
    print("\n" + "=" * 70)
    print("STEP 1 — Discover Bedrock Foundation Models")
    print("=" * 70)

    resp = bedrock.list_foundation_models()
    models = resp["modelSummaries"]
    print(f"  {len(models)} models available:")
    for m in models[:8]:
        stream = "Y" if m.get("responseStreamingSupported") else "N"
        print(f"    - {m['modelId']:50s} ({m['providerName']}) stream={stream}")

    if len(models) > 8:
        print(f"    ... and {len(models) - 8} more")

    # Get details for a specific model
    detail = bedrock.get_foundation_model(
        modelIdentifier="anthropic.claude-3-sonnet-20240229-v1:0"
    )["modelDetails"]
    print(f"\n  Claude 3 Sonnet details:")
    print(f"    Provider  : {detail['providerName']}")
    print(f"    Input     : {detail['inputModalities']}")
    print(f"    Output    : {detail['outputModalities']}")
    print(f"    Streaming : {detail['responseStreamingSupported']}")

    return models


def step_2_create_guardrail(bedrock):
    """Step 2 — Create a security guardrail."""
    print("\n" + "=" * 70)
    print("STEP 2 — Create Security Guardrail")
    print("=" * 70)

    resp = bedrock.create_guardrail(
        name=f"project-guardrail-{uuid.uuid4().hex[:6]}",
        description="Filters sensitive data (PII, credentials) from conversations",
        blockedInputMessaging="Sorry, your message contains sensitive information that cannot be processed.",
        blockedOutputsMessaging="The response was blocked because it contained sensitive information.",
        wordPolicyConfig={
            "wordsConfig": [
                {"text": "password"},
                {"text": "secret_key"},
                {"text": "api_key"},
            ],
            "managedWordListsConfig": [],
        },
    )
    guardrail_id = resp["guardrailId"]
    print(f"  Guardrail created : {guardrail_id}")
    print(f"  Version           : {resp['version']}")

    # Verify
    detail = bedrock.get_guardrail(guardrailIdentifier=guardrail_id)
    print(f"  Status            : {detail['status']}")
    print(f"  Description       : {detail['description']}")

    return guardrail_id


def step_3_upload_documents(s3, bucket):
    """Step 3 — Upload documents to S3."""
    print("\n" + "=" * 70)
    print("STEP 3 — Upload Documents to S3")
    print("=" * 70)

    try:
        s3.create_bucket(Bucket=bucket)
    except s3.exceptions.BucketAlreadyOwnedByYou:
        pass
    print(f"  Bucket: s3://{bucket}/")

    for key, content in KNOWLEDGE_DOCS.items():
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=content.encode("utf-8"),
            ContentType="text/plain",
        )
        print(f"  -> {key} ({len(content)} chars)")

    resp = s3.list_objects_v2(Bucket=bucket, Prefix="docs/")
    print(f"\n  {resp['KeyCount']} objects in s3://{bucket}/docs/")

    # upload_fileobj for an additional file
    summary = "Project summary: AI application with RAG, guardrails, and CloudWatch monitoring."
    s3.upload_fileobj(
        io.BytesIO(summary.encode("utf-8")),
        bucket,
        "docs/summary.txt",
    )
    print(f"  -> docs/summary.txt (via upload_fileobj)")

    return bucket


def step_4_create_knowledge_base(agent, bucket):
    """Step 4 — Create Knowledge Base and Data Source."""
    print("\n" + "=" * 70)
    print("STEP 4 — Create Knowledge Base")
    print("=" * 70)

    resp = agent.create_knowledge_base(
        name="project-knowledge-base",
        description="AI project knowledge base — architecture, deployment, security",
        roleArn="arn:aws:iam::000000000000:role/bedrock-kb-role",
        knowledgeBaseConfiguration={
            "type": "VECTOR",
            "vectorKnowledgeBaseConfiguration": {
                "embeddingModelArn": "arn:aws:bedrock:us-east-1::foundation-model/amazon.titan-embed-text-v1",
            },
        },
    )
    kb_id = resp["knowledgeBase"]["knowledgeBaseId"]
    print(f"  Knowledge Base : {kb_id}")
    print(f"  Status         : {resp['knowledgeBase']['status']}")

    resp = agent.create_data_source(
        knowledgeBaseId=kb_id,
        name="project-s3-source",
        description=f"Documents from s3://{bucket}/docs/",
        dataSourceConfiguration={
            "type": "S3",
            "s3Configuration": {"bucketArn": f"arn:aws:s3:::{bucket}"},
        },
    )
    ds_id = resp["dataSource"]["dataSourceId"]
    print(f"  Data Source    : {ds_id}")
    print(f"  Status         : {resp['dataSource']['status']}")

    return kb_id, ds_id


def step_5_ingest_documents(agent, kb_id, ds_id, bucket):
    """Step 5 — Ingest documents into pgvector."""
    print("\n" + "=" * 70)
    print("STEP 5 — Ingest S3 -> pgvector (embeddings)")
    print("=" * 70)

    resp = agent.start_ingestion_job(
        knowledgeBaseId=kb_id,
        dataSourceId=ds_id,
        description="Initial project document ingestion",
    )
    job_id = resp["ingestionJob"]["ingestionJobId"]
    print(f"  Job started : {job_id}")
    print(f"  Status      : {resp['ingestionJob']['status']}")

    print("  Waiting for ingestion", end="", flush=True)
    for _ in range(15):
        time.sleep(2)
        print(".", end="", flush=True)
        resp = agent.get_ingestion_job(
            knowledgeBaseId=kb_id,
            dataSourceId=ds_id,
            ingestionJobId=job_id,
        )
        status = resp["ingestionJob"]["status"]
        if status in ("COMPLETE", "FAILED", "STOPPED"):
            break
    print()

    stats = resp["ingestionJob"].get("statistics", {})
    print(f"  Final status : {status}")
    print(f"  Scanned      : {stats.get('numberOfDocumentsScanned', '?')}")
    print(f"  Indexed      : {stats.get('numberOfNewDocumentsIndexed', '?')}")
    print(f"  Failed       : {stats.get('numberOfDocumentsFailed', '?')}")

    resp = agent.list_ingestion_jobs(
        knowledgeBaseId=kb_id,
        dataSourceId=ds_id,
    )
    print(f"  Total jobs   : {len(resp.get('ingestionJobSummaries', []))}")

    return job_id


def step_6_semantic_search(agent_runtime, kb_id):
    """Step 6 — Semantic search over the Knowledge Base."""
    print("\n" + "=" * 70)
    print("STEP 6 — Semantic Search (Retrieve)")
    print("=" * 70)

    queries = [
        "How is the application deployed?",
        "What are the security mechanisms?",
        "Which LLM model is used?",
    ]

    for query in queries:
        print(f"\n  Q: \"{query}\"")
        resp = agent_runtime.retrieve(
            knowledgeBaseId=kb_id,
            retrievalQuery={"text": query},
            retrievalConfiguration={
                "vectorSearchConfiguration": {"numberOfResults": 2},
            },
        )
        results = resp.get("retrievalResults", [])
        if results:
            for i, r in enumerate(results):
                score = r.get("score", 0)
                text = r["content"]["text"][:100].replace("\n", " ")
                uri = r.get("location", {}).get("s3Location", {}).get("uri", "?")
                print(f"    [{i+1}] score={score:.3f}  {uri}")
                print(f"        {text}...")
        else:
            print("    (no results — embeddings may not have been generated)")


def step_7_rag(agent_runtime, kb_id, model_id):
    """Step 7 — RAG: Retrieve & Generate."""
    print("\n" + "=" * 70)
    print("STEP 7 — RAG (Retrieve & Generate)")
    print("=" * 70)

    question = "Describe the technical architecture and the deployment strategy of the project."
    print(f"  Q: \"{question}\"")

    resp = agent_runtime.retrieve_and_generate(
        input={"text": question},
        retrieveAndGenerateConfiguration={
            "type": "KNOWLEDGE_BASE",
            "knowledgeBaseConfiguration": {
                "knowledgeBaseId": kb_id,
                "modelArn": f"arn:aws:bedrock:us-east-1::foundation-model/{model_id}",
                "retrievalConfiguration": {
                    "vectorSearchConfiguration": {"numberOfResults": 3},
                },
            },
        },
    )

    answer = resp.get("output", {}).get("text", "(no response)")
    citations = resp.get("citations", [])
    print(f"\n  Response ({len(answer)} chars):")
    for line in answer[:500].split(". "):
        print(f"    {line.strip()}.")
    if len(answer) > 500:
        print(f"    ...")
    print(f"\n  {len(citations)} citation(s)")


def step_8_conversation(bedrock_runtime, model_id):
    """Step 8 — Direct conversation with the LLM."""
    print("\n" + "=" * 70)
    print("STEP 8 — Direct Conversation (Converse + InvokeModel)")
    print("=" * 70)

    # --- Converse API (modern format) ---
    print("\n  [Converse API]")
    resp = bedrock_runtime.converse(
        modelId=model_id,
        messages=[
            {"role": "user", "content": [{"text": "Explain in 2 sentences what Amazon Bedrock is."}]},
        ],
        system=[{"text": "You are an AWS expert. Answer concisely."}],
        inferenceConfig={"maxTokens": 200, "temperature": 0.3},
    )
    text = resp["output"]["message"]["content"][0]["text"]
    usage = resp.get("usage", {})
    print(f"  Response: {text[:200]}")
    print(f"  Tokens  : in={usage.get('inputTokens', '?')} out={usage.get('outputTokens', '?')}")
    print(f"  Stop    : {resp.get('stopReason', '?')}")

    # --- Multi-turn ---
    print("\n  [Multi-turn]")
    resp = bedrock_runtime.converse(
        modelId=model_id,
        messages=[
            {"role": "user", "content": [{"text": "My name is Alice."}]},
            {"role": "assistant", "content": [{"text": "Nice to meet you, Alice! How can I help you?"}]},
            {"role": "user", "content": [{"text": "What is my name?"}]},
        ],
        inferenceConfig={"maxTokens": 50, "temperature": 0.0},
    )
    print(f"  Response: {resp['output']['message']['content'][0]['text']}")

    # --- InvokeModel (Anthropic Messages format) ---
    print("\n  [InvokeModel — Anthropic format]")
    body = json.dumps({
        "messages": [{"role": "user", "content": "Say hello in one sentence."}],
        "max_tokens": 50,
        "anthropic_version": "bedrock-2023-05-31",
    })
    resp = bedrock_runtime.invoke_model(
        modelId=model_id, body=body, contentType="application/json",
    )
    result = json.loads(resp["body"].read())
    print(f"  Response: {result['content'][0]['text']}")
    print(f"  Tokens  : in={result['usage']['input_tokens']} out={result['usage']['output_tokens']}")


def step_9_count_tokens(bedrock_runtime, model_id):
    """Step 9 — Token counting."""
    print("\n" + "=" * 70)
    print("STEP 9 — Token Counting (CountTokens)")
    print("=" * 70)

    texts = [
        "Hello",
        "Amazon Bedrock is an AWS service for LLM inference.",
        "Lorem ipsum " * 100,
    ]

    for text in texts:
        resp = bedrock_runtime.count_tokens(
            modelId=model_id,
            input={"converse": {
                "messages": [{"role": "user", "content": [{"text": text}]}],
            }},
        )
        preview = text[:60].replace("\n", " ")
        print(f"  \"{preview}{'...' if len(text) > 60 else ''}\"  ->  {resp['inputTokens']} tokens")


def step_10_test_guardrail(bedrock_runtime, guardrail_id):
    """Step 10 — Test the guardrail."""
    print("\n" + "=" * 70)
    print("STEP 10 — Guardrail Testing (ApplyGuardrail)")
    print("=" * 70)

    test_cases = [
        ("What is the weather today?", "NONE"),
        ("My password is abc123 and my secret_key is XYZ", "GUARDRAIL_INTERVENED"),
        ("Here is my api_key: sk-1234567890", "GUARDRAIL_INTERVENED"),
        ("Explain the project architecture", "NONE"),
    ]

    for text, expected in test_cases:
        resp = bedrock_runtime.apply_guardrail(
            guardrailIdentifier=guardrail_id,
            guardrailVersion="DRAFT",
            source="INPUT",
            content=[{"text": {"text": text}}],
        )
        action = resp["action"]
        icon = "PASS" if action == expected else "FAIL"
        print(f"  [{icon}] \"{text[:60]}\"")
        print(f"     Action: {action}  (expected: {expected})")


def step_11_cleanup(s3, agent, bedrock, bucket, kb_id, ds_id, guardrail_id):
    """Step 11 — Cleanup."""
    print("\n" + "=" * 70)
    print("STEP 11 — Cleanup")
    print("=" * 70)

    try:
        resp = s3.list_objects_v2(Bucket=bucket)
        objs = resp.get("Contents", [])
        if objs:
            s3.delete_objects(
                Bucket=bucket,
                Delete={"Objects": [{"Key": o["Key"]} for o in objs], "Quiet": True},
            )
        s3.delete_bucket(Bucket=bucket)
        print(f"  OK  Bucket s3://{bucket}/ deleted ({len(objs)} objects)")
    except Exception as e:
        print(f"  WARN  Bucket: {e}")

    try:
        agent.delete_data_source(knowledgeBaseId=kb_id, dataSourceId=ds_id)
        print(f"  OK  Data Source {ds_id} deleted")
    except Exception:
        pass
    try:
        agent.delete_knowledge_base(knowledgeBaseId=kb_id)
        print(f"  OK  Knowledge Base {kb_id} deleted")
    except Exception:
        pass

    try:
        bedrock.delete_guardrail(guardrailIdentifier=guardrail_id)
        print(f"  OK  Guardrail {guardrail_id} deleted")
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="AI/ML project with AWS Bedrock — MiniStack demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--aws", action="store_true", help="Target real AWS instead of MiniStack")
    parser.add_argument("--endpoint", default=None, help=f"Custom endpoint (default: {MINISTACK_ENDPOINT})")
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model ID (default: {DEFAULT_MODEL})")
    parser.add_argument("--no-cleanup", action="store_true", help="Skip cleanup at the end")
    args = parser.parse_args()

    endpoint = None if args.aws else (args.endpoint or MINISTACK_ENDPOINT)
    target = "AWS" if args.aws else f"MiniStack ({endpoint})"

    print("=" * 70)
    print("  AI/ML Project with AWS Bedrock — Full RAG Pipeline")
    print("=" * 70)
    print(f"  Target : {target}")
    print(f"  Region : {args.region}")
    print(f"  Model  : {args.model}")
    print("=" * 70)

    bedrock = make_client("bedrock", endpoint, args.region)
    bedrock_runtime = make_client("bedrock-runtime", endpoint, args.region)
    bedrock_agent = make_client("bedrock-agent", endpoint, args.region)
    bedrock_agent_runtime = make_client("bedrock-agent-runtime", endpoint, args.region)
    s3 = make_client("s3", endpoint, args.region)

    bucket = f"ai-project-{uuid.uuid4().hex[:8]}"
    kb_id = ds_id = guardrail_id = None

    try:
        step_1_discover_models(bedrock)
        guardrail_id = step_2_create_guardrail(bedrock)
        step_3_upload_documents(s3, bucket)
        kb_id, ds_id = step_4_create_knowledge_base(bedrock_agent, bucket)
        step_5_ingest_documents(bedrock_agent, kb_id, ds_id, bucket)
        step_6_semantic_search(bedrock_agent_runtime, kb_id)
        step_7_rag(bedrock_agent_runtime, kb_id, args.model)
        step_8_conversation(bedrock_runtime, args.model)
        step_9_count_tokens(bedrock_runtime, args.model)
        step_10_test_guardrail(bedrock_runtime, guardrail_id)

        print("\n" + "=" * 70)
        print("PIPELINE COMPLETE — All steps succeeded!")
        print("=" * 70)

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

    finally:
        if not args.no_cleanup:
            step_11_cleanup(s3, bedrock_agent, bedrock, bucket, kb_id, ds_id, guardrail_id)


if __name__ == "__main__":
    main()
