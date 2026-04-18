"""
Resource Groups Tagging API emulator.
Phase 1: GetResources across S3, Lambda, SQS, SNS, DynamoDB, EventBridge.
"""

import json
import logging
import os

logger = logging.getLogger("tagging")
REGION = os.environ.get("MINISTACK_REGION", "us-east-1")


# ── Tag format normalisation ──────────────────────────────────────────────────

def _normalise_flat(tag_dict):
    """Convert {k: v} flat dict to [{"Key": k, "Value": v}] list."""
    return [{"Key": k, "Value": v} for k, v in (tag_dict or {}).items()]


def _normalise_list(tag_list):
    """Pass-through [{"Key": k, "Value": v}] list (DynamoDB format)."""
    return tag_list or []


# ── Per-service tag collectors ────────────────────────────────────────────────

def _collect_s3():
    import ministack.services.s3 as svc
    for name, tags in svc._bucket_tags.items():
        yield f"arn:aws:s3:::{name}", _normalise_flat(tags)


def _collect_lambda():
    import ministack.services.lambda_svc as svc
    for name, fn in svc._functions.items():
        arn = f"arn:aws:lambda:{REGION}:{_account()}:function:{name}"
        yield arn, _normalise_flat(fn.get("tags", {}))


def _collect_sqs():
    import ministack.services.sqs as svc
    for url, q in svc._queues.items():
        arn = q.get("attributes", {}).get("QueueArn", "")
        if arn:
            yield arn, _normalise_flat(q.get("tags", {}))


def _collect_sns():
    import ministack.services.sns as svc
    for arn, topic in svc._topics.items():
        yield arn, _normalise_flat(topic.get("tags", {}))


def _collect_dynamodb():
    import ministack.services.dynamodb as svc
    for arn, tags in svc._tags.items():
        yield arn, _normalise_list(tags)


def _collect_eventbridge():
    import ministack.services.eventbridge as svc
    for arn, tags in svc._tags.items():
        yield arn, _normalise_flat(tags)


# ResourceTypeFilter prefix -> collector
_COLLECTORS = {
    "s3":       _collect_s3,
    "lambda":   _collect_lambda,
    "sqs":      _collect_sqs,
    "sns":      _collect_sns,
    "dynamodb": _collect_dynamodb,
    "events":   _collect_eventbridge,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _account():
    from ministack.core.responses import get_account_id
    return get_account_id()


def _matches_type_filters(arn, type_filters):
    if not type_filters:
        return True
    for tf in type_filters:
        svc_prefix = tf.split(":")[0]
        if f"::{svc_prefix}:" in arn or f":{svc_prefix}:" in arn:
            return True
    return False


def _matches_tag_filters(tags, tag_filters):
    """AND across filter keys, OR across values within a key."""
    if not tag_filters:
        return True
    tag_map = {t["Key"]: t["Value"] for t in tags}
    for f in tag_filters:
        key = f.get("Key", "")
        values = f.get("Values", [])
        if key not in tag_map:
            return False
        if values and tag_map[key] not in values:
            return False
    return True


# ── Operation handlers ────────────────────────────────────────────────────────

def _get_resources(data):
    tag_filters = data.get("TagFilters", [])
    type_filters = data.get("ResourceTypeFilters", [])

    if type_filters:
        type_prefixes = {tf.split(":")[0] for tf in type_filters}
        active = {k: v for k, v in _COLLECTORS.items() if k in type_prefixes}
        if not active:
            active = _COLLECTORS
    else:
        active = _COLLECTORS

    results = []
    for collector in active.values():
        try:
            for arn, tags in collector():
                if not _matches_type_filters(arn, type_filters):
                    continue
                if not _matches_tag_filters(tags, tag_filters):
                    continue
                results.append({"ResourceARN": arn, "Tags": tags})
        except Exception:
            pass  # service not yet initialised — skip silently

    return 200, {"Content-Type": "application/x-amz-json-1.1"}, json.dumps({
        "ResourceTagMappingList": results,
        "PaginationToken": "",
    }).encode()


# ── Entry point ───────────────────────────────────────────────────────────────

_HANDLERS = {
    "GetResources": _get_resources,
}


async def handle_request(method, path, headers, body, query_params):
    target = headers.get("x-amz-target", "")
    action = target.split(".")[-1] if "." in target else ""

    try:
        data = json.loads(body) if body else {}
    except json.JSONDecodeError:
        return 400, {"Content-Type": "application/x-amz-json-1.1"}, json.dumps({
            "__type": "SerializationException",
            "message": "Invalid JSON",
        }).encode()

    handler = _HANDLERS.get(action)
    if not handler:
        return 400, {"Content-Type": "application/x-amz-json-1.1"}, json.dumps({
            "__type": "InvalidRequestException",
            "message": f"Unknown action: {action}",
        }).encode()

    return handler(data)
