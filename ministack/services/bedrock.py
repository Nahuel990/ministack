"""
AWS Bedrock Control Plane Service Emulator.
REST-based API for managing inference profiles and resource tags.
Supports: ListInferenceProfiles, ListTagsForResource, TagResource, UntagResource.
"""

import json
import logging
import os
import re
import threading
from urllib.parse import unquote

import yaml

from ministack.core.responses import error_response_json, json_response, new_uuid, now_iso

logger = logging.getLogger("bedrock")

ACCOUNT_ID = os.environ.get("MINISTACK_ACCOUNT_ID", "000000000000")
REGION = os.environ.get("MINISTACK_REGION", "us-east-1")
MODELS_CONFIG_PATH = os.environ.get("BEDROCK_MODELS_CONFIG", "config/bedrock_models.yaml")

_tags: dict = {}  # resource_arn -> {key: value}
_tags_lock = threading.Lock()
_models_config: dict = {}


def _load_models_config():
    """Load model mapping config from YAML file."""
    global _models_config
    if _models_config:
        return _models_config
    for path in [MODELS_CONFIG_PATH, "/app/config/bedrock_models.yaml", "config/bedrock_models.yaml"]:
        try:
            with open(path) as f:
                _models_config = yaml.safe_load(f) or {}
                logger.info("Loaded Bedrock models config from %s", path)
                return _models_config
        except FileNotFoundError:
            continue
    logger.warning("Bedrock models config not found, using empty config")
    _models_config = {"models": {}, "default_model": "qwen2.5:3b", "embedding_model": "nomic-embed-text"}
    return _models_config


def get_models_config():
    """Public accessor for other bedrock modules."""
    return _load_models_config()


def resolve_model(bedrock_model_id: str) -> str:
    """Map a Bedrock model ID to a local model name."""
    config = _load_models_config()
    models = config.get("models", {})
    return models.get(bedrock_model_id, config.get("default_model", "qwen2.5:3b"))


def _build_inference_profiles():
    """Build inference profile list from model config."""
    config = _load_models_config()
    models = config.get("models", {})
    profiles = []
    for model_id, local_model in models.items():
        profile_id = f"arn:aws:bedrock:{REGION}:{ACCOUNT_ID}:inference-profile/{model_id}"
        profiles.append({
            "inferenceProfileId": profile_id,
            "inferenceProfileName": model_id,
            "modelSource": {
                "copyFrom": f"arn:aws:bedrock:{REGION}::foundation-model/{model_id}",
            },
            "models": [
                {
                    "modelArn": f"arn:aws:bedrock:{REGION}::foundation-model/{model_id}",
                }
            ],
            "inferenceProfileArn": profile_id,
            "status": "ACTIVE",
            "type": "SYSTEM_DEFINED",
            "createdAt": now_iso(),
            "updatedAt": now_iso(),
            "description": f"MiniStack local profile for {model_id} (backed by {local_model})",
        })
    return profiles


# ---------------------------------------------------------------------------
# Path routing patterns
# ---------------------------------------------------------------------------

# GET /inference-profiles
_RE_LIST_PROFILES = re.compile(r"^/inference-profiles/?$")
# GET/POST/DELETE /tags/{resourceArn}
_RE_TAGS = re.compile(r"^/tags/(.+)$")


async def handle_request(method, path, headers, body, query_params):
    """Main entry point for Bedrock control plane requests."""
    # ListInferenceProfiles
    m = _RE_LIST_PROFILES.match(path)
    if m and method == "GET":
        return _list_inference_profiles(query_params)

    # Tags operations
    m = _RE_TAGS.match(path)
    if m:
        resource_arn = unquote(m.group(1))
        if method == "GET":
            return _list_tags_for_resource(resource_arn)
        elif method == "POST":
            return _tag_resource(resource_arn, body)
        elif method == "DELETE":
            return _untag_resource(resource_arn, query_params)

    return error_response_json("UnrecognizedClientException",
                               f"Unrecognized operation: {method} {path}", 400)


def _list_inference_profiles(query_params):
    """ListInferenceProfiles — returns all configured inference profiles."""
    profiles = _build_inference_profiles()

    max_results = int(query_params.get("maxResults", [100])[0]) if isinstance(
        query_params.get("maxResults"), list) else int(query_params.get("maxResults", 100))
    next_token = query_params.get("nextToken", [None])[0] if isinstance(
        query_params.get("nextToken"), list) else query_params.get("nextToken")

    # Simple pagination
    start = 0
    if next_token:
        try:
            start = int(next_token)
        except ValueError:
            start = 0

    page = profiles[start:start + max_results]
    result = {"inferenceProfileSummaries": page}
    if start + max_results < len(profiles):
        result["nextToken"] = str(start + max_results)

    return json_response(result)


def _list_tags_for_resource(resource_arn):
    """ListTagsForResource — return tags for a Bedrock resource."""
    with _tags_lock:
        tags = _tags.get(resource_arn, {})
    tag_list = [{"key": k, "value": v} for k, v in tags.items()]
    return json_response({"tags": tag_list})


def _tag_resource(resource_arn, body):
    """TagResource — add/update tags on a Bedrock resource."""
    try:
        data = json.loads(body) if body else {}
    except json.JSONDecodeError:
        return error_response_json("ValidationException", "Invalid JSON body", 400)

    new_tags = data.get("tags", [])
    with _tags_lock:
        existing = _tags.setdefault(resource_arn, {})
        for tag in new_tags:
            existing[tag["key"]] = tag["value"]

    return json_response({})


def _untag_resource(resource_arn, query_params):
    """UntagResource — remove tags from a Bedrock resource."""
    keys = query_params.get("tagKeys", [])
    if isinstance(keys, str):
        keys = [keys]

    with _tags_lock:
        tags = _tags.get(resource_arn, {})
        for k in keys:
            tags.pop(k, None)

    return json_response({})


def reset():
    """Clear all in-memory state."""
    global _models_config
    with _tags_lock:
        _tags.clear()
    _models_config = {}


def get_state():
    """Return serializable state for persistence."""
    return {"tags": dict(_tags)}
