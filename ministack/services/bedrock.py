"""
AWS Bedrock Control Plane Service Emulator.
REST-based API for managing foundation models, inference profiles, guardrails, and tags.
Supports: ListFoundationModels, GetFoundationModel, ListInferenceProfiles,
          CreateGuardrail, GetGuardrail, ListGuardrails, UpdateGuardrail, DeleteGuardrail,
          ListTagsForResource, TagResource, UntagResource.
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
_guardrails: dict = {}  # guardrail_id -> guardrail metadata
_guardrails_lock = threading.Lock()


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

_RE_LIST_PROFILES = re.compile(r"^/inference-profiles/?$")
_RE_GET_PROFILE = re.compile(r"^/inference-profiles/(.+)$")
_RE_LIST_FOUNDATION = re.compile(r"^/foundation-models/?$")
_RE_GET_FOUNDATION = re.compile(r"^/foundation-models/(.+)$")
_RE_GUARDRAIL_VERSIONS = re.compile(r"^/guardrails/([^/]+)/versions/?$")
_RE_GUARDRAIL_ID = re.compile(r"^/guardrails/([^/]+)$")
_RE_GUARDRAILS = re.compile(r"^/guardrails/?$")


async def handle_request(method, path, headers, body, query_params):
    """Main entry point for Bedrock control plane requests."""
    # ListInferenceProfiles — GET /inference-profiles
    if _RE_LIST_PROFILES.match(path) and method == "GET":
        return _list_inference_profiles(query_params)

    # GetInferenceProfile — GET /inference-profiles/{profileId}
    m = _RE_GET_PROFILE.match(path)
    if m and method == "GET":
        return _get_inference_profile(unquote(m.group(1)))

    # ListFoundationModels — GET /foundation-models
    if _RE_LIST_FOUNDATION.match(path) and method == "GET":
        return _list_foundation_models(query_params)

    # GetFoundationModel — GET /foundation-models/{modelId}
    m = _RE_GET_FOUNDATION.match(path)
    if m and method == "GET":
        return _get_foundation_model(unquote(m.group(1)))

    # CreateGuardrailVersion — POST /guardrails/{id}/versions
    m = _RE_GUARDRAIL_VERSIONS.match(path)
    if m and method == "POST":
        return _create_guardrail_version(m.group(1), body)

    # Guardrails CRUD
    m = _RE_GUARDRAIL_ID.match(path)
    if m:
        guardrail_id = m.group(1)
        if method == "GET":
            return _get_guardrail(guardrail_id, query_params)
        elif method == "PUT":
            return _update_guardrail(guardrail_id, body)
        elif method == "DELETE":
            return _delete_guardrail(guardrail_id)

    if _RE_GUARDRAILS.match(path):
        if method == "POST":
            return _create_guardrail(body)
        elif method == "GET":
            return _list_guardrails(query_params)

    # Tags operations — boto3 sends POST /listTagsForResource, /tagResource, /untagResource
    if method == "POST":
        try:
            data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            data = {}
        resource_arn = data.get("resourceARN", "")
        if path == "/listTagsForResource":
            return _list_tags_for_resource(resource_arn)
        elif path == "/tagResource":
            return _tag_resource(resource_arn, body)
        elif path == "/untagResource":
            return _untag_resource(resource_arn, data)

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


def _untag_resource(resource_arn, data):
    """UntagResource — remove tags from a Bedrock resource."""
    keys = data.get("tagKeys", [])
    if isinstance(keys, str):
        keys = [keys]

    with _tags_lock:
        tags = _tags.get(resource_arn, {})
        for k in keys:
            tags.pop(k, None)

    return json_response({})


# ---------------------------------------------------------------------------
# Foundation Models
# ---------------------------------------------------------------------------

_MODEL_PROVIDERS = {
    "anthropic": "Anthropic",
    "amazon": "Amazon",
    "meta": "Meta",
    "cohere": "Cohere",
    "mistral": "Mistral AI",
    "ai21": "AI21 Labs",
    "stability": "Stability AI",
}


def _build_foundation_models(query_params=None):
    """Build foundation model list from model config."""
    config = _load_models_config()
    models = config.get("models", {})
    by_provider = query_params.get("byProvider", [None])[0] if isinstance(
        query_params.get("byProvider"), list) else query_params.get("byProvider") if query_params else None

    summaries = []
    for model_id in models:
        provider_key = model_id.split(".")[0] if "." in model_id else "unknown"
        provider_name = _MODEL_PROVIDERS.get(provider_key, provider_key.title())

        if by_provider and provider_name.lower() != by_provider.lower():
            continue

        is_embed = "embed" in model_id.lower() or "titan-embed" in model_id.lower()
        summaries.append({
            "modelId": model_id,
            "modelName": model_id.replace(".", " ").replace("-", " ").title(),
            "modelArn": f"arn:aws:bedrock:{REGION}::foundation-model/{model_id}",
            "providerName": provider_name,
            "inputModalities": ["TEXT"],
            "outputModalities": ["EMBEDDING"] if is_embed else ["TEXT"],
            "responseStreamingSupported": not is_embed,
            "customizationsSupported": [],
            "inferenceTypesSupported": ["ON_DEMAND"],
            "modelLifecycle": {"status": "ACTIVE"},
        })
    return summaries


def _get_inference_profile(profile_id: str):
    """GetInferenceProfile — return a specific inference profile."""
    profiles = _build_inference_profiles()
    # Match by full ARN or by model ID
    for p in profiles:
        if p["inferenceProfileId"] == profile_id or p["inferenceProfileName"] == profile_id:
            return json_response({"inferenceProfile": p})
    return error_response_json("ResourceNotFoundException",
                               f"Inference profile {profile_id} not found", 404)


def _list_foundation_models(query_params):
    """ListFoundationModels — returns all configured foundation models."""
    return json_response({"modelSummaries": _build_foundation_models(query_params)})


def _get_foundation_model(model_id: str):
    """GetFoundationModel — returns details for a specific foundation model."""
    config = _load_models_config()
    models = config.get("models", {})
    if model_id not in models:
        return error_response_json("ResourceNotFoundException",
                                   f"Could not find model {model_id}", 404)

    provider_key = model_id.split(".")[0] if "." in model_id else "unknown"
    provider_name = _MODEL_PROVIDERS.get(provider_key, provider_key.title())
    is_embed = "embed" in model_id.lower()

    return json_response({"modelDetails": {
        "modelId": model_id,
        "modelName": model_id.replace(".", " ").replace("-", " ").title(),
        "modelArn": f"arn:aws:bedrock:{REGION}::foundation-model/{model_id}",
        "providerName": provider_name,
        "inputModalities": ["TEXT"],
        "outputModalities": ["EMBEDDING"] if is_embed else ["TEXT"],
        "responseStreamingSupported": not is_embed,
        "customizationsSupported": [],
        "inferenceTypesSupported": ["ON_DEMAND"],
        "modelLifecycle": {"status": "ACTIVE"},
    }})


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------

def _create_guardrail(body: bytes):
    """CreateGuardrail — create a new guardrail configuration."""
    try:
        data = json.loads(body) if body else {}
    except json.JSONDecodeError:
        return error_response_json("ValidationException", "Invalid JSON body", 400)

    name = data.get("name", "")
    if not name:
        return error_response_json("ValidationException", "name is required", 400)

    guardrail_id = new_uuid()[:10]
    guardrail_arn = f"arn:aws:bedrock:{REGION}:{ACCOUNT_ID}:guardrail/{guardrail_id}"
    now = now_iso()

    guardrail = {
        "guardrailId": guardrail_id,
        "guardrailArn": guardrail_arn,
        "name": name,
        "description": data.get("description", ""),
        "version": "DRAFT",
        "status": "READY",
        "blockedInputMessaging": data.get("blockedInputMessaging", "Input blocked by guardrail."),
        "blockedOutputsMessaging": data.get("blockedOutputsMessaging", "Output blocked by guardrail."),
        "contentPolicy": data.get("contentPolicyConfig", {}),
        "topicPolicy": data.get("topicPolicyConfig", {}),
        "wordPolicy": data.get("wordPolicyConfig", {}),
        "sensitiveInformationPolicy": data.get("sensitiveInformationPolicyConfig", {}),
        "contextualGroundingPolicy": data.get("contextualGroundingPolicyConfig", {}),
        "createdAt": now,
        "updatedAt": now,
    }

    with _guardrails_lock:
        _guardrails[guardrail_id] = guardrail

    return json_response({
        "guardrailId": guardrail_id,
        "guardrailArn": guardrail_arn,
        "version": "DRAFT",
        "createdAt": now,
    }, 202)


def _get_guardrail(guardrail_id: str, query_params):
    """GetGuardrail — return guardrail configuration."""
    with _guardrails_lock:
        guardrail = _guardrails.get(guardrail_id)

    if not guardrail:
        return error_response_json("ResourceNotFoundException",
                                   f"Guardrail {guardrail_id} not found", 404)
    return json_response(guardrail)


def _list_guardrails(query_params):
    """ListGuardrails — list all guardrails."""
    max_results = int(query_params.get("maxResults", [100])[0]) if isinstance(
        query_params.get("maxResults"), list) else int(query_params.get("maxResults", 100))

    with _guardrails_lock:
        items = list(_guardrails.values())

    summaries = [{
        "id": g["guardrailId"],
        "arn": g["guardrailArn"],
        "name": g["name"],
        "description": g.get("description", ""),
        "status": g["status"],
        "version": g["version"],
        "createdAt": g["createdAt"],
        "updatedAt": g["updatedAt"],
    } for g in items[:max_results]]

    result = {"guardrails": summaries}
    if len(items) > max_results:
        result["nextToken"] = str(max_results)
    return json_response(result)


def _update_guardrail(guardrail_id: str, body: bytes):
    """UpdateGuardrail — update an existing guardrail."""
    try:
        data = json.loads(body) if body else {}
    except json.JSONDecodeError:
        return error_response_json("ValidationException", "Invalid JSON body", 400)

    with _guardrails_lock:
        guardrail = _guardrails.get(guardrail_id)
        if not guardrail:
            return error_response_json("ResourceNotFoundException",
                                       f"Guardrail {guardrail_id} not found", 404)

        for field in ("name", "description", "blockedInputMessaging", "blockedOutputsMessaging"):
            if field in data:
                guardrail[field] = data[field]
        for policy in ("contentPolicyConfig", "topicPolicyConfig", "wordPolicyConfig",
                       "sensitiveInformationPolicyConfig", "contextualGroundingPolicyConfig"):
            if policy in data:
                guardrail[policy.replace("Config", "")] = data[policy]
        guardrail["updatedAt"] = now_iso()

    return json_response({
        "guardrailId": guardrail_id,
        "guardrailArn": guardrail["guardrailArn"],
        "version": guardrail["version"],
        "updatedAt": guardrail["updatedAt"],
    })


def _delete_guardrail(guardrail_id: str):
    """DeleteGuardrail — delete a guardrail."""
    with _guardrails_lock:
        if guardrail_id not in _guardrails:
            return error_response_json("ResourceNotFoundException",
                                       f"Guardrail {guardrail_id} not found", 404)
        del _guardrails[guardrail_id]
    return json_response({}, 202)


def _create_guardrail_version(guardrail_id: str, body: bytes):
    """CreateGuardrailVersion — create a numbered version from the DRAFT."""
    with _guardrails_lock:
        guardrail = _guardrails.get(guardrail_id)
        if not guardrail:
            return error_response_json("ResourceNotFoundException",
                                       f"Guardrail {guardrail_id} not found", 404)

        # Determine next version number
        existing_versions = [g.get("version", "0") for g in _guardrails.values()
                             if g.get("guardrailId") == guardrail_id and g.get("version", "DRAFT") != "DRAFT"]
        next_ver = str(max([int(v) for v in existing_versions if v.isdigit()] or [0]) + 1)

        # Create versioned copy
        versioned = dict(guardrail)
        versioned["version"] = next_ver
        versioned["updatedAt"] = now_iso()
        _guardrails[f"{guardrail_id}:{next_ver}"] = versioned

    return json_response({
        "guardrailId": guardrail_id,
        "version": next_ver,
    }, 202)


def get_guardrail_config(guardrail_id: str):
    """Public accessor — used by bedrock_runtime to fetch guardrail details."""
    with _guardrails_lock:
        return _guardrails.get(guardrail_id)


def reset():
    """Clear all in-memory state."""
    global _models_config
    with _tags_lock:
        _tags.clear()
    with _guardrails_lock:
        _guardrails.clear()
    _models_config = {}


def get_state():
    """Return serializable state for persistence."""
    return {"tags": dict(_tags), "guardrails": dict(_guardrails)}
