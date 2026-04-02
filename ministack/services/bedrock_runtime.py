"""
AWS Bedrock Runtime Service Emulator.
REST-based API for model inference and guardrails.
Supports: Converse, ApplyGuardrail.

Proxies inference requests to LiteLLM (which routes to Ollama or GitHub Copilot).
Requires LiteLLM to be running — returns ServiceUnavailableException if unavailable.
"""

import json
import logging
import os
import re
import time

from ministack.core.responses import error_response_json, json_response, new_uuid, now_iso

logger = logging.getLogger("bedrock-runtime")

LITELLM_BASE_URL = os.environ.get("LITELLM_BASE_URL", "http://litellm:4000")
ACCOUNT_ID = os.environ.get("MINISTACK_ACCOUNT_ID", "000000000000")
REGION = os.environ.get("MINISTACK_REGION", "us-east-1")

# In-memory guardrail storage
_guardrails: dict = {}  # guardrail_id -> config

# ---------------------------------------------------------------------------
# Path routing patterns
# ---------------------------------------------------------------------------

# POST /model/{modelId}/converse
_RE_CONVERSE = re.compile(r"^/model/([^/]+)/converse$")
# POST /model/{modelId}/invoke
_RE_INVOKE = re.compile(r"^/model/([^/]+)/invoke$")
# POST /guardrail/{guardrailId}/version/{version}/apply
_RE_GUARDRAIL = re.compile(r"^/guardrail/([^/]+)/version/([^/]+)/apply$")


async def handle_request(method, path, headers, body, query_params):
    """Main entry point for Bedrock Runtime requests."""
    # Converse API
    m = _RE_CONVERSE.match(path)
    if m and method == "POST":
        model_id = m.group(1)
        return await _converse(model_id, body)

    # InvokeModel API
    m = _RE_INVOKE.match(path)
    if m and method == "POST":
        model_id = m.group(1)
        return await _invoke_model(model_id, body, headers)

    # ApplyGuardrail API
    m = _RE_GUARDRAIL.match(path)
    if m and method == "POST":
        guardrail_id = m.group(1)
        version = m.group(2)
        return await _apply_guardrail(guardrail_id, version, body)

    return error_response_json("UnrecognizedClientException",
                               f"Unrecognized operation: {method} {path}", 400)


async def _converse(model_id: str, body: bytes):
    """
    Converse API — proxy to LiteLLM for inference.
    Transforms Bedrock Converse request format to OpenAI chat completion format,
    then transforms the response back.
    """
    try:
        data = json.loads(body) if body else {}
    except json.JSONDecodeError:
        return error_response_json("ValidationException", "Invalid JSON body", 400)

    # Import bedrock module to resolve model mapping
    from ministack.services.bedrock import resolve_model
    local_model = resolve_model(model_id)

    # Transform Bedrock messages to OpenAI format
    bedrock_messages = data.get("messages", [])
    openai_messages = []
    for msg in bedrock_messages:
        role = msg.get("role", "user")
        content_blocks = msg.get("content", [])
        text_parts = []
        for block in content_blocks:
            if isinstance(block, dict) and "text" in block:
                text_parts.append(block["text"])
            elif isinstance(block, str):
                text_parts.append(block)
        openai_messages.append({"role": role, "content": " ".join(text_parts) if text_parts else ""})

    # Add system prompt if present
    system_prompts = data.get("system", [])
    if system_prompts:
        system_text = " ".join(
            s.get("text", s) if isinstance(s, dict) else str(s)
            for s in system_prompts
        )
        openai_messages.insert(0, {"role": "system", "content": system_text})

    # Inference config
    inference_config = data.get("inferenceConfig", {})
    temperature = inference_config.get("temperature", 0.7)
    max_tokens = inference_config.get("maxTokens", 1024)
    top_p = inference_config.get("topP", 1.0)

    # Call LiteLLM
    import aiohttp
    litellm_payload = {
        "model": local_model,
        "messages": openai_messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{LITELLM_BASE_URL}/v1/chat/completions",
                json=litellm_payload,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                if resp.status != 200:
                    error_body = await resp.text()
                    logger.error("LiteLLM returned %d: %s", resp.status, error_body)
                    return error_response_json("ModelErrorException",
                                               f"Inference backend error: {error_body}", 500)
                result = await resp.json()
    except (aiohttp.ClientError, OSError) as e:
        logger.error("Failed to connect to LiteLLM at %s: %s", LITELLM_BASE_URL, e)
        return error_response_json("ServiceUnavailableException",
                                   f"Inference backend (LiteLLM) is unavailable at {LITELLM_BASE_URL}: {e}", 503)

    # Transform OpenAI response to Bedrock Converse format
    choice = result.get("choices", [{}])[0]
    message = choice.get("message", {})
    response_text = message.get("content", "")
    finish_reason = choice.get("finish_reason", "end_turn")

    # Map OpenAI finish reasons to Bedrock stop reasons
    stop_reason_map = {
        "stop": "end_turn",
        "length": "max_tokens",
        "content_filter": "content_filtered",
    }
    stop_reason = stop_reason_map.get(finish_reason, "end_turn")

    usage = result.get("usage", {})
    bedrock_response = {
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": response_text}],
            }
        },
        "stopReason": stop_reason,
        "usage": {
            "inputTokens": usage.get("prompt_tokens", 0),
            "outputTokens": usage.get("completion_tokens", 0),
            "totalTokens": usage.get("total_tokens", 0),
        },
        "metrics": {
            "latencyMs": int((time.time() % 1) * 1000),
        },
        "ResponseMetadata": {
            "RequestId": new_uuid(),
            "HTTPStatusCode": 200,
        },
    }

    return json_response(bedrock_response)


async def _invoke_model(model_id: str, body: bytes, headers: dict):
    """
    InvokeModel — legacy model invocation API.
    Supports Anthropic Messages format, Amazon Titan, and generic text completion.
    Proxies to LiteLLM and transforms the response to the provider-specific format.
    """
    try:
        data = json.loads(body) if body else {}
    except json.JSONDecodeError:
        return error_response_json("ValidationException", "Invalid JSON body", 400)

    from ministack.services.bedrock import resolve_model
    local_model = resolve_model(model_id)

    # Build OpenAI-format messages from the provider-specific input
    openai_messages = []
    max_tokens = 1024
    temperature = 0.7

    if "messages" in data:
        # Anthropic Messages API format
        system = data.get("system", "")
        if system:
            openai_messages.append({"role": "system", "content": system})
        for msg in data["messages"]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, list):
                text_parts = [b.get("text", "") for b in content if isinstance(b, dict) and "text" in b]
                content = " ".join(text_parts)
            openai_messages.append({"role": role, "content": content})
        max_tokens = data.get("max_tokens", 1024)
        temperature = data.get("temperature", 0.7)
    elif "inputText" in data:
        # Amazon Titan format
        openai_messages.append({"role": "user", "content": data["inputText"]})
        tc = data.get("textGenerationConfig", {})
        max_tokens = tc.get("maxTokenCount", 1024)
        temperature = tc.get("temperature", 0.7)
    elif "prompt" in data:
        # Generic / Llama / Mistral format
        openai_messages.append({"role": "user", "content": data["prompt"]})
        max_tokens = data.get("max_gen_len", data.get("max_tokens", 1024))
        temperature = data.get("temperature", 0.7)
    else:
        openai_messages.append({"role": "user", "content": json.dumps(data)})

    import aiohttp
    litellm_payload = {
        "model": local_model,
        "messages": openai_messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{LITELLM_BASE_URL}/v1/chat/completions",
                json=litellm_payload,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                if resp.status != 200:
                    error_body = await resp.text()
                    return error_response_json("ModelErrorException",
                                               f"Inference backend error: {error_body}", 500)
                result = await resp.json()
    except (aiohttp.ClientError, OSError) as e:
        return error_response_json("ServiceUnavailableException",
                                   f"Inference backend unavailable: {e}", 503)

    choice = result.get("choices", [{}])[0]
    response_text = choice.get("message", {}).get("content", "")
    finish_reason = choice.get("finish_reason", "stop")
    usage = result.get("usage", {})

    # Format response based on model provider
    if "anthropic" in model_id.lower():
        # Anthropic Messages response format
        response_body = {
            "id": f"msg_{new_uuid()[:24]}",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": response_text}],
            "model": model_id,
            "stop_reason": "end_turn" if finish_reason == "stop" else finish_reason,
            "stop_sequence": None,
            "usage": {
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
            },
        }
    elif "titan" in model_id.lower():
        # Amazon Titan response format
        response_body = {
            "inputTextTokenCount": usage.get("prompt_tokens", 0),
            "results": [{
                "tokenCount": usage.get("completion_tokens", 0),
                "outputText": response_text,
                "completionReason": "FINISH",
            }],
        }
    else:
        # Generic format (Llama, Mistral, etc.)
        response_body = {
            "generation": response_text,
            "prompt_token_count": usage.get("prompt_tokens", 0),
            "generation_token_count": usage.get("completion_tokens", 0),
            "stop_reason": finish_reason,
        }

    resp_body = json.dumps(response_body).encode("utf-8")
    return 200, {
        "Content-Type": "application/json",
        "x-amzn-bedrock-input-token-count": str(usage.get("prompt_tokens", 0)),
        "x-amzn-bedrock-output-token-count": str(usage.get("completion_tokens", 0)),
    }, resp_body


async def _apply_guardrail(guardrail_id: str, version: str, body: bytes):
    """
    ApplyGuardrail — checks content against configured guardrail patterns.
    Uses regex-based content filtering from bedrock_models.yaml config.
    """
    try:
        data = json.loads(body) if body else {}
    except json.JSONDecodeError:
        return error_response_json("ValidationException", "Invalid JSON body", 400)

    source = data.get("source", "INPUT")
    content = data.get("content", [])

    # Load guardrail config — merge YAML defaults with dynamic guardrails
    from ministack.services.bedrock import get_models_config, get_guardrail_config
    config = get_models_config()
    guardrail_config = config.get("guardrails", {})
    blocked_patterns = list(guardrail_config.get("blocked_patterns", []))

    # Also check dynamically created guardrails for word policies
    dynamic = get_guardrail_config(guardrail_id)
    if dynamic:
        word_policy = dynamic.get("wordPolicy", {})
        for w in word_policy.get("wordsConfig", []):
            if isinstance(w, dict) and "text" in w:
                blocked_patterns.append(re.escape(w["text"]))

    # Check each content block against blocked patterns
    assessments = []
    action = "NONE"

    for block in content:
        text = ""
        if isinstance(block, dict):
            if "text" in block:
                text = block["text"].get("text", "") if isinstance(block["text"], dict) else block["text"]
        elif isinstance(block, str):
            text = block

        block_assessment = {
            "contentPolicy": {"filters": []},
            "wordPolicy": {"customWords": [], "managedWordLists": []},
            "sensitiveInformationPolicy": {"piiEntities": [], "regexes": []},
            "topicPolicy": {"topics": []},
        }

        for pattern in blocked_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                action = "GUARDRAIL_INTERVENED"
                block_assessment["wordPolicy"]["customWords"].append({
                    "match": pattern,
                    "action": "BLOCKED",
                })

        assessments.append(block_assessment)

    # Build output — if blocked, replace content
    outputs = []
    if action == "GUARDRAIL_INTERVENED":
        outputs.append({
            "text": "Sorry, the model cannot answer this question. The guardrail blocked the content.",
        })
    else:
        for block in content:
            if isinstance(block, dict) and "text" in block:
                text_val = block["text"].get("text", "") if isinstance(block["text"], dict) else block["text"]
                outputs.append({"text": text_val})
            elif isinstance(block, str):
                outputs.append({"text": block})

    response = {
        "action": action,
        "outputs": outputs,
        "assessments": assessments,
        "usage": {
            "topicPolicyUnits": 1,
            "contentPolicyUnits": 1,
            "wordPolicyUnits": 1,
            "sensitiveInformationPolicyUnits": 1,
            "sensitiveInformationPolicyFreeUnits": 0,
        },
    }

    return json_response(response)


def reset():
    """Clear all in-memory state."""
    _guardrails.clear()
