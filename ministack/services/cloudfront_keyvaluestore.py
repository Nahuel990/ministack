"""
CloudFront KeyValueStore Data-Plane Service Emulator.
JSON REST API — signing name: cloudfront-keyvaluestore.

Paths are under /key-value-stores/{KvsARN}/...

Supports:
  DescribeKeyValueStore, ListKeys, GetKey, PutKey, DeleteKey, UpdateKeys
"""

import copy
import json
import logging
import re
from urllib.parse import unquote

from ministack.core.persistence import load_state
from ministack.core.responses import AccountScopedDict, json_response, new_uuid

logger = logging.getLogger("cloudfront-keyvaluestore")

# ---------------------------------------------------------------------------
# Path regexes — ARN contains a slash (e.g. arn:aws:cloudfront::123:key-value-store/name)
# The store name portion ([a-zA-Z0-9-_]+) never contains a slash, so we anchor
# the /keys boundary against the name segment to avoid ambiguity.
# ---------------------------------------------------------------------------
_KEY_RE = re.compile(r"^/key-value-stores/(arn:.+?/[a-zA-Z0-9_-]+)/keys/(.+)$")
_KEYS_RE = re.compile(r"^/key-value-stores/(arn:.+?/[a-zA-Z0-9_-]+)/keys/?$")
_STORE_RE = re.compile(r"^/key-value-stores/(arn:.+)$")

# ---------------------------------------------------------------------------
# In-memory state — keyed by KVS ARN
# ---------------------------------------------------------------------------
_stores = AccountScopedDict()  # arn -> {"etag": str, "items": {key: value}}


def reset():
    _stores.clear()


def get_state():
    return copy.deepcopy({"stores": _stores})


def restore_state(data):
    if not data:
        return
    _stores.clear()
    for k, v in (data.get("stores") or {}).items():
        _stores[k] = v


try:
    _restored = load_state("cloudfront_keyvaluestore")
    if _restored:
        restore_state(_restored)
except Exception:
    logging.getLogger(__name__).exception("Failed to restore persisted state; continuing with fresh store")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _error(code: str, message: str, status: int) -> tuple:
    body = json.dumps({"Message": message, "__type": code}).encode()
    return status, {"Content-Type": "application/json"}, body


def _get_store(arn: str):
    store = _stores.get(arn)
    if store is None:
        from ministack.services.cloudfront import _kvstores

        kvs = None
        for v in _kvstores.values():
            if v["ARN"] == arn:
                kvs = v
                break
        if kvs is None:
            return None
        store = {"etag": new_uuid(), "items": {}}
        _stores[arn] = store
    return store


def _compute_size(items: dict) -> int:
    total = 0
    for k, v in items.items():
        total += len(k.encode("utf-8")) + len(v.encode("utf-8"))
    return total


# ---------------------------------------------------------------------------
# Request dispatcher
# ---------------------------------------------------------------------------


async def handle_request(method, path, headers, body, query_params):
    logger.debug("%s %s", method, path)

    m = _KEY_RE.match(path)
    if m:
        arn = unquote(m.group(1))
        key = unquote(m.group(2))
        if method == "GET":
            return _get_key(arn, key)
        if method == "PUT":
            return _put_key(arn, key, headers, body)
        if method == "DELETE":
            return _delete_key(arn, key, headers)

    m = _KEYS_RE.match(path)
    if m:
        arn = unquote(m.group(1))
        if method == "GET":
            return _list_keys(arn, query_params)
        if method == "POST":
            return _update_keys(arn, headers, body)

    m = _STORE_RE.match(path)
    if m:
        arn = unquote(m.group(1))
        if method == "GET":
            return _describe_store(arn)

    return _error("InvalidRequestException", f"No route for {method} {path}", 400)


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def _describe_store(arn: str):
    store = _get_store(arn)
    if store is None:
        return _error("ResourceNotFoundException", f"Key value store {arn} was not found.", 404)

    from ministack.services.cloudfront import _kvstores

    kvs_meta = None
    for v in _kvstores.values():
        if v["ARN"] == arn:
            kvs_meta = v
            break

    items = store["items"]
    resp = {
        "KvsARN": arn,
        "ItemCount": len(items),
        "TotalSizeInBytes": _compute_size(items),
        "Status": "READY",
        "Created": 0,
        "LastModified": 0,
    }
    if kvs_meta and kvs_meta.get("LastModifiedTime"):
        resp["Created"] = 0
        resp["LastModified"] = 0

    return 200, {"Content-Type": "application/json", "ETag": store["etag"]}, json.dumps(resp).encode()


def _list_keys(arn: str, query_params):
    store = _get_store(arn)
    if store is None:
        return _error("ResourceNotFoundException", f"Key value store {arn} was not found.", 404)

    max_results = (
        int((query_params.get("MaxResults") or [None])[0] or "10")
        if isinstance(query_params.get("MaxResults"), list)
        else int(query_params.get("MaxResults", "10") or "10")
    )
    next_token = None
    if isinstance(query_params.get("NextToken"), list):
        next_token = query_params["NextToken"][0] if query_params["NextToken"] else None
    elif isinstance(query_params.get("NextToken"), str):
        next_token = query_params["NextToken"] or None

    all_keys = sorted(store["items"].keys())
    start_idx = 0
    if next_token and next_token in all_keys:
        start_idx = all_keys.index(next_token)

    page = all_keys[start_idx : start_idx + max_results]
    items = [{"Key": k, "Value": store["items"][k]} for k in page]

    resp = {"Items": items}
    if start_idx + max_results < len(all_keys):
        resp["NextToken"] = all_keys[start_idx + max_results]

    return json_response(resp)


def _get_key(arn: str, key: str):
    store = _get_store(arn)
    if store is None:
        return _error("ResourceNotFoundException", f"Key value store {arn} was not found.", 404)

    value = store["items"].get(key)
    if value is None:
        return _error("ResourceNotFoundException", f"Key {key} was not found.", 404)

    resp = {
        "Key": key,
        "Value": value,
        "ItemCount": len(store["items"]),
        "TotalSizeInBytes": _compute_size(store["items"]),
    }
    return json_response(resp)


def _put_key(arn: str, key: str, headers, body):
    store = _get_store(arn)
    if store is None:
        return _error("ResourceNotFoundException", f"Key value store {arn} was not found.", 404)

    if_match = headers.get("if-match")
    if not if_match:
        return _error("ValidationException", "If-Match header is required.", 400)
    if if_match != store["etag"]:
        return _error("ConflictException", "The provided If-Match value does not match the current ETag.", 409)

    try:
        data = json.loads(body) if body else {}
    except (json.JSONDecodeError, TypeError):
        return _error("ValidationException", "Invalid JSON body.", 400)

    value = data.get("Value")
    if value is None:
        return _error("ValidationException", "Value is required.", 400)

    store["items"][key] = value
    store["etag"] = new_uuid()

    resp = {
        "ItemCount": len(store["items"]),
        "TotalSizeInBytes": _compute_size(store["items"]),
    }
    return 200, {"Content-Type": "application/json", "ETag": store["etag"]}, json.dumps(resp).encode()


def _delete_key(arn: str, key: str, headers):
    store = _get_store(arn)
    if store is None:
        return _error("ResourceNotFoundException", f"Key value store {arn} was not found.", 404)

    if_match = headers.get("if-match")
    if not if_match:
        return _error("ValidationException", "If-Match header is required.", 400)
    if if_match != store["etag"]:
        return _error("ConflictException", "The provided If-Match value does not match the current ETag.", 409)

    store["items"].pop(key, None)
    store["etag"] = new_uuid()

    resp = {
        "ItemCount": len(store["items"]),
        "TotalSizeInBytes": _compute_size(store["items"]),
    }
    return 200, {"Content-Type": "application/json", "ETag": store["etag"]}, json.dumps(resp).encode()


def _update_keys(arn: str, headers, body):
    store = _get_store(arn)
    if store is None:
        return _error("ResourceNotFoundException", f"Key value store {arn} was not found.", 404)

    if_match = headers.get("if-match")
    if not if_match:
        return _error("ValidationException", "If-Match header is required.", 400)
    if if_match != store["etag"]:
        return _error("ConflictException", "The provided If-Match value does not match the current ETag.", 409)

    try:
        data = json.loads(body) if body else {}
    except (json.JSONDecodeError, TypeError):
        return _error("ValidationException", "Invalid JSON body.", 400)

    puts = data.get("Puts", [])
    deletes = data.get("Deletes", [])

    for item in puts:
        k = item.get("Key")
        v = item.get("Value")
        if k is not None and v is not None:
            store["items"][k] = v

    for item in deletes:
        k = item.get("Key")
        if k is not None:
            store["items"].pop(k, None)

    store["etag"] = new_uuid()

    resp = {
        "ItemCount": len(store["items"]),
        "TotalSizeInBytes": _compute_size(store["items"]),
    }
    return 200, {"Content-Type": "application/json", "ETag": store["etag"]}, json.dumps(resp).encode()
