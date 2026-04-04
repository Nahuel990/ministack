"""
AWS AppSync Service Emulator.

GraphQL API management service — REST/JSON protocol via /v1/apis/* paths.

Supports:
  GraphQL APIs:  CreateGraphQLApi, GetGraphQLApi, ListGraphQLApis,
                 UpdateGraphQLApi, DeleteGraphQLApi
  API Keys:      CreateApiKey, ListApiKeys, DeleteApiKey
  Data Sources:  CreateDataSource, GetDataSource, ListDataSources, DeleteDataSource
  Resolvers:     CreateResolver, GetResolver, ListResolvers, DeleteResolver
  Types:         CreateType, ListTypes, GetType
  Tags:          TagResource, UntagResource, ListTagsForResource

Wire protocol:
  REST/JSON — path-based routing under /v1/apis.
  Credential scope: appsync
"""

import copy
import json
import logging
import os
import re
import time

from ministack.core.persistence import PERSIST_STATE, load_state
from ministack.core.responses import error_response_json, json_response, new_uuid

logger = logging.getLogger("appsync")

ACCOUNT_ID = os.environ.get("MINISTACK_ACCOUNT_ID", "000000000000")
REGION = os.environ.get("MINISTACK_REGION", "us-east-1")

# ---------------------------------------------------------------------------
# In-memory state
# ---------------------------------------------------------------------------

_apis: dict = {}            # apiId -> api record
_api_keys: dict = {}        # apiId -> {keyId -> key record}
_data_sources: dict = {}    # apiId -> {name -> data source record}
_resolvers: dict = {}       # apiId -> {typeName -> {fieldName -> resolver record}}
_types: dict = {}           # apiId -> {typeName -> type record}
_tags: dict = {}            # resource_arn -> {key: value}

# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _load_persisted():
    if not PERSIST_STATE:
        return
    data = load_state("appsync")
    if data:
        restore_state(data)
        logger.info("Loaded persisted state for appsync")

_load_persisted()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now():
    return int(time.time())


def _api_arn(api_id):
    return f"arn:aws:appsync:{REGION}:{ACCOUNT_ID}:apis/{api_id}"


def _json(status, body):
    return json_response(body, status)


# ---------------------------------------------------------------------------
# GraphQL APIs
# ---------------------------------------------------------------------------

def _create_graphql_api(body):
    api_id = new_uuid()[:8]
    name = body.get("name", "")
    auth_type = body.get("authenticationType", "API_KEY")
    additional_auth = body.get("additionalAuthenticationProviders", [])
    log_config = body.get("logConfig")
    user_pool_config = body.get("userPoolConfig")
    openid_config = body.get("openIDConnectConfig")
    xray = body.get("xrayEnabled", False)
    tags = body.get("tags", {})
    lambda_auth = body.get("lambdaAuthorizerConfig")

    arn = _api_arn(api_id)
    now = _now()

    record = {
        "apiId": api_id,
        "name": name,
        "authenticationType": auth_type,
        "arn": arn,
        "uris": {
            "GRAPHQL": f"https://{api_id}.appsync-api.{REGION}.amazonaws.com/graphql",
            "REALTIME": f"wss://{api_id}.appsync-realtime-api.{REGION}.amazonaws.com/graphql",
        },
        "additionalAuthenticationProviders": additional_auth,
        "xrayEnabled": xray,
        "wafWebAclArn": body.get("wafWebAclArn"),
        "createdAt": now,
        "lastUpdatedAt": now,
    }
    if log_config:
        record["logConfig"] = log_config
    if user_pool_config:
        record["userPoolConfig"] = user_pool_config
    if openid_config:
        record["openIDConnectConfig"] = openid_config
    if lambda_auth:
        record["lambdaAuthorizerConfig"] = lambda_auth

    _apis[api_id] = record
    _api_keys[api_id] = {}
    _data_sources[api_id] = {}
    _resolvers[api_id] = {}
    _types[api_id] = {}

    if tags:
        _tags[arn] = tags

    return _json(200, {"graphqlApi": record})


def _get_graphql_api(api_id):
    api = _apis.get(api_id)
    if not api:
        return error_response_json("NotFoundException", f"GraphQL API {api_id} not found", 404)
    return _json(200, {"graphqlApi": api})


def _list_graphql_apis(query_params):
    apis = list(_apis.values())
    return _json(200, {"graphqlApis": apis})


def _update_graphql_api(api_id, body):
    api = _apis.get(api_id)
    if not api:
        return error_response_json("NotFoundException", f"GraphQL API {api_id} not found", 404)

    if "name" in body:
        api["name"] = body["name"]
    if "authenticationType" in body:
        api["authenticationType"] = body["authenticationType"]
    if "additionalAuthenticationProviders" in body:
        api["additionalAuthenticationProviders"] = body["additionalAuthenticationProviders"]
    if "logConfig" in body:
        api["logConfig"] = body["logConfig"]
    if "userPoolConfig" in body:
        api["userPoolConfig"] = body["userPoolConfig"]
    if "openIDConnectConfig" in body:
        api["openIDConnectConfig"] = body["openIDConnectConfig"]
    if "xrayEnabled" in body:
        api["xrayEnabled"] = body["xrayEnabled"]
    if "lambdaAuthorizerConfig" in body:
        api["lambdaAuthorizerConfig"] = body["lambdaAuthorizerConfig"]

    api["lastUpdatedAt"] = _now()
    return _json(200, {"graphqlApi": api})


def _delete_graphql_api(api_id):
    if api_id not in _apis:
        return error_response_json("NotFoundException", f"GraphQL API {api_id} not found", 404)

    arn = _apis[api_id]["arn"]
    del _apis[api_id]
    _api_keys.pop(api_id, None)
    _data_sources.pop(api_id, None)
    _resolvers.pop(api_id, None)
    _types.pop(api_id, None)
    _tags.pop(arn, None)

    return _json(200, {})


# ---------------------------------------------------------------------------
# API Keys
# ---------------------------------------------------------------------------

def _create_api_key(api_id, body):
    if api_id not in _apis:
        return error_response_json("NotFoundException", f"GraphQL API {api_id} not found", 404)

    key_id = "da2-" + new_uuid()[:26]
    now = _now()
    expires = body.get("expires", now + 604800)  # default 7 days
    description = body.get("description", "")

    record = {
        "id": key_id,
        "description": description,
        "expires": expires,
        "createdAt": now,
        "lastUpdatedAt": now,
        "deletes": expires + 5184000,  # 60 days after expiry
    }

    _api_keys.setdefault(api_id, {})[key_id] = record
    return _json(200, {"apiKey": record})


def _list_api_keys(api_id):
    if api_id not in _apis:
        return error_response_json("NotFoundException", f"GraphQL API {api_id} not found", 404)

    keys = list(_api_keys.get(api_id, {}).values())
    return _json(200, {"apiKeys": keys})


def _delete_api_key(api_id, key_id):
    if api_id not in _apis:
        return error_response_json("NotFoundException", f"GraphQL API {api_id} not found", 404)

    keys = _api_keys.get(api_id, {})
    if key_id not in keys:
        return error_response_json("NotFoundException", f"API key {key_id} not found", 404)

    del keys[key_id]
    return _json(200, {})


# ---------------------------------------------------------------------------
# Data Sources
# ---------------------------------------------------------------------------

def _create_data_source(api_id, body):
    if api_id not in _apis:
        return error_response_json("NotFoundException", f"GraphQL API {api_id} not found", 404)

    name = body.get("name", "")
    ds_type = body.get("type", "NONE")
    description = body.get("description", "")
    service_role_arn = body.get("serviceRoleArn", "")

    arn = f"{_apis[api_id]['arn']}/datasources/{name}"

    record = {
        "dataSourceArn": arn,
        "name": name,
        "type": ds_type,
        "description": description,
        "serviceRoleArn": service_role_arn,
        "createdAt": _now(),
        "lastUpdatedAt": _now(),
    }

    if ds_type == "AMAZON_DYNAMODB":
        record["dynamodbConfig"] = body.get("dynamodbConfig", {})
    elif ds_type == "AWS_LAMBDA":
        record["lambdaConfig"] = body.get("lambdaConfig", {})
    elif ds_type == "AMAZON_ELASTICSEARCH" or ds_type == "AMAZON_OPENSEARCH_SERVICE":
        record["elasticsearchConfig"] = body.get("elasticsearchConfig", {})
    elif ds_type == "HTTP":
        record["httpConfig"] = body.get("httpConfig", {})
    elif ds_type == "RELATIONAL_DATABASE":
        record["relationalDatabaseConfig"] = body.get("relationalDatabaseConfig", {})

    _data_sources.setdefault(api_id, {})[name] = record
    return _json(200, {"dataSource": record})


def _get_data_source(api_id, name):
    if api_id not in _apis:
        return error_response_json("NotFoundException", f"GraphQL API {api_id} not found", 404)

    ds = _data_sources.get(api_id, {}).get(name)
    if not ds:
        return error_response_json("NotFoundException", f"Data source {name} not found", 404)

    return _json(200, {"dataSource": ds})


def _list_data_sources(api_id):
    if api_id not in _apis:
        return error_response_json("NotFoundException", f"GraphQL API {api_id} not found", 404)

    sources = list(_data_sources.get(api_id, {}).values())
    return _json(200, {"dataSources": sources})


def _delete_data_source(api_id, name):
    if api_id not in _apis:
        return error_response_json("NotFoundException", f"GraphQL API {api_id} not found", 404)

    sources = _data_sources.get(api_id, {})
    if name not in sources:
        return error_response_json("NotFoundException", f"Data source {name} not found", 404)

    del sources[name]
    return _json(200, {})


# ---------------------------------------------------------------------------
# Resolvers
# ---------------------------------------------------------------------------

def _create_resolver(api_id, type_name, body):
    if api_id not in _apis:
        return error_response_json("NotFoundException", f"GraphQL API {api_id} not found", 404)

    field_name = body.get("fieldName", "")
    data_source_name = body.get("dataSourceName")
    request_template = body.get("requestMappingTemplate", "")
    response_template = body.get("responseMappingTemplate", "")
    kind = body.get("kind", "UNIT")
    pipeline_config = body.get("pipelineConfig")
    caching_config = body.get("cachingConfig")
    runtime = body.get("runtime")
    code = body.get("code")

    arn = f"{_apis[api_id]['arn']}/types/{type_name}/resolvers/{field_name}"

    record = {
        "typeName": type_name,
        "fieldName": field_name,
        "dataSourceName": data_source_name,
        "resolverArn": arn,
        "requestMappingTemplate": request_template,
        "responseMappingTemplate": response_template,
        "kind": kind,
        "createdAt": _now(),
        "lastUpdatedAt": _now(),
    }
    if pipeline_config:
        record["pipelineConfig"] = pipeline_config
    if caching_config:
        record["cachingConfig"] = caching_config
    if runtime:
        record["runtime"] = runtime
    if code:
        record["code"] = code

    _resolvers.setdefault(api_id, {}).setdefault(type_name, {})[field_name] = record
    return _json(200, {"resolver": record})


def _get_resolver(api_id, type_name, field_name):
    if api_id not in _apis:
        return error_response_json("NotFoundException", f"GraphQL API {api_id} not found", 404)

    resolver = _resolvers.get(api_id, {}).get(type_name, {}).get(field_name)
    if not resolver:
        return error_response_json("NotFoundException",
                                   f"Resolver {type_name}.{field_name} not found", 404)

    return _json(200, {"resolver": resolver})


def _list_resolvers(api_id, type_name):
    if api_id not in _apis:
        return error_response_json("NotFoundException", f"GraphQL API {api_id} not found", 404)

    resolvers = list(_resolvers.get(api_id, {}).get(type_name, {}).values())
    return _json(200, {"resolvers": resolvers})


def _delete_resolver(api_id, type_name, field_name):
    if api_id not in _apis:
        return error_response_json("NotFoundException", f"GraphQL API {api_id} not found", 404)

    type_resolvers = _resolvers.get(api_id, {}).get(type_name, {})
    if field_name not in type_resolvers:
        return error_response_json("NotFoundException",
                                   f"Resolver {type_name}.{field_name} not found", 404)

    del type_resolvers[field_name]
    return _json(200, {})


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

def _create_type(api_id, body):
    if api_id not in _apis:
        return error_response_json("NotFoundException", f"GraphQL API {api_id} not found", 404)

    definition = body.get("definition", "")
    fmt = body.get("format", "SDL")

    # Extract type name from SDL definition (e.g. "type Query { ... }" -> "Query")
    name_match = re.search(r"(?:type|input|enum|interface|union|scalar)\s+(\w+)", definition)
    type_name = name_match.group(1) if name_match else "Unknown"

    arn = f"{_apis[api_id]['arn']}/types/{type_name}"

    record = {
        "name": type_name,
        "description": body.get("description", ""),
        "arn": arn,
        "definition": definition,
        "format": fmt,
        "createdAt": _now(),
        "lastUpdatedAt": _now(),
    }

    _types.setdefault(api_id, {})[type_name] = record
    return _json(200, {"type": record})


def _get_type(api_id, type_name, query_params):
    if api_id not in _apis:
        return error_response_json("NotFoundException", f"GraphQL API {api_id} not found", 404)

    fmt = "SDL"
    if query_params.get("format"):
        fmt_val = query_params["format"]
        fmt = fmt_val[0] if isinstance(fmt_val, list) else fmt_val

    t = _types.get(api_id, {}).get(type_name)
    if not t:
        return error_response_json("NotFoundException", f"Type {type_name} not found", 404)

    return _json(200, {"type": t})


def _list_types(api_id, query_params):
    if api_id not in _apis:
        return error_response_json("NotFoundException", f"GraphQL API {api_id} not found", 404)

    types = list(_types.get(api_id, {}).values())
    return _json(200, {"types": types})


# ---------------------------------------------------------------------------
# Tags
# ---------------------------------------------------------------------------

def _tag_resource(body):
    arn = body.get("resourceArn", "")
    tags = body.get("tags", {})
    _tags.setdefault(arn, {}).update(tags)
    return _json(200, {})


def _untag_resource(arn, query_params):
    tag_keys = query_params.get("tagKeys", [])
    if isinstance(tag_keys, str):
        tag_keys = [tag_keys]
    existing = _tags.get(arn, {})
    for k in tag_keys:
        existing.pop(k, None)
    return _json(200, {})


def _list_tags_for_resource(arn):
    tags = _tags.get(arn, {})
    return _json(200, {"tags": tags})


# ---------------------------------------------------------------------------
# Request router
# ---------------------------------------------------------------------------

# Path patterns for routing
_PATH_RE = re.compile(r"^/v1/apis(?:/([^/]+))?(?:/([^/]+))?(?:/([^/]+))?(?:/([^/]+))?(?:/([^/]+))?")
# /v1/apis                          -> groups: (None, None, None, None, None)
# /v1/apis/{apiId}                  -> groups: (apiId, None, None, None, None)
# /v1/apis/{apiId}/apikeys          -> groups: (apiId, "apikeys", None, None, None)
# /v1/apis/{apiId}/apikeys/{id}     -> groups: (apiId, "apikeys", id, None, None)
# /v1/apis/{apiId}/datasources      -> groups: (apiId, "datasources", None, None, None)
# /v1/apis/{apiId}/datasources/{n}  -> groups: (apiId, "datasources", name, None, None)
# /v1/apis/{apiId}/types            -> groups: (apiId, "types", None, None, None)
# /v1/apis/{apiId}/types/{t}/resolvers          -> (apiId, "types", t, "resolvers", None)
# /v1/apis/{apiId}/types/{t}/resolvers/{field}  -> (apiId, "types", t, "resolvers", field)


async def handle_request(method, path, headers, body, query_params):
    """Main entry point — route AppSync REST requests."""

    # Tags endpoint: /v1/tags/{resourceArn}
    if path.startswith("/v1/tags/"):
        from urllib.parse import unquote
        arn = unquote(path[len("/v1/tags/"):])
        if method == "POST":
            data = json.loads(body) if body else {}
            data["resourceArn"] = arn
            return _tag_resource(data)
        elif method == "DELETE":
            return _untag_resource(arn, query_params)
        else:  # GET
            return _list_tags_for_resource(arn)

    m = _PATH_RE.match(path)
    if not m:
        return error_response_json("NotFoundException", f"Unknown path: {path}", 404)

    api_id, sub1, sub2, sub3, sub4 = m.groups()

    data = {}
    if body:
        try:
            data = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            data = {}

    # POST /v1/apis — CreateGraphQLApi
    if api_id is None and sub1 is None:
        if method == "POST":
            return _create_graphql_api(data)
        elif method == "GET":
            return _list_graphql_apis(query_params)

    # /v1/apis/{apiId}
    if api_id and sub1 is None:
        if method == "GET":
            return _get_graphql_api(api_id)
        elif method == "POST":
            return _update_graphql_api(api_id, data)
        elif method == "DELETE":
            return _delete_graphql_api(api_id)

    # /v1/apis/{apiId}/apikeys
    if sub1 == "apikeys":
        if sub2 is None:
            if method == "POST":
                return _create_api_key(api_id, data)
            elif method == "GET":
                return _list_api_keys(api_id)
        else:
            # /v1/apis/{apiId}/apikeys/{keyId}
            if method == "DELETE":
                return _delete_api_key(api_id, sub2)

    # /v1/apis/{apiId}/datasources
    if sub1 == "datasources":
        if sub2 is None:
            if method == "POST":
                return _create_data_source(api_id, data)
            elif method == "GET":
                return _list_data_sources(api_id)
        else:
            # /v1/apis/{apiId}/datasources/{name}
            if method == "GET":
                return _get_data_source(api_id, sub2)
            elif method == "DELETE":
                return _delete_data_source(api_id, sub2)

    # /v1/apis/{apiId}/types
    if sub1 == "types":
        if sub2 is None:
            if method == "POST":
                return _create_type(api_id, data)
            elif method == "GET":
                return _list_types(api_id, query_params)
        elif sub3 == "resolvers":
            # /v1/apis/{apiId}/types/{typeName}/resolvers
            type_name = sub2
            if sub4 is None:
                if method == "POST":
                    return _create_resolver(api_id, type_name, data)
                elif method == "GET":
                    return _list_resolvers(api_id, type_name)
            else:
                # /v1/apis/{apiId}/types/{typeName}/resolvers/{fieldName}
                field_name = sub4
                if method == "GET":
                    return _get_resolver(api_id, type_name, field_name)
                elif method == "DELETE":
                    return _delete_resolver(api_id, type_name, field_name)
        else:
            # /v1/apis/{apiId}/types/{typeName} — GetType
            if sub3 is None and method == "GET":
                return _get_type(api_id, sub2, query_params)

    return error_response_json("BadRequestException", f"Unsupported route: {method} {path}")


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------

def reset():
    """Clear all in-memory state."""
    _apis.clear()
    _api_keys.clear()
    _data_sources.clear()
    _resolvers.clear()
    _types.clear()
    _tags.clear()


def get_state():
    """Return a deep copy of all state for persistence."""
    return copy.deepcopy({
        "apis": _apis,
        "api_keys": _api_keys,
        "data_sources": _data_sources,
        "resolvers": _resolvers,
        "types": _types,
        "tags": _tags,
    })


def restore_state(data):
    """Restore state from persisted data."""
    _apis.update(data.get("apis", {}))
    _api_keys.update(data.get("api_keys", {}))
    _data_sources.update(data.get("data_sources", {}))
    _resolvers.update(data.get("resolvers", {}))
    _types.update(data.get("types", {}))
    _tags.update(data.get("tags", {}))
