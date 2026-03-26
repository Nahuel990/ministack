# Contributing to MiniStack

Thanks for wanting to contribute. The codebase is intentionally simple — each AWS service is a single self-contained Python file. Adding a new service or fixing a bug should take minutes, not hours.

## Adding a New Service

Every service follows the same 4-step pattern:

### 1. Create `services/myservice.py`

```python
"""
MyService Emulator.
JSON-based API via X-Amz-Target (or Query API).
Supports: OperationOne, OperationTwo, ...
"""

import json
import logging
from core.responses import json_response, error_response_json, new_uuid

logger = logging.getLogger("myservice")

ACCOUNT_ID = "000000000000"
REGION = "us-east-1"

_state: dict = {}  # in-memory storage


async def handle_request(method, path, headers, body, query_params):
    target = headers.get("x-amz-target", "")
    action = target.split(".")[-1] if "." in target else ""

    try:
        data = json.loads(body) if body else {}
    except json.JSONDecodeError:
        return error_response_json("SerializationException", "Invalid JSON", 400)

    handlers = {
        "OperationOne": _operation_one,
        "OperationTwo": _operation_two,
    }

    handler = handlers.get(action)
    if not handler:
        return error_response_json("InvalidAction", f"Unknown action: {action}", 400)
    return handler(data)


def _operation_one(data):
    # implement
    return json_response({"result": "ok"})


def _operation_two(data):
    # implement
    return json_response({})
```

**Protocol guide:**
- JSON services (DynamoDB, SecretsManager, Glue, Athena, etc.) — use `json_response` / `error_response_json`, route via `X-Amz-Target`
- XML services (S3, SQS, SNS, IAM, RDS, ElastiCache) — build XML responses, route via `Action` query param
- REST services (Lambda, ECS) — route via URL path

### 2. Register in `app.py`

```python
from services import myservice

SERVICE_HANDLERS = {
    # ... existing ...
    "myservice": myservice.handle_request,
}
```

### 3. Add detection to `core/router.py`

```python
SERVICE_PATTERNS = {
    # ... existing ...
    "myservice": {
        "target_prefixes": ["AWSMyService"],   # for X-Amz-Target routing
        "host_patterns": [r"myservice\."],      # for host-based routing
    },
}
```

And add any `Action`-based operations to `action_service_map` in `detect_service()`.

### 4. Add tests to `tests/test_services.py`

```python
@test("MyService: basic operation")
def test_myservice_basic():
    svc = boto3.client("myservice", **kwargs)
    resp = svc.operation_one(Param="value")
    assert resp["result"] == "ok"
```

Add your test function to the `tests` list in `main()`.

---

## Running Tests Locally

```bash
# Start the stack
docker compose up -d

# Install deps
pip install boto3 pytest pytest-cov duckdb

# Run all tests
pytest tests/ -v

# Run a specific test
pytest tests/ -v -k "test_myservice"
```

---

## Code Conventions

- **One file per service** — keep everything for a service in `services/myservice.py`
- **In-memory state** — use module-level dicts (`_things: dict = {}`)
- **No external AWS deps** — no `boto3`, `botocore`, or `aws-sdk` in service code
- **Minimal dependencies** — `duckdb` and `docker` are optional; guard with try/except
- **Error responses** — match real AWS error codes and HTTP status codes as closely as possible
- **Logging** — use `logger = logging.getLogger("servicename")` and log at DEBUG for request details, INFO for significant events, WARNING for degraded behavior

---

## Pull Request Checklist

- [ ] New service file in `services/`
- [ ] Registered in `app.py` SERVICE_HANDLERS
- [ ] Detection patterns added to `core/router.py`
- [ ] Tests added and passing (`pytest tests/ -v`)
- [ ] Service added to the table in `README.md`
- [ ] Entry added to `CHANGELOG.md`

---

## What We're Looking For

High-value contributions right now:

- **API Gateway REST API (v1)** — resource trees, methods, integration types, deployment stages; HTTP API v2 is already implemented
- **Cognito** — user pools, sign-up/sign-in, token issuance
- **Route53** — hosted zones, record sets, health checks
- **ACM** — certificate provisioning and validation stubs
- **Firehose** — delivery streams to S3/Elasticsearch
- **More S3 operations** — object lock, replication configuration, website hosting
- **Virtual-hosted style S3** — `bucket.localhost:4566` routing in addition to path-style
- **Athena without DuckDB** — graceful degradation or lighter SQL engine option

---

## Questions?

Open a GitHub Discussion or file an issue with the `question` label.
