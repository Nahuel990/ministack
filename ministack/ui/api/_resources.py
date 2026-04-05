"""Generic resource browsing — stats, resource listing, and resource detail."""

from ministack.ui import interceptor
from ministack.ui.api._common import get_query_param, json_response, safe_serialize

_registry = None


def _get_registry():
    global _registry
    if _registry is None:
        _registry = _build_resource_registry()
    return _registry


def _build_resource_registry():
    """Build the registry lazily on first call (avoids import-time circular deps)."""
    from ministack.services import (
        acm,
        alb,
        apigateway,
        appsync,
        athena,
        cloudfront,
        cloudwatch,
        cloudwatch_logs,
        cognito,
        dynamodb,
        ec2,
        ecr,
        ecs,
        efs,
        elasticache,
        emr,
        eventbridge,
        firehose,
        glue,
        iam_sts,
        kinesis,
        kms,
        lambda_svc,
        rds,
        route53,
        s3,
        secretsmanager,
        ses,
        sns,
        sqs,
        ssm,
        stepfunctions,
        waf,
    )

    return {
        "s3": [("buckets", s3, "_buckets")],
        "sqs": [("queues", sqs, "_queues")],
        "sns": [("topics", sns, "_topics")],
        "dynamodb": [("tables", dynamodb, "_tables")],
        "lambda": [("functions", lambda_svc, "_functions"), ("layers", lambda_svc, "_layers")],
        "iam": [("users", iam_sts, "_users"), ("roles", iam_sts, "_roles"), ("policies", iam_sts, "_policies")],
        "secretsmanager": [("secrets", secretsmanager, "_secrets")],
        "logs": [("log_groups", cloudwatch_logs, "_log_groups")],
        "ssm": [("parameters", ssm, "_parameters")],
        "events": [("event_buses", eventbridge, "_event_buses"), ("rules", eventbridge, "_rules")],
        "kinesis": [("streams", kinesis, "_streams")],
        "monitoring": [("alarms", cloudwatch, "_alarms"), ("dashboards", cloudwatch, "_dashboards")],
        "ses": [("identities", ses, "_identities") if hasattr(ses, "_identities") else ("templates", ses, "_templates")],
        "states": [("state_machines", stepfunctions, "_state_machines"), ("executions", stepfunctions, "_executions")],
        "ecs": [("clusters", ecs, "_clusters"), ("task_definitions", ecs, "_task_defs"), ("services", ecs, "_services")],
        "rds": [("db_instances", rds, "_instances"), ("db_clusters", rds, "_clusters")],
        "elasticache": [("cache_clusters", elasticache, "_clusters") if hasattr(elasticache, "_clusters") else ("cache_clusters", elasticache, "_cache_clusters")],
        "glue": [("databases", glue, "_databases"), ("tables", glue, "_tables")],
        "athena": [("workgroups", athena, "_workgroups") if hasattr(athena, "_workgroups") else ("queries", athena, "_queries")],
        "apigateway": [("apis", apigateway, "_apis")],
        "firehose": [("delivery_streams", firehose, "_streams")],
        "route53": [("hosted_zones", route53, "_hosted_zones")],
        "cognito-idp": [("user_pools", cognito, "_user_pools")],
        "cognito-identity": [("identity_pools", cognito, "_identity_pools")],
        "ec2": [
            ("instances", ec2, "_instances"),
            ("vpcs", ec2, "_vpcs"),
            ("subnets", ec2, "_subnets"),
            ("security_groups", ec2, "_security_groups"),
            ("volumes", ec2, "_volumes"),
        ],
        "elasticmapreduce": [("clusters", emr, "_clusters")],
        "elasticloadbalancing": [("load_balancers", alb, "_load_balancers")],
        "elasticfilesystem": [("file_systems", efs, "_file_systems")],
        "kms": [("keys", kms, "_keys")],
        "cloudfront": [("distributions", cloudfront, "_distributions")],
        "appsync": [("graphql_apis", appsync, "_apis")],
        "ecr": [("repositories", ecr, "_repositories")],
        "acm": [("certificates", acm, "_certificates")],
        "wafv2": [("web_acls", waf, "_web_acls")],
        "cloudformation": [],
    }


async def handle_stats(send):
    """Return aggregated resource counts for all services."""
    registry = _get_registry()
    services = {}
    total = 0

    for svc_name, entries in registry.items():
        resources = {}
        for label, module, attr_name in entries:
            try:
                obj = getattr(module, attr_name, None)
                count = len(obj) if obj is not None else 0
            except Exception:
                count = 0
            resources[label] = count
            total += count
        services[svc_name] = {
            "status": "available",
            "resources": resources,
        }

    await json_response(send, {
        "services": services,
        "total_resources": total,
        "uptime_seconds": interceptor.get_uptime(),
    })


async def handle_resources(rel_path: str, query_params: dict, send):
    """Handle /resources/{service} and /resources/{service}/{type}/{id}."""
    parts = [p for p in rel_path.split("/") if p]
    registry = _get_registry()

    if not parts:
        await json_response(send, {"error": "Service name required"}, status=400)
        return

    svc_name = parts[0]
    if svc_name not in registry:
        await json_response(send, {"error": f"Unknown service: {svc_name}"}, status=404)
        return

    entries = registry[svc_name]

    # GET /resources/{service} — list all resource types and their items
    if len(parts) == 1:
        type_filter = get_query_param(query_params, "type")
        resources = {}
        for label, module, attr_name in entries:
            if type_filter and label != type_filter:
                continue
            try:
                obj = getattr(module, attr_name, {})
                if isinstance(obj, dict):
                    resources[label] = [
                        _summarize_resource(key, value)
                        for key, value in list(obj.items())[:200]
                    ]
                else:
                    resources[label] = [{"id": str(i)} for i in range(min(len(obj), 200))]
            except Exception as e:
                resources[label] = {"error": str(e)}

        await json_response(send, {"service": svc_name, "resources": resources})
        return

    # GET /resources/{service}/{type}/{id} — single resource detail
    if len(parts) >= 3:
        res_type = parts[1]
        res_id = "/".join(parts[2:])
        for label, module, attr_name in entries:
            if label != res_type:
                continue
            try:
                obj = getattr(module, attr_name, {})
                if isinstance(obj, dict) and res_id in obj:
                    detail = obj[res_id]
                    await json_response(send, {
                        "service": svc_name,
                        "type": res_type,
                        "id": res_id,
                        "detail": safe_serialize(detail),
                    })
                    return
            except Exception as e:
                await json_response(send, {"error": str(e)}, status=500)
                return

        await json_response(send, {"error": f"Resource not found: {res_type}/{res_id}"}, status=404)
        return

    await json_response(send, {"error": "Invalid resource path"}, status=400)


def _summarize_resource(key, value) -> dict:
    """Extract a summary dict from a resource state entry."""
    summary = {"id": str(key)}

    if isinstance(value, dict):
        for field in ("Name", "name", "Arn", "arn", "ARN", "Status", "status",
                      "State", "state", "CreatedAt", "created", "CreationDate",
                      "Engine", "Runtime", "runtime", "FunctionName", "TableName",
                      "BucketName", "QueueArn", "TopicArn", "Description"):
            if field in value:
                val = value[field]
                if isinstance(val, (str, int, float, bool)):
                    summary[field] = val

        for count_field in ("objects", "items", "messages", "records"):
            if count_field in value and isinstance(value[count_field], (dict, list)):
                summary[f"{count_field}_count"] = len(value[count_field])

    return summary
