"""
CloudFormation helpers — XML response formatting and parameter extraction utilities.
"""

from html import escape as _esc

from ministack.core.responses import new_uuid

CFN_NS = "http://cloudformation.amazonaws.com/doc/2010-05-08/"


def _p(params, key, default=""):
    """Extract a single value from parsed query-string params."""
    val = params.get(key, [default])
    return val[0] if isinstance(val, list) else val


def _xml(status, root_tag, inner):
    body = (
        f'<?xml version="1.0" encoding="UTF-8"?>'
        f'<{root_tag} xmlns="{CFN_NS}">'
        f'{inner}'
        f'<ResponseMetadata><RequestId>{new_uuid()}</RequestId></ResponseMetadata>'
        f'</{root_tag}>'
    ).encode("utf-8")
    return status, {"Content-Type": "application/xml"}, body


def _error(code, message, status=400):
    t = "Sender" if status < 500 else "Receiver"
    body = (
        f'<?xml version="1.0" encoding="UTF-8"?>'
        f'<ErrorResponse xmlns="{CFN_NS}">'
        f'<Error><Type>{t}</Type><Code>{code}</Code>'
        f'<Message>{_esc(message)}</Message></Error>'
        f'<RequestId>{new_uuid()}</RequestId>'
        f'</ErrorResponse>'
    ).encode("utf-8")
    return status, {"Content-Type": "application/xml"}, body


def _extract_members(params, prefix):
    """Extract Parameters.member.N.Key/Value or Tags.member.N.Key/Value."""
    result = []
    i = 1
    while True:
        key = (_p(params, f"{prefix}.member.{i}.ParameterKey")
               or _p(params, f"{prefix}.member.{i}.Key"))
        if not key:
            break
        value = (_p(params, f"{prefix}.member.{i}.ParameterValue")
                 or _p(params, f"{prefix}.member.{i}.Value"))
        result.append({"Key": key, "Value": value or ""})
        i += 1
    return result


def _extract_stack_status_filters(params):
    """Extract StackStatusFilter.member.N values."""
    filters = []
    i = 1
    while True:
        val = _p(params, f"StackStatusFilter.member.{i}")
        if not val:
            break
        filters.append(val)
        i += 1
    return filters
