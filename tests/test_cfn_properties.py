"""Hypothesis property-based tests for CloudFormation engine pure functions."""

import json
import string
import re

import pytest
from hypothesis import given, settings, assume
import hypothesis.strategies as st

from ministack.services.cloudformation import (
    _parse_template,
    _validate_template,
    _resolve_parameters,
    _evaluate_conditions,
    _resolve_refs,
    _extract_deps,
    _topological_sort,
    _NO_VALUE,
)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
STACK_NAME = "test"
STACK_ID = "arn:aws:cloudformation:us-east-1:000000000000:stack/test/uuid"
EMPTY_RESOURCES: dict = {}
EMPTY_PARAMS: dict = {}
EMPTY_CONDITIONS: dict = {}
EMPTY_MAPPINGS: dict = {}

# Intrinsic function key prefixes to filter out
_INTRINSIC_PREFIXES = ("Ref", "Fn::Join", "Fn::Sub", "Fn::Select", "Fn::Split",
                       "Fn::If", "Fn::Base64", "Fn::FindInMap", "Fn::GetAtt",
                       "Fn::GetAZs", "Fn::Cidr", "Fn::ImportValue", "Fn::Equals",
                       "Fn::And", "Fn::Or", "Fn::Not", "Condition")

def _has_intrinsic_keys(obj):
    """Return True if obj (or any nested value) contains intrinsic function keys."""
    if isinstance(obj, dict):
        for key in obj:
            if key in _INTRINSIC_PREFIXES or key.startswith("Fn::"):
                return True
            if _has_intrinsic_keys(obj[key]):
                return True
    elif isinstance(obj, list):
        return any(_has_intrinsic_keys(item) for item in obj)
    return False

# ---------------------------------------------------------------------------
# Shared strategies
# ---------------------------------------------------------------------------
_json_values = st.recursive(
    st.text(min_size=0, max_size=20) | st.integers(-1000, 1000) | st.booleans(),
    lambda children: (
        st.lists(children, max_size=5)
        | st.dictionaries(
            st.text(alphabet=string.ascii_letters, min_size=1, max_size=8),
            children,
            max_size=5,
        )
    ),
    max_leaves=20,
)

@st.composite
def random_dag(draw):
    """Generate a random DAG as a CloudFormation resources dict.

    Resources are named R0..R(N-1). Edges go from higher-index to lower-index
    only (j depends on i where i < j), guaranteeing acyclicity.
    """
    n = draw(st.integers(min_value=2, max_value=15))
    resources = {}
    for j in range(n):
        deps = []
        for i in range(j):
            if draw(st.booleans()):
                deps.append(f"R{i}")
        defn = {"Type": "AWS::CloudFormation::WaitConditionHandle"}
        if deps:
            defn["DependsOn"] = deps
        resources[f"R{j}"] = defn
    return resources

@st.composite
def cyclic_resources(draw):
    """Generate resources with a guaranteed circular dependency (ring)."""
    n = draw(st.integers(min_value=3, max_value=8))
    resources = {}
    for i in range(n):
        dep_index = (i - 1) % n
        resources[f"R{i}"] = {
            "Type": "AWS::CloudFormation::WaitConditionHandle",
            "DependsOn": [f"R{dep_index}"],
        }
    return resources

@given(
    st.dictionaries(
        st.text(alphabet=string.ascii_letters, min_size=1, max_size=8),
        _json_values,
        min_size=1,
        max_size=5,
    )
)
@settings(max_examples=100)
def test_ht001_parse_template_json_roundtrip(d):
    d["Resources"] = {}
    raw = json.dumps(d)
    parsed = _parse_template(raw)
    for key in d:
        assert key in parsed, f"Key {key!r} missing after parse"

@given(_json_values)
@settings(max_examples=100)
def test_ht002_resolve_refs_identity_on_plain_values(value):
    assume(not _has_intrinsic_keys(value))
    result = _resolve_refs(
        value, EMPTY_RESOURCES, EMPTY_PARAMS, EMPTY_CONDITIONS,
        EMPTY_MAPPINGS, STACK_NAME, STACK_ID,
    )
    assert result == value

@given(random_dag())
@settings(max_examples=100)
def test_ht003_topo_sort_all_resources_present(resources):
    order = _topological_sort(resources, {})
    assert set(order) == set(resources.keys())

@given(random_dag())
@settings(max_examples=100)
def test_ht004_topo_sort_deps_respected(resources):
    order = _topological_sort(resources, {})
    for name, defn in resources.items():
        deps = defn.get("DependsOn", [])
        if isinstance(deps, str):
            deps = [deps]
        for dep in deps:
            assert order.index(dep) < order.index(name), (
                f"{dep} should come before {name} in topological order"
            )

@given(random_dag())
@settings(max_examples=100)
def test_ht005_topo_sort_deterministic(resources):
    order1 = _topological_sort(resources, {})
    order2 = _topological_sort(resources, {})
    assert order1 == order2

@given(cyclic_resources())
@settings(max_examples=100)
def test_ht006_topo_sort_circular_detection(resources):
    with pytest.raises(ValueError, match=r"(?i)circular"):
        _topological_sort(resources, {})

@given(
    st.lists(
        st.tuples(
            st.text(alphabet=string.ascii_letters, min_size=1, max_size=10),
            st.text(min_size=0, max_size=10),
            st.text(min_size=0, max_size=10),
        ),
        min_size=1,
        max_size=5,
    )
)
@settings(max_examples=100)
def test_ht007_condition_evaluator_is_pure(cond_specs):
    conditions = {}
    for name, left, right in cond_specs:
        conditions[name] = {"Fn::Equals": [left, right]}
    template = {"Conditions": conditions}
    params: dict = {}
    result1 = _evaluate_conditions(template, params)
    result2 = _evaluate_conditions(template, params)
    assert result1 == result2

@given(
    st.text(alphabet=string.ascii_letters + string.digits, min_size=1, max_size=3),
    st.lists(
        st.text(alphabet=string.ascii_letters + string.digits, min_size=1, max_size=10),
        min_size=1,
        max_size=10,
    ),
)
@settings(max_examples=100)
def test_ht008_join_split_roundtrip(delimiter, parts):
    # Ensure no part contains the delimiter
    assume(all(delimiter not in part for part in parts))

    joined = _resolve_refs(
        {"Fn::Join": [delimiter, parts]},
        EMPTY_RESOURCES, EMPTY_PARAMS, EMPTY_CONDITIONS,
        EMPTY_MAPPINGS, STACK_NAME, STACK_ID,
    )
    split_back = _resolve_refs(
        {"Fn::Split": [delimiter, joined]},
        EMPTY_RESOURCES, EMPTY_PARAMS, EMPTY_CONDITIONS,
        EMPTY_MAPPINGS, STACK_NAME, STACK_ID,
    )
    assert split_back == parts

@given(
    st.lists(
        st.text(alphabet=string.ascii_letters, min_size=1, max_size=10),
        min_size=1,
        max_size=20,
    ),
    st.data(),
)
@settings(max_examples=100)
def test_ht009_fn_select_valid_index(items, data):
    idx = data.draw(st.integers(min_value=0, max_value=len(items) - 1))
    result = _resolve_refs(
        {"Fn::Select": [idx, items]},
        EMPTY_RESOURCES, EMPTY_PARAMS, EMPTY_CONDITIONS,
        EMPTY_MAPPINGS, STACK_NAME, STACK_ID,
    )
    assert result == items[idx]

@given(
    st.lists(
        st.text(alphabet=string.ascii_letters + string.digits, min_size=1, max_size=10),
        min_size=2,
        max_size=5,
        unique=True,
    ),
    st.text(alphabet=string.ascii_letters + string.digits, min_size=1, max_size=10),
)
@settings(max_examples=100)
def test_ht010_parameter_allowed_values(allowed, test_value):
    template = {
        "Parameters": {
            "Param1": {
                "Type": "String",
                "AllowedValues": allowed,
            }
        }
    }
    provided = [{"Key": "Param1", "Value": test_value}]
    if test_value in allowed:
        result = _resolve_parameters(template, provided)
        assert result["Param1"]["Value"] == test_value
    else:
        with pytest.raises(ValueError, match="AllowedValues"):
            _resolve_parameters(template, provided)

@given(
    st.lists(
        st.text(alphabet=string.ascii_letters, min_size=1, max_size=8),
        min_size=1,
        max_size=5,
        unique=True,
    ),
)
@settings(max_examples=100)
def test_ht011_fn_sub_completeness(var_names):
    template_str = " ".join(f"${{{name}}}" for name in var_names)
    mapping = {name: f"val_{name}" for name in var_names}
    result = _resolve_refs(
        {"Fn::Sub": [template_str, mapping]},
        EMPTY_RESOURCES, EMPTY_PARAMS, EMPTY_CONDITIONS,
        EMPTY_MAPPINGS, STACK_NAME, STACK_ID,
    )
    assert "${" not in result, (
        f"Unresolved substitution in result: {result!r}"
    )
