"""
EMR (Elastic MapReduce) Service Emulator.
JSON protocol via X-Amz-Target: ElasticMapReduce.{Operation}
In-memory only — no real Spark/Hadoop execution.

Supports:
  Clusters:        RunJobFlow, DescribeCluster, ListClusters, TerminateJobFlows,
                   ModifyCluster, SetTerminationProtection, SetVisibleToAllUsers
  Steps:           AddJobFlowSteps, DescribeStep, ListSteps, CancelSteps
  Instance Fleets: AddInstanceFleet, ListInstanceFleets, ModifyInstanceFleet
  Instance Groups: AddInstanceGroups, ListInstanceGroups, ModifyInstanceGroups
  Bootstrap:       ListBootstrapActions
  Tags:            AddTags, RemoveTags
  Block Public Access: GetBlockPublicAccessConfiguration, PutBlockPublicAccessConfiguration
"""

import json
import logging
import os
import random
import string
import time

from ministack.core.responses import error_response_json, json_response, new_uuid
from ministack.core import k8s_spark

logger = logging.getLogger("emr")

ACCOUNT_ID = os.environ.get("MINISTACK_ACCOUNT_ID", "000000000000")
REGION = os.environ.get("MINISTACK_REGION", "us-east-1")

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

_clusters: dict = {}   # cluster_id -> cluster record
_steps: dict = {}      # cluster_id -> [step records]
_block_public_access: dict = {
    "BlockPublicSecurityGroupRules": False,
    "PermittedPublicSecurityGroupRuleRanges": [],
}

# ---------------------------------------------------------------------------
# ID generators
# ---------------------------------------------------------------------------

def _cluster_id():
    chars = string.ascii_uppercase + string.digits
    return "j-" + "".join(random.choices(chars, k=13))

def _step_id():
    chars = string.ascii_uppercase + string.digits
    return "s-" + "".join(random.choices(chars, k=13))

def _fleet_id():
    chars = string.ascii_uppercase + string.digits
    return "if-" + "".join(random.choices(chars, k=13))

def _group_id():
    chars = string.ascii_uppercase + string.digits
    return "ig-" + "".join(random.choices(chars, k=13))

def _now_iso():
    """Return current time as epoch seconds (float).

    AWS EMR's JSON wire protocol uses epoch-second timestamps, not ISO-8601.
    The AWS SDK v2 unmarshalls these as numbers.
    """
    return time.time()

# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def _run_job_flow(data):
    name = data.get("Name")
    if not name:
        return error_response_json("ValidationException", "Name is required", 400)

    cluster_id = _cluster_id()
    arn = f"arn:aws:elasticmapreduce:{REGION}:{ACCOUNT_ID}:cluster/{cluster_id}"
    instances = data.get("Instances", {})
    keep_alive = instances.get("KeepJobFlowAliveWhenNoSteps", False)
    tags = data.get("Tags", [])
    applications = data.get("Applications", [])
    bootstrap_actions = data.get("BootstrapActions", [])
    release_label = data.get("ReleaseLabel", "emr-6.10.0")
    log_uri = data.get("LogUri", "")
    service_role = data.get("ServiceRole", "EMR_DefaultRole")
    job_flow_role = data.get("JobFlowRole", "EMR_EC2_DefaultRole")
    visible_to_all = data.get("VisibleToAllUsers", True)
    termination_protected = instances.get("TerminationProtected", False)
    now = _now_iso()

    # Derive initial state: WAITING if keep_alive, else TERMINATED (no real execution)
    initial_state = "WAITING" if keep_alive else "TERMINATED"

    # Build instance fleets/groups from Instances config
    instance_fleets = []
    instance_groups = []

    if instances.get("InstanceFleets"):
        for fleet in instances["InstanceFleets"]:
            instance_fleets.append({
                "Id": _fleet_id(),
                "Name": fleet.get("Name", fleet.get("InstanceFleetType", "MASTER")),
                "Status": {"State": "RUNNING", "StateChangeReason": {}, "Timeline": {"CreationDateTime": now}},
                "InstanceFleetType": fleet.get("InstanceFleetType", "MASTER"),
                "TargetOnDemandCapacity": fleet.get("TargetOnDemandCapacity", 0),
                "TargetSpotCapacity": fleet.get("TargetSpotCapacity", 0),
                "ProvisionedOnDemandCapacity": fleet.get("TargetOnDemandCapacity", 0),
                "ProvisionedSpotCapacity": fleet.get("TargetSpotCapacity", 0),
                "InstanceTypeSpecifications": fleet.get("InstanceTypeConfigs", []),
            })
    elif instances.get("InstanceGroups"):
        for ig in instances["InstanceGroups"]:
            instance_groups.append({
                "Id": _group_id(),
                "Name": ig.get("Name", ig.get("InstanceRole", "MASTER")),
                "Market": ig.get("Market", "ON_DEMAND"),
                "InstanceGroupType": ig.get("InstanceRole", "MASTER"),
                "InstanceType": ig.get("InstanceType", "m5.xlarge"),
                "RequestedInstanceCount": ig.get("InstanceCount", 1),
                "RunningInstanceCount": ig.get("InstanceCount", 1),
                "Status": {"State": "RUNNING", "StateChangeReason": {}, "Timeline": {"CreationDateTime": now}},
            })
    else:
        # Simple mode: MasterInstanceType / SlaveInstanceType / InstanceCount
        master_type = instances.get("MasterInstanceType", "m5.xlarge")
        slave_type = instances.get("SlaveInstanceType", "m5.xlarge")
        instance_count = instances.get("InstanceCount", 1)
        instance_groups = [
            {
                "Id": _group_id(),
                "Name": "Master",
                "Market": "ON_DEMAND",
                "InstanceGroupType": "MASTER",
                "InstanceType": master_type,
                "RequestedInstanceCount": 1,
                "RunningInstanceCount": 1,
                "Status": {"State": "RUNNING", "StateChangeReason": {}, "Timeline": {"CreationDateTime": now}},
            },
        ]
        if instance_count > 1:
            instance_groups.append({
                "Id": _group_id(),
                "Name": "Core",
                "Market": "ON_DEMAND",
                "InstanceGroupType": "CORE",
                "InstanceType": slave_type,
                "RequestedInstanceCount": instance_count - 1,
                "RunningInstanceCount": instance_count - 1,
                "Status": {"State": "RUNNING", "StateChangeReason": {}, "Timeline": {"CreationDateTime": now}},
            })

    collection_type = "INSTANCE_FLEET" if instance_fleets else "INSTANCE_GROUP"

    _clusters[cluster_id] = {
        "Id": cluster_id,
        "Name": name,
        "ClusterArn": arn,
        "Status": {
            "State": initial_state,
            "StateChangeReason": {"Code": "", "Message": ""},
            "Timeline": {"CreationDateTime": now, "ReadyDateTime": now},
        },
        "Ec2InstanceAttributes": {
            "Ec2KeyName": instances.get("Ec2KeyName", ""),
            "Ec2SubnetId": instances.get("Ec2SubnetId", ""),
            "Ec2AvailabilityZone": f"{REGION}a",
            "IamInstanceProfile": job_flow_role,
            "EmrManagedMasterSecurityGroup": instances.get("EmrManagedMasterSecurityGroup", ""),
            "EmrManagedSlaveSecurityGroup": instances.get("EmrManagedSlaveSecurityGroup", ""),
        },
        "InstanceCollectionType": collection_type,
        "LogUri": log_uri,
        "ReleaseLabel": release_label,
        "AutoTerminate": not keep_alive,
        "TerminationProtected": termination_protected,
        "VisibleToAllUsers": visible_to_all,
        "Applications": applications,
        "Tags": tags,
        "ServiceRole": service_role,
        "NormalizedInstanceHours": 0,
        "MasterPublicDnsName": "ec2-0-0-0-0.compute-1.amazonaws.com",
        "StepConcurrencyLevel": data.get("StepConcurrencyLevel", 1),
        "BootstrapActions": bootstrap_actions,
        "InstanceFleets": instance_fleets,
        "InstanceGroups": instance_groups,
    }

    # Steps passed at creation time
    steps_in = data.get("Steps", [])
    _steps[cluster_id] = []
    for step in steps_in:
        _steps[cluster_id].append(_make_step(step))

    return json_response({"JobFlowId": cluster_id, "ClusterArn": arn})


def _describe_cluster(data):
    cluster_id = data.get("ClusterId")
    cluster = _clusters.get(cluster_id)
    if not cluster:
        return error_response_json("InvalidRequestException",
                                   f"Cluster id '{cluster_id}' is not valid.", 400)
    return json_response({"Cluster": cluster})


def _list_clusters(data):
    state_filter = data.get("ClusterStates", [])
    result = []
    for c in _clusters.values():
        state = c["Status"]["State"]
        if state_filter and state not in state_filter:
            continue
        result.append({
            "Id": c["Id"],
            "Name": c["Name"],
            "Status": c["Status"],
            "NormalizedInstanceHours": c["NormalizedInstanceHours"],
            "ClusterArn": c["ClusterArn"],
        })
    return json_response({"Clusters": result})


def _terminate_job_flows(data):
    ids = data.get("JobFlowIds", [])
    for cid in ids:
        cluster = _clusters.get(cid)
        if cluster:
            if cluster.get("TerminationProtected"):
                return error_response_json(
                    "ValidationException",
                    f"Cluster {cid} is protected from termination. Disable termination protection first.", 400
                )
            cluster["Status"]["State"] = "TERMINATED"
            cluster["Status"]["StateChangeReason"] = {"Code": "USER_REQUEST", "Message": "User request"}
    return json_response({})


def _modify_cluster(data):
    cluster_id = data.get("ClusterId")
    cluster = _clusters.get(cluster_id)
    if not cluster:
        return error_response_json("InvalidRequestException",
                                   f"Cluster id '{cluster_id}' is not valid.", 400)
    if "StepConcurrencyLevel" in data:
        cluster["StepConcurrencyLevel"] = data["StepConcurrencyLevel"]
    return json_response({"StepConcurrencyLevel": cluster["StepConcurrencyLevel"]})


def _set_termination_protection(data):
    ids = data.get("JobFlowIds", [])
    protected = data.get("TerminationProtected", False)
    for cid in ids:
        if cid in _clusters:
            _clusters[cid]["TerminationProtected"] = protected
    return json_response({})


def _set_visible_to_all_users(data):
    ids = data.get("JobFlowIds", [])
    visible = data.get("VisibleToAllUsers", True)
    for cid in ids:
        if cid in _clusters:
            _clusters[cid]["VisibleToAllUsers"] = visible
    return json_response({})


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

# Spark-submit flags that take a value argument (flag + next token)
_SPARK_SUBMIT_VALUE_FLAGS = {
    "--class", "--master", "--deploy-mode", "--name", "--jars", "--packages",
    "--exclude-packages", "--repositories", "--py-files", "--files",
    "--driver-memory", "--driver-java-options", "--driver-library-path",
    "--driver-class-path", "--executor-memory", "--executor-cores",
    "--num-executors", "--total-executor-cores", "--queue", "--proxy-user",
    "--principal", "--keytab", "--conf", "--properties-file",
    "--driver-cores",
}


def _parse_spark_submit_args(args: list[str]) -> tuple[str, str, dict, list[str]]:
    """Parse a spark-submit argument list into components.

    Given args like: [--class, Foo, --master, yarn, --deploy-mode, cluster, --conf, k=v, s3://app.jar, app-arg1]
    Returns: (entry_point, class_name, spark_conf, app_args)

    Everything before the first non-flag token is a spark-submit option.
    The first non-flag token is the entry point (JAR/py file).
    Everything after that is application arguments.
    """
    class_name = ""
    spark_conf = {}
    i = 0
    entry_point = ""
    app_args = []

    # Parse spark-submit flags
    while i < len(args):
        arg = args[i]
        if arg == "--class" and i + 1 < len(args):
            class_name = args[i + 1]
            i += 2
        elif arg == "--conf" and i + 1 < len(args):
            kv = args[i + 1]
            if "=" in kv:
                k, v = kv.split("=", 1)
                spark_conf[k] = v
            i += 2
        elif arg in _SPARK_SUBMIT_VALUE_FLAGS and i + 1 < len(args):
            # Skip other flags with values (--master, --deploy-mode, etc.)
            i += 2
        elif arg.startswith("--"):
            # Boolean flag
            i += 1
        else:
            # First non-flag token is the entry point
            entry_point = arg
            app_args = args[i + 1:]
            break

    return entry_point, class_name, spark_conf, app_args


def _make_step(step_config, cluster_id=None):
    """Build an EMR step record from a step config.

    If spark config is present (k8s mode), creates a real K8s Job for the step.
    Otherwise, the step is immediately marked COMPLETED (mock mode).
    """
    now = _now_iso()
    step_id = _step_id()
    hadoop_jar = step_config.get("HadoopJarStep", {})
    entry_point = hadoop_jar.get("Jar", "")
    class_name = hadoop_jar.get("MainClass", "")
    args = hadoop_jar.get("Args", [])
    properties = hadoop_jar.get("Properties", [])

    k8s_job_name = None
    if k8s_spark.is_k8s_mode():
        # Real execution: create K8s Job
        effective_class = class_name
        effective_entry = entry_point
        effective_args = list(args)

        # Handle command-runner.jar pattern: EMR wraps spark-submit as
        #   Jar: command-runner.jar
        #   Args: [spark-submit, --class, Foo, --master, yarn, --deploy-mode, cluster, --conf, k=v, s3://app.jar, app-arg1, ...]
        # We need to parse the inner spark-submit command line to extract
        # the real entry point, class, confs, and application args.
        if entry_point.endswith("command-runner.jar") and effective_args and effective_args[0] == "spark-submit":
            effective_entry, effective_class, spark_conf_from_args, effective_args = \
                _parse_spark_submit_args(effective_args[1:])  # skip "spark-submit"
            # Merge any --conf from args into properties-based spark_conf
            if spark_conf_from_args:
                properties_conf = {p["Key"]: p["Value"] for p in properties} if properties else {}
                properties_conf.update(spark_conf_from_args)
                properties = [{"Key": k, "Value": v} for k, v in properties_conf.items()]
        else:
            # Parse --class from Args if MainClass not set (EMR CLI puts --class in Args)
            if not effective_class and "--class" in effective_args:
                idx = effective_args.index("--class")
                if idx + 1 < len(effective_args):
                    effective_class = effective_args[idx + 1]
                    effective_args = effective_args[:idx] + effective_args[idx + 2:]

        # Convert Properties list to spark conf dict
        spark_conf = {p["Key"]: p["Value"] for p in properties} if properties else {}

        k8s_job_name = f"emr-step-{step_id.lower()}"
        k8s_spark.create_spark_job(
            job_name=k8s_job_name,
            entry_point=effective_entry,
            class_name=effective_class,
            spark_args=effective_args if effective_args else None,
            spark_conf=spark_conf if spark_conf else None,
            labels={
                "ministack/service": "emr-ec2",
                "ministack/step": step_id,
                "ministack/cluster": cluster_id or "",
            },
        )
        initial_state = "RUNNING"
        end_time = None
    else:
        initial_state = "COMPLETED"
        end_time = now

    return {
        "Id": step_id,
        "Name": step_config.get("Name", ""),
        "Config": {
            "Jar": entry_point,
            "Properties": {p["Key"]: p["Value"] for p in properties} if properties else {},
            "MainClass": class_name,
            "Args": args,
        },
        "ActionOnFailure": step_config.get("ActionOnFailure", "CONTINUE"),
        "Status": {
            "State": initial_state,
            "StateChangeReason": {},
            "Timeline": {"CreationDateTime": now, "StartDateTime": now, "EndDateTime": end_time},
        },
        "_k8s_job_name": k8s_job_name,
    }


def _add_job_flow_steps(data):
    cluster_id = data.get("JobFlowId")
    if cluster_id not in _clusters:
        return error_response_json("InvalidRequestException",
                                   f"Cluster id '{cluster_id}' is not valid.", 400)
    step_ids = []
    for step_config in data.get("Steps", []):
        step = _make_step(step_config, cluster_id=cluster_id)
        _steps.setdefault(cluster_id, []).append(step)
        step_ids.append(step["Id"])
    return json_response({"StepIds": step_ids})


def _describe_step(data):
    cluster_id = data.get("ClusterId")
    step_id = data.get("StepId")
    for step in _steps.get(cluster_id, []):
        if step["Id"] == step_id:
            # In k8s mode, update state from K8s before returning
            k8s_job_name = step.get("_k8s_job_name")
            if k8s_job_name and step["Status"]["State"] not in ("COMPLETED", "FAILED", "CANCELLED"):
                k8s_state = k8s_spark.get_job_state(k8s_job_name)
                step["Status"]["State"] = k8s_state["state"]
                if k8s_state["state"] in ("COMPLETED", "FAILED", "CANCELLED"):
                    step["Status"]["Timeline"]["EndDateTime"] = _now_iso()
            return json_response({"Step": step})
    return error_response_json("InvalidRequestException",
                               f"Step id '{step_id}' is not valid.", 400)


def _list_steps(data):
    cluster_id = data.get("ClusterId")
    state_filter = data.get("StepStates", [])
    steps = _steps.get(cluster_id, [])
    if state_filter:
        steps = [s for s in steps if s["Status"]["State"] in state_filter]
    return json_response({"Steps": steps})


def _cancel_steps(data):
    cluster_id = data.get("ClusterId")
    step_ids = data.get("StepIds", [])
    cancelled = []
    for step in _steps.get(cluster_id, []):
        if step["Id"] in step_ids and step["Status"]["State"] in ("PENDING", "RUNNING"):
            # In k8s mode, delete the K8s Job
            k8s_job_name = step.get("_k8s_job_name")
            if k8s_job_name:
                k8s_spark.delete_job(k8s_job_name)
            step["Status"]["State"] = "CANCELLED"
            cancelled.append({"StepId": step["Id"], "Status": "SUBMITTED"})
        elif step["Id"] in step_ids:
            cancelled.append({"StepId": step["Id"], "Status": "FAILED_TO_CANCEL",
                               "Reason": f"Step in state {step['Status']['State']} cannot be cancelled"})
    return json_response({"CancelStepsInfoList": cancelled})


# ---------------------------------------------------------------------------
# Instance Fleets
# ---------------------------------------------------------------------------

def _add_instance_fleet(data):
    cluster_id = data.get("ClusterId")
    cluster = _clusters.get(cluster_id)
    if not cluster:
        return error_response_json("InvalidRequestException",
                                   f"Cluster id '{cluster_id}' is not valid.", 400)
    fleet = data.get("InstanceFleet", {})
    now = _now_iso()
    fleet_id = _fleet_id()
    record = {
        "Id": fleet_id,
        "Name": fleet.get("Name", fleet.get("InstanceFleetType", "TASK")),
        "Status": {"State": "RUNNING", "StateChangeReason": {}, "Timeline": {"CreationDateTime": now}},
        "InstanceFleetType": fleet.get("InstanceFleetType", "TASK"),
        "TargetOnDemandCapacity": fleet.get("TargetOnDemandCapacity", 0),
        "TargetSpotCapacity": fleet.get("TargetSpotCapacity", 0),
        "ProvisionedOnDemandCapacity": fleet.get("TargetOnDemandCapacity", 0),
        "ProvisionedSpotCapacity": fleet.get("TargetSpotCapacity", 0),
        "InstanceTypeSpecifications": fleet.get("InstanceTypeConfigs", []),
    }
    cluster["InstanceFleets"].append(record)
    return json_response({"ClusterArn": cluster["ClusterArn"], "InstanceFleetId": fleet_id})


def _list_instance_fleets(data):
    cluster_id = data.get("ClusterId")
    cluster = _clusters.get(cluster_id)
    if not cluster:
        return error_response_json("InvalidRequestException",
                                   f"Cluster id '{cluster_id}' is not valid.", 400)
    return json_response({"InstanceFleets": cluster.get("InstanceFleets", [])})


def _modify_instance_fleet(data):
    cluster_id = data.get("ClusterId")
    cluster = _clusters.get(cluster_id)
    if not cluster:
        return error_response_json("InvalidRequestException",
                                   f"Cluster id '{cluster_id}' is not valid.", 400)
    fleet_mod = data.get("InstanceFleet", {})
    fleet_id = fleet_mod.get("InstanceFleetId")
    for fleet in cluster.get("InstanceFleets", []):
        if fleet["Id"] == fleet_id:
            if "TargetOnDemandCapacity" in fleet_mod:
                fleet["TargetOnDemandCapacity"] = fleet_mod["TargetOnDemandCapacity"]
                fleet["ProvisionedOnDemandCapacity"] = fleet_mod["TargetOnDemandCapacity"]
            if "TargetSpotCapacity" in fleet_mod:
                fleet["TargetSpotCapacity"] = fleet_mod["TargetSpotCapacity"]
                fleet["ProvisionedSpotCapacity"] = fleet_mod["TargetSpotCapacity"]
            break
    return json_response({})


# ---------------------------------------------------------------------------
# Instance Groups
# ---------------------------------------------------------------------------

def _add_instance_groups(data):
    cluster_id = data.get("JobFlowId")
    cluster = _clusters.get(cluster_id)
    if not cluster:
        return error_response_json("InvalidRequestException",
                                   f"Cluster id '{cluster_id}' is not valid.", 400)
    now = _now_iso()
    group_ids = []
    for ig in data.get("InstanceGroups", []):
        gid = _group_id()
        record = {
            "Id": gid,
            "Name": ig.get("Name", ig.get("InstanceRole", "TASK")),
            "Market": ig.get("Market", "ON_DEMAND"),
            "InstanceGroupType": ig.get("InstanceRole", "TASK"),
            "InstanceType": ig.get("InstanceType", "m5.xlarge"),
            "RequestedInstanceCount": ig.get("InstanceCount", 1),
            "RunningInstanceCount": ig.get("InstanceCount", 1),
            "Status": {"State": "RUNNING", "StateChangeReason": {}, "Timeline": {"CreationDateTime": now}},
        }
        cluster["InstanceGroups"].append(record)
        group_ids.append(gid)
    return json_response({"JobFlowId": cluster_id, "InstanceGroupIds": group_ids})


def _list_instance_groups(data):
    cluster_id = data.get("ClusterId")
    cluster = _clusters.get(cluster_id)
    if not cluster:
        return error_response_json("InvalidRequestException",
                                   f"Cluster id '{cluster_id}' is not valid.", 400)
    return json_response({"InstanceGroups": cluster.get("InstanceGroups", [])})


def _modify_instance_groups(data):
    cluster_id = data.get("ClusterId")
    cluster = _clusters.get(cluster_id)
    if not cluster:
        return error_response_json("InvalidRequestException",
                                   f"Cluster id '{cluster_id}' is not valid.", 400)
    for mod in data.get("InstanceGroups", []):
        gid = mod.get("InstanceGroupId")
        for ig in cluster.get("InstanceGroups", []):
            if ig["Id"] == gid:
                if "InstanceCount" in mod:
                    ig["RequestedInstanceCount"] = mod["InstanceCount"]
                    ig["RunningInstanceCount"] = mod["InstanceCount"]
                break
    return json_response({})


# ---------------------------------------------------------------------------
# Bootstrap Actions
# ---------------------------------------------------------------------------

def _list_bootstrap_actions(data):
    cluster_id = data.get("ClusterId")
    cluster = _clusters.get(cluster_id)
    if not cluster:
        return error_response_json("InvalidRequestException",
                                   f"Cluster id '{cluster_id}' is not valid.", 400)
    actions = [
        {
            "Name": ba.get("Name", ""),
            "ScriptPath": ba.get("ScriptBootstrapAction", {}).get("Path", ""),
            "Args": ba.get("ScriptBootstrapAction", {}).get("Args", []),
        }
        for ba in cluster.get("BootstrapActions", [])
    ]
    return json_response({"BootstrapActions": actions})


# ---------------------------------------------------------------------------
# Tags
# ---------------------------------------------------------------------------

def _add_tags(data):
    resource_id = data.get("ResourceId")
    cluster = _clusters.get(resource_id)
    if not cluster:
        return error_response_json("InvalidRequestException",
                                   f"Resource id '{resource_id}' is not valid.", 400)
    new_tags = data.get("Tags", [])
    existing = {t["Key"]: i for i, t in enumerate(cluster["Tags"])}
    for tag in new_tags:
        idx = existing.get(tag["Key"])
        if idx is not None:
            cluster["Tags"][idx] = tag
        else:
            cluster["Tags"].append(tag)
            existing[tag["Key"]] = len(cluster["Tags"]) - 1
    return json_response({})


def _remove_tags(data):
    resource_id = data.get("ResourceId")
    cluster = _clusters.get(resource_id)
    if not cluster:
        return error_response_json("InvalidRequestException",
                                   f"Resource id '{resource_id}' is not valid.", 400)
    keys = set(data.get("TagKeys", []))
    cluster["Tags"] = [t for t in cluster["Tags"] if t["Key"] not in keys]
    return json_response({})


# ---------------------------------------------------------------------------
# Block Public Access
# ---------------------------------------------------------------------------

def _get_block_public_access_configuration(data):
    return json_response({
        "BlockPublicAccessConfiguration": _block_public_access,
        "BlockPublicAccessConfigurationMetadata": {
            "CreationDateTime": _now_iso(),
            "CreatedByArn": f"arn:aws:iam::{ACCOUNT_ID}:root",
        },
    })


def _put_block_public_access_configuration(data):
    config = data.get("BlockPublicAccessConfiguration", {})
    _block_public_access["BlockPublicSecurityGroupRules"] = config.get(
        "BlockPublicSecurityGroupRules", False
    )
    _block_public_access["PermittedPublicSecurityGroupRuleRanges"] = config.get(
        "PermittedPublicSecurityGroupRuleRanges", []
    )
    return json_response({})


# ---------------------------------------------------------------------------
# Request routing
# ---------------------------------------------------------------------------

_HANDLERS = {
    "RunJobFlow": _run_job_flow,
    "DescribeCluster": _describe_cluster,
    "ListClusters": _list_clusters,
    "TerminateJobFlows": _terminate_job_flows,
    "ModifyCluster": _modify_cluster,
    "SetTerminationProtection": _set_termination_protection,
    "SetVisibleToAllUsers": _set_visible_to_all_users,
    "AddJobFlowSteps": _add_job_flow_steps,
    "DescribeStep": _describe_step,
    "ListSteps": _list_steps,
    "CancelSteps": _cancel_steps,
    "AddInstanceFleet": _add_instance_fleet,
    "ListInstanceFleets": _list_instance_fleets,
    "ModifyInstanceFleet": _modify_instance_fleet,
    "AddInstanceGroups": _add_instance_groups,
    "ListInstanceGroups": _list_instance_groups,
    "ModifyInstanceGroups": _modify_instance_groups,
    "ListBootstrapActions": _list_bootstrap_actions,
    "AddTags": _add_tags,
    "RemoveTags": _remove_tags,
    "GetBlockPublicAccessConfiguration": _get_block_public_access_configuration,
    "PutBlockPublicAccessConfiguration": _put_block_public_access_configuration,
}


async def handle_request(method, path, headers, body, query_params):
    target = headers.get("x-amz-target", "")
    action = target.split(".")[-1] if "." in target else ""

    try:
        data = json.loads(body) if body else {}
    except json.JSONDecodeError:
        return error_response_json("SerializationException", "Invalid JSON", 400)

    handler = _HANDLERS.get(action)
    if not handler:
        return error_response_json("InvalidAction", f"Unknown EMR action: {action}", 400)
    return handler(data)


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

def reset():
    _clusters.clear()
    _steps.clear()
    _block_public_access["BlockPublicSecurityGroupRules"] = False
    _block_public_access["PermittedPublicSecurityGroupRuleRanges"] = []
