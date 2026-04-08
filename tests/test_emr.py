import io
import json
import os
import time
import zipfile
from urllib.parse import urlparse
import pytest
from botocore.exceptions import ClientError
import uuid as _uuid_mod

def test_emr_run_job_flow_simple(emr):
    resp = emr.run_job_flow(
        Name="test-cluster",
        ReleaseLabel="emr-6.10.0",
        Instances={
            "MasterInstanceType": "m5.xlarge",
            "SlaveInstanceType": "m5.xlarge",
            "InstanceCount": 3,
            "KeepJobFlowAliveWhenNoSteps": True,
        },
        JobFlowRole="EMR_EC2_DefaultRole",
        ServiceRole="EMR_DefaultRole",
    )
    assert resp["JobFlowId"].startswith("j-")
    assert "ClusterArn" in resp
    assert "elasticmapreduce" in resp["ClusterArn"]

def test_emr_describe_cluster(emr):
    jf = emr.run_job_flow(
        Name="describe-test",
        ReleaseLabel="emr-6.10.0",
        Instances={
            "MasterInstanceType": "m5.xlarge",
            "InstanceCount": 1,
            "KeepJobFlowAliveWhenNoSteps": True,
        },
        JobFlowRole="EMR_EC2_DefaultRole",
        ServiceRole="EMR_DefaultRole",
    )
    cluster_id = jf["JobFlowId"]
    desc = emr.describe_cluster(ClusterId=cluster_id)
    cluster = desc["Cluster"]
    assert cluster["Id"] == cluster_id
    assert cluster["Name"] == "describe-test"
    assert cluster["Status"]["State"] == "WAITING"
    assert cluster["ReleaseLabel"] == "emr-6.10.0"

def test_emr_list_clusters(emr):
    emr.run_job_flow(
        Name="list-test",
        ReleaseLabel="emr-6.10.0",
        Instances={
            "MasterInstanceType": "m5.xlarge",
            "InstanceCount": 1,
            "KeepJobFlowAliveWhenNoSteps": True,
        },
        JobFlowRole="EMR_EC2_DefaultRole",
        ServiceRole="EMR_DefaultRole",
    )
    resp = emr.list_clusters()
    assert len(resp["Clusters"]) >= 1
    assert all("Id" in c for c in resp["Clusters"])

def test_emr_terminate_job_flows(emr):
    jf = emr.run_job_flow(
        Name="terminate-test",
        ReleaseLabel="emr-6.10.0",
        Instances={
            "MasterInstanceType": "m5.xlarge",
            "InstanceCount": 1,
            "KeepJobFlowAliveWhenNoSteps": True,
        },
        JobFlowRole="EMR_EC2_DefaultRole",
        ServiceRole="EMR_DefaultRole",
    )
    cluster_id = jf["JobFlowId"]
    emr.terminate_job_flows(JobFlowIds=[cluster_id])
    desc = emr.describe_cluster(ClusterId=cluster_id)
    assert desc["Cluster"]["Status"]["State"] == "TERMINATED"

def test_emr_termination_protection(emr):
    jf = emr.run_job_flow(
        Name="protected-cluster",
        ReleaseLabel="emr-6.10.0",
        Instances={
            "MasterInstanceType": "m5.xlarge",
            "InstanceCount": 1,
            "KeepJobFlowAliveWhenNoSteps": True,
            "TerminationProtected": True,
        },
        JobFlowRole="EMR_EC2_DefaultRole",
        ServiceRole="EMR_DefaultRole",
    )
    cluster_id = jf["JobFlowId"]
    import botocore.exceptions

    try:
        emr.terminate_job_flows(JobFlowIds=[cluster_id])
        assert False, "should have raised"
    except botocore.exceptions.ClientError as e:
        assert "ValidationException" in str(e) or "protected" in str(e).lower()

def test_emr_add_and_list_steps(emr):
    jf = emr.run_job_flow(
        Name="steps-cluster",
        ReleaseLabel="emr-6.10.0",
        Instances={
            "MasterInstanceType": "m5.xlarge",
            "InstanceCount": 1,
            "KeepJobFlowAliveWhenNoSteps": True,
        },
        JobFlowRole="EMR_EC2_DefaultRole",
        ServiceRole="EMR_DefaultRole",
    )
    cluster_id = jf["JobFlowId"]
    step_resp = emr.add_job_flow_steps(
        JobFlowId=cluster_id,
        Steps=[
            {
                "Name": "my-spark-step",
                "ActionOnFailure": "CONTINUE",
                "HadoopJarStep": {
                    "Jar": "command-runner.jar",
                    "Args": [
                        "spark-submit",
                        "--class",
                        "com.example.Main",
                        "s3://bucket/app.jar",
                    ],
                },
            }
        ],
    )
    assert len(step_resp["StepIds"]) == 1
    step_id = step_resp["StepIds"][0]
    assert step_id.startswith("s-")

    steps = emr.list_steps(ClusterId=cluster_id)
    assert any(s["Id"] == step_id for s in steps["Steps"])

def test_emr_describe_step(emr):
    jf = emr.run_job_flow(
        Name="describe-step-cluster",
        ReleaseLabel="emr-6.10.0",
        Instances={
            "MasterInstanceType": "m5.xlarge",
            "InstanceCount": 1,
            "KeepJobFlowAliveWhenNoSteps": True,
        },
        JobFlowRole="EMR_EC2_DefaultRole",
        ServiceRole="EMR_DefaultRole",
    )
    cluster_id = jf["JobFlowId"]
    step_resp = emr.add_job_flow_steps(
        JobFlowId=cluster_id,
        Steps=[
            {
                "Name": "step1",
                "ActionOnFailure": "CONTINUE",
                "HadoopJarStep": {"Jar": "command-runner.jar", "Args": []},
            }
        ],
    )
    step_id = step_resp["StepIds"][0]
    desc = emr.describe_step(ClusterId=cluster_id, StepId=step_id)
    assert desc["Step"]["Id"] == step_id
    assert desc["Step"]["Status"]["State"] == "COMPLETED"

def test_emr_tags(emr):
    jf = emr.run_job_flow(
        Name="tagged-cluster",
        ReleaseLabel="emr-6.10.0",
        Instances={
            "MasterInstanceType": "m5.xlarge",
            "InstanceCount": 1,
            "KeepJobFlowAliveWhenNoSteps": True,
        },
        JobFlowRole="EMR_EC2_DefaultRole",
        ServiceRole="EMR_DefaultRole",
        Tags=[{"Key": "env", "Value": "test"}],
    )
    cluster_id = jf["JobFlowId"]
    emr.add_tags(ResourceId=cluster_id, Tags=[{"Key": "team", "Value": "data"}])
    desc = emr.describe_cluster(ClusterId=cluster_id)
    tag_map = {t["Key"]: t["Value"] for t in desc["Cluster"]["Tags"]}
    assert tag_map["env"] == "test"
    assert tag_map["team"] == "data"

    emr.remove_tags(ResourceId=cluster_id, TagKeys=["env"])
    desc2 = emr.describe_cluster(ClusterId=cluster_id)
    tag_keys = [t["Key"] for t in desc2["Cluster"]["Tags"]]
    assert "env" not in tag_keys
    assert "team" in tag_keys

def test_emr_auto_terminate_state(emr):
    """Cluster with KeepJobFlowAliveWhenNoSteps=False starts as TERMINATED."""
    jf = emr.run_job_flow(
        Name="auto-terminate-cluster",
        ReleaseLabel="emr-6.10.0",
        Instances={
            "MasterInstanceType": "m5.xlarge",
            "InstanceCount": 1,
            "KeepJobFlowAliveWhenNoSteps": False,
        },
        JobFlowRole="EMR_EC2_DefaultRole",
        ServiceRole="EMR_DefaultRole",
    )
    cluster_id = jf["JobFlowId"]
    desc = emr.describe_cluster(ClusterId=cluster_id)
    assert desc["Cluster"]["Status"]["State"] == "TERMINATED"
    assert desc["Cluster"]["AutoTerminate"] is True

def test_emr_modify_cluster(emr):
    jf = emr.run_job_flow(
        Name="modify-cluster",
        ReleaseLabel="emr-6.10.0",
        Instances={
            "MasterInstanceType": "m5.xlarge",
            "InstanceCount": 1,
            "KeepJobFlowAliveWhenNoSteps": True,
        },
        JobFlowRole="EMR_EC2_DefaultRole",
        ServiceRole="EMR_DefaultRole",
    )
    cluster_id = jf["JobFlowId"]
    resp = emr.modify_cluster(ClusterId=cluster_id, StepConcurrencyLevel=5)
    assert resp["StepConcurrencyLevel"] == 5

def test_emr_block_public_access(emr):
    resp = emr.get_block_public_access_configuration()
    assert "BlockPublicAccessConfiguration" in resp
    assert resp["BlockPublicAccessConfiguration"]["BlockPublicSecurityGroupRules"] is False

    emr.put_block_public_access_configuration(
        BlockPublicAccessConfiguration={
            "BlockPublicSecurityGroupRules": True,
            "PermittedPublicSecurityGroupRuleRanges": [{"MinRange": 22, "MaxRange": 22}],
        }
    )
    resp2 = emr.get_block_public_access_configuration()
    assert resp2["BlockPublicAccessConfiguration"]["BlockPublicSecurityGroupRules"] is True

def test_emr_instance_groups(emr):
    jf = emr.run_job_flow(
        Name="ig-cluster",
        ReleaseLabel="emr-6.10.0",
        Instances={
            "InstanceGroups": [
                {
                    "Name": "Master",
                    "InstanceRole": "MASTER",
                    "InstanceType": "m5.xlarge",
                    "InstanceCount": 1,
                },
                {
                    "Name": "Core",
                    "InstanceRole": "CORE",
                    "InstanceType": "m5.xlarge",
                    "InstanceCount": 2,
                },
            ],
            "KeepJobFlowAliveWhenNoSteps": True,
        },
        JobFlowRole="EMR_EC2_DefaultRole",
        ServiceRole="EMR_DefaultRole",
    )
    cluster_id = jf["JobFlowId"]
    groups = emr.list_instance_groups(ClusterId=cluster_id)
    assert len(groups["InstanceGroups"]) >= 2

    new_group_resp = emr.add_instance_groups(
        JobFlowId=cluster_id,
        InstanceGroups=[
            {
                "Name": "Task",
                "InstanceRole": "TASK",
                "InstanceType": "m5.xlarge",
                "InstanceCount": 2,
            }
        ],
    )
    assert len(new_group_resp["InstanceGroupIds"]) == 1
    groups2 = emr.list_instance_groups(ClusterId=cluster_id)
    assert len(groups2["InstanceGroups"]) == 3

def test_emr_instance_fleets(emr):
    """AddInstanceFleet / ListInstanceFleets / ModifyInstanceFleet."""
    resp = emr.run_job_flow(
        Name="fleet-test-v44",
        ReleaseLabel="emr-6.15.0",
        Instances={
            "KeepJobFlowAliveWhenNoSteps": True,
            "InstanceFleets": [
                {"InstanceFleetType": "MASTER", "Name": "master-fleet",
                 "TargetOnDemandCapacity": 1,
                 "InstanceTypeConfigs": [{"InstanceType": "m5.xlarge"}]},
            ],
        },
        JobFlowRole="EMR_EC2_DefaultRole",
        ServiceRole="EMR_DefaultRole",
    )
    cluster_id = resp["JobFlowId"]

    # Add a CORE fleet
    add_resp = emr.add_instance_fleet(
        ClusterId=cluster_id,
        InstanceFleet={
            "InstanceFleetType": "CORE", "Name": "core-fleet",
            "TargetOnDemandCapacity": 2,
            "InstanceTypeConfigs": [{"InstanceType": "m5.xlarge"}],
        },
    )
    fleet_id = add_resp["InstanceFleetId"]
    assert fleet_id

    # List fleets
    fleets = emr.list_instance_fleets(ClusterId=cluster_id)
    fleet_types = [f["InstanceFleetType"] for f in fleets["InstanceFleets"]]
    assert "MASTER" in fleet_types
    assert "CORE" in fleet_types

    emr.terminate_job_flows(JobFlowIds=[cluster_id])
