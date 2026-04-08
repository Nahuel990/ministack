import io
import json
import os
import time
import zipfile
from urllib.parse import urlparse
import pytest
from botocore.exceptions import ClientError
import uuid as _uuid_mod

def test_glue_catalog(glue):
    glue.create_database(DatabaseInput={"Name": "test_db", "Description": "Test database"})
    glue.create_table(
        DatabaseName="test_db",
        TableInput={
            "Name": "test_table",
            "StorageDescriptor": {
                "Columns": [
                    {"Name": "id", "Type": "int"},
                    {"Name": "name", "Type": "string"},
                ],
                "Location": "s3://my-bucket/data/",
                "InputFormat": "org.apache.hadoop.mapred.TextInputFormat",
                "OutputFormat": "org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat",
                "SerdeInfo": {"SerializationLibrary": "org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe"},
            },
            "TableType": "EXTERNAL_TABLE",
        },
    )
    resp = glue.get_table(DatabaseName="test_db", Name="test_table")
    assert resp["Table"]["Name"] == "test_table"

def test_glue_list(glue):
    dbs = glue.get_databases()
    assert any(d["Name"] == "test_db" for d in dbs["DatabaseList"])
    tables = glue.get_tables(DatabaseName="test_db")
    assert any(t["Name"] == "test_table" for t in tables["TableList"])

def test_glue_job(glue):
    glue.create_job(
        Name="test-job",
        Role="arn:aws:iam::000000000000:role/GlueRole",
        Command={"Name": "glueetl", "ScriptLocation": "s3://my-bucket/scripts/etl.py"},
        GlueVersion="3.0",
    )
    resp = glue.start_job_run(JobName="test-job")
    assert "JobRunId" in resp
    runs = glue.get_job_runs(JobName="test-job")
    assert len(runs["JobRuns"]) == 1

def test_glue_crawler(glue):
    glue.create_crawler(
        Name="test-crawler",
        Role="arn:aws:iam::000000000000:role/GlueRole",
        DatabaseName="test_db",
        Targets={"S3Targets": [{"Path": "s3://my-bucket/data/"}]},
    )
    resp = glue.get_crawler(Name="test-crawler")
    assert resp["Crawler"]["Name"] == "test-crawler"
    glue.start_crawler(Name="test-crawler")

def test_glue_database_v2(glue):
    glue.create_database(DatabaseInput={"Name": "glue_db_v2", "Description": "v2 DB"})
    resp = glue.get_database(Name="glue_db_v2")
    assert resp["Database"]["Name"] == "glue_db_v2"
    assert resp["Database"]["Description"] == "v2 DB"

    glue.update_database(
        Name="glue_db_v2",
        DatabaseInput={"Name": "glue_db_v2", "Description": "updated"},
    )
    resp2 = glue.get_database(Name="glue_db_v2")
    assert resp2["Database"]["Description"] == "updated"

    glue.delete_database(Name="glue_db_v2")
    with pytest.raises(ClientError) as exc:
        glue.get_database(Name="glue_db_v2")
    assert exc.value.response["Error"]["Code"] == "EntityNotFoundException"

def test_glue_table_v2(glue):
    glue.create_database(DatabaseInput={"Name": "glue_tbl_v2db"})
    glue.create_table(
        DatabaseName="glue_tbl_v2db",
        TableInput={
            "Name": "tbl_v2",
            "StorageDescriptor": {
                "Columns": [
                    {"Name": "id", "Type": "int"},
                    {"Name": "name", "Type": "string"},
                ],
                "Location": "s3://bucket/tbl_v2/",
                "InputFormat": "org.apache.hadoop.mapred.TextInputFormat",
                "OutputFormat": "org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat",
                "SerdeInfo": {"SerializationLibrary": "org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe"},
            },
            "TableType": "EXTERNAL_TABLE",
        },
    )
    resp = glue.get_table(DatabaseName="glue_tbl_v2db", Name="tbl_v2")
    assert resp["Table"]["Name"] == "tbl_v2"
    assert len(resp["Table"]["StorageDescriptor"]["Columns"]) == 2

    glue.update_table(
        DatabaseName="glue_tbl_v2db",
        TableInput={"Name": "tbl_v2", "Description": "updated table"},
    )
    resp2 = glue.get_table(DatabaseName="glue_tbl_v2db", Name="tbl_v2")
    assert resp2["Table"]["Description"] == "updated table"

    glue.delete_table(DatabaseName="glue_tbl_v2db", Name="tbl_v2")
    with pytest.raises(ClientError) as exc:
        glue.get_table(DatabaseName="glue_tbl_v2db", Name="tbl_v2")
    assert exc.value.response["Error"]["Code"] == "EntityNotFoundException"

def test_glue_list_v2(glue):
    glue.create_database(DatabaseInput={"Name": "glue_lst_v2db"})
    glue.create_table(
        DatabaseName="glue_lst_v2db",
        TableInput={
            "Name": "lt_a",
            "StorageDescriptor": {
                "Columns": [{"Name": "c", "Type": "string"}],
                "Location": "s3://b/lt_a/",
                "InputFormat": "TIF",
                "OutputFormat": "TOF",
                "SerdeInfo": {"SerializationLibrary": "SL"},
            },
        },
    )
    glue.create_table(
        DatabaseName="glue_lst_v2db",
        TableInput={
            "Name": "lt_b",
            "StorageDescriptor": {
                "Columns": [{"Name": "c", "Type": "string"}],
                "Location": "s3://b/lt_b/",
                "InputFormat": "TIF",
                "OutputFormat": "TOF",
                "SerdeInfo": {"SerializationLibrary": "SL"},
            },
        },
    )
    dbs = glue.get_databases()
    assert any(d["Name"] == "glue_lst_v2db" for d in dbs["DatabaseList"])
    tables = glue.get_tables(DatabaseName="glue_lst_v2db")
    names = [t["Name"] for t in tables["TableList"]]
    assert "lt_a" in names
    assert "lt_b" in names

def test_glue_job_v2(glue):
    glue.create_job(
        Name="glue-job-v2",
        Role="arn:aws:iam::000000000000:role/R",
        Command={"Name": "glueetl", "ScriptLocation": "s3://b/s.py"},
        GlueVersion="3.0",
    )
    job = glue.get_job(JobName="glue-job-v2")["Job"]
    assert job["Name"] == "glue-job-v2"

    run_resp = glue.start_job_run(JobName="glue-job-v2", Arguments={"--key": "val"})
    run_id = run_resp["JobRunId"]
    assert run_id

    run = glue.get_job_run(JobName="glue-job-v2", RunId=run_id)["JobRun"]
    assert run["Id"] == run_id
    assert run["JobName"] == "glue-job-v2"

    runs = glue.get_job_runs(JobName="glue-job-v2")["JobRuns"]
    assert any(r["Id"] == run_id for r in runs)

def test_glue_crawler_v2(glue):
    glue.create_database(DatabaseInput={"Name": "glue_cr_v2db"})
    glue.create_crawler(
        Name="glue-cr-v2",
        Role="arn:aws:iam::000000000000:role/R",
        DatabaseName="glue_cr_v2db",
        Targets={"S3Targets": [{"Path": "s3://b/data/"}]},
    )
    cr = glue.get_crawler(Name="glue-cr-v2")["Crawler"]
    assert cr["Name"] == "glue-cr-v2"
    assert cr["State"] == "READY"

    glue.start_crawler(Name="glue-cr-v2")
    cr2 = glue.get_crawler(Name="glue-cr-v2")["Crawler"]
    assert cr2["State"] == "RUNNING"

def test_glue_tags_v2(glue):
    glue.create_database(DatabaseInput={"Name": "glue_tag_v2db"})
    arn = "arn:aws:glue:us-east-1:000000000000:database/glue_tag_v2db"
    glue.tag_resource(ResourceArn=arn, TagsToAdd={"env": "test", "team": "data"})
    resp = glue.get_tags(ResourceArn=arn)
    assert resp["Tags"]["env"] == "test"
    assert resp["Tags"]["team"] == "data"

    glue.untag_resource(ResourceArn=arn, TagsToRemove=["team"])
    resp2 = glue.get_tags(ResourceArn=arn)
    assert resp2["Tags"] == {"env": "test"}

def test_glue_partition_v2(glue):
    glue.create_database(DatabaseInput={"Name": "glue_part_v2db"})
    glue.create_table(
        DatabaseName="glue_part_v2db",
        TableInput={
            "Name": "ptbl_v2",
            "StorageDescriptor": {
                "Columns": [{"Name": "data", "Type": "string"}],
                "Location": "s3://b/pt/",
                "InputFormat": "TIF",
                "OutputFormat": "TOF",
                "SerdeInfo": {"SerializationLibrary": "SL"},
            },
            "PartitionKeys": [
                {"Name": "year", "Type": "string"},
                {"Name": "month", "Type": "string"},
            ],
        },
    )
    glue.create_partition(
        DatabaseName="glue_part_v2db",
        TableName="ptbl_v2",
        PartitionInput={
            "Values": ["2024", "01"],
            "StorageDescriptor": {
                "Columns": [{"Name": "data", "Type": "string"}],
                "Location": "s3://b/pt/year=2024/month=01/",
                "InputFormat": "TIF",
                "OutputFormat": "TOF",
                "SerdeInfo": {"SerializationLibrary": "SL"},
            },
        },
    )
    glue.create_partition(
        DatabaseName="glue_part_v2db",
        TableName="ptbl_v2",
        PartitionInput={
            "Values": ["2024", "02"],
            "StorageDescriptor": {
                "Columns": [{"Name": "data", "Type": "string"}],
                "Location": "s3://b/pt/year=2024/month=02/",
                "InputFormat": "TIF",
                "OutputFormat": "TOF",
                "SerdeInfo": {"SerializationLibrary": "SL"},
            },
        },
    )
    resp = glue.get_partition(
        DatabaseName="glue_part_v2db",
        TableName="ptbl_v2",
        PartitionValues=["2024", "01"],
    )
    assert resp["Partition"]["Values"] == ["2024", "01"]

    parts = glue.get_partitions(DatabaseName="glue_part_v2db", TableName="ptbl_v2")
    assert len(parts["Partitions"]) == 2

def test_glue_connection_v2(glue):
    glue.create_connection(
        ConnectionInput={
            "Name": "glue-conn-v2",
            "ConnectionType": "JDBC",
            "ConnectionProperties": {
                "JDBC_CONNECTION_URL": "jdbc:postgresql://host/db",
                "USERNAME": "user",
                "PASSWORD": "pass",
            },
        }
    )
    resp = glue.get_connection(Name="glue-conn-v2")
    assert resp["Connection"]["Name"] == "glue-conn-v2"
    assert resp["Connection"]["ConnectionType"] == "JDBC"

    conns = glue.get_connections()
    assert any(c["Name"] == "glue-conn-v2" for c in conns["ConnectionList"])

    glue.delete_connection(ConnectionName="glue-conn-v2")
    with pytest.raises(ClientError) as exc:
        glue.get_connection(Name="glue-conn-v2")
    assert exc.value.response["Error"]["Code"] == "EntityNotFoundException"

def test_glue_trigger(glue):
    glue.create_trigger(Name="test-trig", Type="ON_DEMAND", Actions=[{"JobName": "nonexistent-job"}])
    resp = glue.get_trigger(Name="test-trig")
    assert resp["Trigger"]["Name"] == "test-trig"
    assert resp["Trigger"]["State"] == "CREATED"
    glue.start_trigger(Name="test-trig")
    resp2 = glue.get_trigger(Name="test-trig")
    assert resp2["Trigger"]["State"] == "ACTIVATED"
    glue.stop_trigger(Name="test-trig")
    resp3 = glue.get_trigger(Name="test-trig")
    assert resp3["Trigger"]["State"] == "DEACTIVATED"
    glue.delete_trigger(Name="test-trig")

def test_glue_workflow(glue):
    glue.create_workflow(Name="test-wf", Description="Test workflow")
    resp = glue.get_workflow(Name="test-wf")
    assert resp["Workflow"]["Name"] == "test-wf"
    run = glue.start_workflow_run(Name="test-wf")
    assert "RunId" in run
    glue.delete_workflow(Name="test-wf")

def test_glue_partition_crud(glue):
    """CreatePartition / GetPartition / GetPartitions / DeletePartition."""
    glue.create_database(DatabaseInput={"Name": "qa-glue-partdb"})
    glue.create_table(
        DatabaseName="qa-glue-partdb",
        TableInput={
            "Name": "qa-glue-parttbl",
            "StorageDescriptor": {
                "Columns": [],
                "Location": "s3://bucket/key",
                "InputFormat": "",
                "OutputFormat": "",
                "SerdeInfo": {},
            },
            "PartitionKeys": [{"Name": "dt", "Type": "string"}],
        },
    )
    glue.create_partition(
        DatabaseName="qa-glue-partdb",
        TableName="qa-glue-parttbl",
        PartitionInput={
            "Values": ["2024-01-01"],
            "StorageDescriptor": {
                "Columns": [],
                "Location": "s3://bucket/key/dt=2024-01-01",
                "InputFormat": "",
                "OutputFormat": "",
                "SerdeInfo": {},
            },
        },
    )
    part = glue.get_partition(
        DatabaseName="qa-glue-partdb",
        TableName="qa-glue-parttbl",
        PartitionValues=["2024-01-01"],
    )["Partition"]
    assert part["Values"] == ["2024-01-01"]
    parts = glue.get_partitions(DatabaseName="qa-glue-partdb", TableName="qa-glue-parttbl")["Partitions"]
    assert len(parts) == 1
    glue.delete_partition(
        DatabaseName="qa-glue-partdb",
        TableName="qa-glue-parttbl",
        PartitionValues=["2024-01-01"],
    )
    parts2 = glue.get_partitions(DatabaseName="qa-glue-partdb", TableName="qa-glue-parttbl")["Partitions"]
    assert len(parts2) == 0

def test_glue_duplicate_partition_error(glue):
    """CreatePartition with duplicate values raises AlreadyExistsException."""
    glue.create_database(DatabaseInput={"Name": "qa-glue-duppartdb"})
    glue.create_table(
        DatabaseName="qa-glue-duppartdb",
        TableInput={
            "Name": "qa-glue-dupparttbl",
            "StorageDescriptor": {
                "Columns": [],
                "Location": "s3://b/k",
                "InputFormat": "",
                "OutputFormat": "",
                "SerdeInfo": {},
            },
            "PartitionKeys": [{"Name": "dt", "Type": "string"}],
        },
    )
    part_input = {
        "Values": ["2024-01-01"],
        "StorageDescriptor": {
            "Columns": [],
            "Location": "s3://b/k/dt=2024-01-01",
            "InputFormat": "",
            "OutputFormat": "",
            "SerdeInfo": {},
        },
    }
    glue.create_partition(
        DatabaseName="qa-glue-duppartdb",
        TableName="qa-glue-dupparttbl",
        PartitionInput=part_input,
    )
    with pytest.raises(ClientError) as exc:
        glue.create_partition(
            DatabaseName="qa-glue-duppartdb",
            TableName="qa-glue-dupparttbl",
            PartitionInput=part_input,
        )
    assert exc.value.response["Error"]["Code"] == "AlreadyExistsException"
