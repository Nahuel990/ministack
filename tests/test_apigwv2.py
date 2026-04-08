import io
import json
import os
import time
import zipfile
from urllib.parse import urlparse
import pytest
from botocore.exceptions import ClientError
import uuid as _uuid_mod

def test_apigwv2_created_date_is_unix_timestamp(apigw):
    resp = apigw.create_api(Name="tf-date-test-v2", ProtocolType="HTTP")
    created = resp["CreatedDate"]
    import datetime

    assert isinstance(created, datetime.datetime), (
        f"CreatedDate should be datetime (parsed from Unix int), got {type(created)}"
    )
    apigw.delete_api(ApiId=resp["ApiId"])
