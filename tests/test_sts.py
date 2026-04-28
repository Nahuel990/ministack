import io
import json
import os
import time
import zipfile
from urllib.parse import urlparse
import pytest
from botocore.exceptions import ClientError
import uuid as _uuid_mod

def test_sts_get_caller_identity(sts):
    resp = sts.get_caller_identity()
    assert resp["Account"] == "000000000000"

def test_sts_assume_role_returns_credentials(sts):
    resp = sts.assume_role(
        RoleArn="arn:aws:iam::000000000000:role/test-role",
        RoleSessionName="intg-session",
    )
    creds = resp["Credentials"]
    assert "AccessKeyId" in creds
    assert "SecretAccessKey" in creds
    assert "SessionToken" in creds
    assert "Expiration" in creds
    assert resp["AssumedRoleUser"]["Arn"]

def test_sts_get_access_key_info(sts):
    resp = sts.get_access_key_info(AccessKeyId="AKIAIOSFODNN7EXAMPLE")
    assert "Account" in resp
    assert resp["Account"] == "000000000000"

def test_sts_get_caller_identity_full(sts):
    resp = sts.get_caller_identity()
    assert resp["Account"] == "000000000000"
    assert "Arn" in resp
    assert "UserId" in resp

def test_sts_assume_role(sts):
    resp = sts.assume_role(
        RoleArn="arn:aws:iam::000000000000:role/iam-test-role",
        RoleSessionName="test-session",
        DurationSeconds=900,
    )
    creds = resp["Credentials"]
    assert creds["AccessKeyId"].startswith("ASIA")
    assert len(creds["SecretAccessKey"]) > 0
    assert len(creds["SessionToken"]) > 0
    assert "Expiration" in creds

    assumed = resp["AssumedRoleUser"]
    assert "test-session" in assumed["Arn"]
    assert "AssumedRoleId" in assumed


def test_sts_assumed_role_arn_uses_sts_service(sts):
    """Real AWS returns AssumeRole's AssumedRoleUser.Arn under the sts
    service, not iam — e.g. arn:aws:sts::123456789012:assumed-role/demo/Sess.
    Pinning this against future regressions."""
    resp = sts.assume_role(
        RoleArn="arn:aws:iam::000000000000:role/demo",
        RoleSessionName="TestAR",
    )
    arn = resp["AssumedRoleUser"]["Arn"]
    assert arn == "arn:aws:sts::000000000000:assumed-role/demo/TestAR", arn

    resp_wi = sts.assume_role_with_web_identity(
        RoleArn="arn:aws:iam::000000000000:role/demo",
        RoleSessionName="WebSess",
        WebIdentityToken="dummy.jwt.token",
    )
    arn_wi = resp_wi["AssumedRoleUser"]["Arn"]
    assert arn_wi == "arn:aws:sts::000000000000:assumed-role/demo/WebSess", arn_wi

def test_sts_get_session_token(sts):
    resp = sts.get_session_token(DurationSeconds=900)
    creds = resp["Credentials"]
    assert "AccessKeyId" in creds
    assert "SecretAccessKey" in creds
    assert "SessionToken" in creds
    assert "Expiration" in creds

def test_sts_assume_role_with_web_identity(sts, iam):
    iam.create_role(
        RoleName="test-oidc-role",
        AssumeRolePolicyDocument='{"Version":"2012-10-17","Statement":[]}',
    )
    role_arn = f"arn:aws:iam::000000000000:role/test-oidc-role"
    resp = sts.assume_role_with_web_identity(
        RoleArn=role_arn,
        RoleSessionName="ci-session",
        WebIdentityToken="fake-oidc-token-value",
    )
    creds = resp["Credentials"]
    assert "AccessKeyId" in creds
    assert "SecretAccessKey" in creds
    assert "SessionToken" in creds
    assert "Expiration" in creds


def test_sts_get_web_identity_token():
    """GetWebIdentityToken returns a valid JWT with expected claims."""
    import urllib.request, re, base64
    endpoint = os.environ.get("MINISTACK_ENDPOINT", "http://localhost:4566")
    req = urllib.request.Request(
        endpoint,
        data=b"Action=GetWebIdentityToken&Audience=my-service&SigningAlgorithm=RS256&DurationSeconds=300",
        method="POST",
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Authorization": "AWS4-HMAC-SHA256 Credential=test/20240101/us-east-1/sts/aws4_request, SignedHeaders=host, Signature=fake",
        },
    )
    with urllib.request.urlopen(req) as r:
        assert r.status == 200
        body = r.read().decode()

    assert "<WebIdentityToken>" in body
    token = re.search(r"<WebIdentityToken>(.+?)</WebIdentityToken>", body).group(1)
    parts = token.split(".")
    assert len(parts) == 3, f"JWT should have 3 parts, got {len(parts)}"

    def _pad(s):
        return s + "=" * (-len(s) % 4)

    header = json.loads(base64.urlsafe_b64decode(_pad(parts[0])))
    payload = json.loads(base64.urlsafe_b64decode(_pad(parts[1])))

    assert header["alg"] == "RS256"
    assert header["typ"] == "JWT"
    assert payload["aud"] == "my-service"
    assert payload["iss"] == "https://sts.amazonaws.com"
    assert "sub" in payload
    assert "exp" in payload
    assert payload["exp"] - payload["iat"] == 300
