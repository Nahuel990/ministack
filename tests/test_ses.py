import io
import json
import os
import time
import zipfile
from urllib.parse import urlparse
import pytest
from botocore.exceptions import ClientError
import uuid as _uuid_mod

def test_ses_parse_smtp_host_not_set():
    from ministack.services.ses import _parse_smtp_host
    assert _parse_smtp_host() is None


def test_ses_parse_smtp_host_with_port():
    os.environ['SMTP_HOST'] = '127.0.0.1:1025'
    from ministack.services.ses import _parse_smtp_host
    assert _parse_smtp_host() == ('127.0.0.1', 1025)


def test_ses_parse_smtp_host_without_port():
    os.environ['SMTP_HOST'] = 'mail.example.com'
    from ministack.services.ses import _parse_smtp_host
    assert _parse_smtp_host() == ('mail.example.com', 25)


def test_ses_parse_smtp_host_hostname_with_port():
    os.environ['SMTP_HOST'] = 'smtp.gmail.com:587'
    from ministack.services.ses import _parse_smtp_host
    assert _parse_smtp_host() == ('smtp.gmail.com', 587)


def test_ses_send(ses):
    ses.verify_email_identity(EmailAddress="sender@example.com")
    resp = ses.send_email(
        Source="sender@example.com",
        Destination={"ToAddresses": ["recipient@example.com"]},
        Message={
            "Subject": {"Data": "Test Subject"},
            "Body": {"Text": {"Data": "Hello from MiniStack SES"}},
        },
    )
    assert "MessageId" in resp

def test_ses_list_identities(ses):
    ses.verify_email_identity(EmailAddress="another@example.com")
    resp = ses.list_identities()
    assert "sender@example.com" in resp["Identities"]

def test_ses_quota(ses):
    resp = ses.get_send_quota()
    assert resp["Max24HourSend"] == 50000.0

def test_ses_verify_identity_v2(ses):
    ses.verify_email_identity(EmailAddress="ses-v2@example.com")
    identities = ses.list_identities()["Identities"]
    assert "ses-v2@example.com" in identities

    attrs = ses.get_identity_verification_attributes(Identities=["ses-v2@example.com"])
    assert "ses-v2@example.com" in attrs["VerificationAttributes"]
    assert attrs["VerificationAttributes"]["ses-v2@example.com"]["VerificationStatus"] == "Success"

def test_ses_send_email_v2(ses):
    ses.verify_email_identity(EmailAddress="ses-send-v2@example.com")
    resp = ses.send_email(
        Source="ses-send-v2@example.com",
        Destination={
            "ToAddresses": ["to@example.com"],
            "CcAddresses": ["cc@example.com"],
        },
        Message={"Subject": {"Data": "Test V2"}, "Body": {"Text": {"Data": "Body v2"}}},
    )
    assert "MessageId" in resp

def test_ses_list_identities_v2(ses):
    ses.verify_email_identity(EmailAddress="ses-li-v2@example.com")
    ses.verify_domain_identity(Domain="example-v2.com")
    email_ids = ses.list_identities(IdentityType="EmailAddress")["Identities"]
    assert "ses-li-v2@example.com" in email_ids
    domain_ids = ses.list_identities(IdentityType="Domain")["Identities"]
    assert "example-v2.com" in domain_ids

def test_ses_quota_v2(ses):
    resp = ses.get_send_quota()
    assert resp["Max24HourSend"] == 50000.0
    assert resp["MaxSendRate"] == 14.0
    assert "SentLast24Hours" in resp

def test_ses_send_raw_email_v2(ses):
    ses.verify_email_identity(EmailAddress="raw-v2@example.com")
    raw = (
        "From: raw-v2@example.com\r\n"
        "To: dest-v2@example.com\r\n"
        "Subject: Raw V2\r\n"
        "Content-Type: text/plain\r\n\r\n"
        "Raw body v2"
    )
    resp = ses.send_raw_email(RawMessage={"Data": raw})
    assert "MessageId" in resp

def test_ses_configuration_set_v2(ses):
    ses.create_configuration_set(ConfigurationSet={"Name": "ses-cs-v2"})
    listed = ses.list_configuration_sets()["ConfigurationSets"]
    assert any(cs["Name"] == "ses-cs-v2" for cs in listed)

    described = ses.describe_configuration_set(ConfigurationSetName="ses-cs-v2")
    assert described["ConfigurationSet"]["Name"] == "ses-cs-v2"

    ses.delete_configuration_set(ConfigurationSetName="ses-cs-v2")
    listed2 = ses.list_configuration_sets()["ConfigurationSets"]
    assert not any(cs["Name"] == "ses-cs-v2" for cs in listed2)

def test_ses_template_v2(ses):
    ses.create_template(
        Template={
            "TemplateName": "ses-tpl-v2",
            "SubjectPart": "Hello {{name}}",
            "TextPart": "Hi {{name}}, order #{{oid}}",
            "HtmlPart": "<h1>Hi {{name}}</h1>",
        }
    )
    resp = ses.get_template(TemplateName="ses-tpl-v2")
    assert resp["Template"]["TemplateName"] == "ses-tpl-v2"
    assert "{{name}}" in resp["Template"]["SubjectPart"]

    listed = ses.list_templates()["TemplatesMetadata"]
    assert any(t["Name"] == "ses-tpl-v2" for t in listed)

    ses.update_template(
        Template={
            "TemplateName": "ses-tpl-v2",
            "SubjectPart": "Updated {{name}}",
            "TextPart": "Updated",
            "HtmlPart": "<p>Updated</p>",
        }
    )
    resp2 = ses.get_template(TemplateName="ses-tpl-v2")
    assert "Updated" in resp2["Template"]["SubjectPart"]

    ses.delete_template(TemplateName="ses-tpl-v2")
    with pytest.raises(ClientError):
        ses.get_template(TemplateName="ses-tpl-v2")

def test_ses_send_templated_v2(ses):
    ses.verify_email_identity(EmailAddress="tpl-v2@example.com")
    ses.create_template(
        Template={
            "TemplateName": "ses-tpl-send-v2",
            "SubjectPart": "Hey {{name}}",
            "TextPart": "Hi {{name}}",
            "HtmlPart": "<h1>Hi {{name}}</h1>",
        }
    )
    resp = ses.send_templated_email(
        Source="tpl-v2@example.com",
        Destination={"ToAddresses": ["r@example.com"]},
        Template="ses-tpl-send-v2",
        TemplateData=json.dumps({"name": "Alice"}),
    )
    assert "MessageId" in resp

def test_ses_send_templated_email(ses):
    """SendTemplatedEmail renders template and stores email."""
    ses.verify_email_identity(EmailAddress="sender@example.com")
    ses.create_template(
        Template={
            "TemplateName": "qa-ses-tmpl",
            "SubjectPart": "Hello {{name}}",
            "TextPart": "Hi {{name}}, welcome!",
            "HtmlPart": "<p>Hi {{name}}</p>",
        }
    )
    resp = ses.send_templated_email(
        Source="sender@example.com",
        Destination={"ToAddresses": ["user@example.com"]},
        Template="qa-ses-tmpl",
        TemplateData=json.dumps({"name": "Alice"}),
    )
    assert "MessageId" in resp

def test_ses_verify_domain(ses):
    """VerifyDomainIdentity returns a verification token."""
    resp = ses.verify_domain_identity(Domain="example.com")
    assert "VerificationToken" in resp
    assert len(resp["VerificationToken"]) > 0
    identities = ses.list_identities(IdentityType="Domain")["Identities"]
    assert "example.com" in identities

def test_ses_configuration_set_crud(ses):
    """CreateConfigurationSet / DescribeConfigurationSet / DeleteConfigurationSet."""
    ses.create_configuration_set(ConfigurationSet={"Name": "qa-ses-config"})
    desc = ses.describe_configuration_set(ConfigurationSetName="qa-ses-config")
    assert desc["ConfigurationSet"]["Name"] == "qa-ses-config"
    sets = ses.list_configuration_sets()["ConfigurationSets"]
    assert any(s["Name"] == "qa-ses-config" for s in sets)
    ses.delete_configuration_set(ConfigurationSetName="qa-ses-config")
    sets2 = ses.list_configuration_sets()["ConfigurationSets"]
    assert not any(s["Name"] == "qa-ses-config" for s in sets2)

def test_ses_v2_send_email(sesv2):
    resp = sesv2.send_email(
        FromEmailAddress="sender@example.com",
        Destination={"ToAddresses": ["recipient@example.com"]},
        Content={
            "Simple": {
                "Subject": {"Data": "Test Subject"},
                "Body": {"Text": {"Data": "Hello world"}},
            }
        },
    )
    assert resp["MessageId"].startswith("ministack-")

def test_ses_v2_email_identity_crud(sesv2):
    sesv2.create_email_identity(EmailIdentity="test-domain.com")
    resp = sesv2.get_email_identity(EmailIdentity="test-domain.com")
    assert resp["VerifiedForSendingStatus"] is True
    lst = sesv2.list_email_identities()
    names = [e["IdentityName"] for e in lst["EmailIdentities"]]
    assert "test-domain.com" in names
    sesv2.delete_email_identity(EmailIdentity="test-domain.com")
    lst2 = sesv2.list_email_identities()
    names2 = [e["IdentityName"] for e in lst2["EmailIdentities"]]
    assert "test-domain.com" not in names2

def test_ses_v2_configuration_set_crud(sesv2):
    sesv2.create_configuration_set(ConfigurationSetName="my-cfg-set")
    resp = sesv2.get_configuration_set(ConfigurationSetName="my-cfg-set")
    assert resp["ConfigurationSetName"] == "my-cfg-set"
    lst = sesv2.list_configuration_sets()
    assert "my-cfg-set" in lst["ConfigurationSets"]
    sesv2.delete_configuration_set(ConfigurationSetName="my-cfg-set")
    lst2 = sesv2.list_configuration_sets()
    assert "my-cfg-set" not in lst2["ConfigurationSets"]

def test_ses_v2_get_account(sesv2):
    resp = sesv2.get_account()
    assert resp["SendingEnabled"] is True
    assert resp["ProductionAccessEnabled"] is True
