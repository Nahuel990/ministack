"""Unit tests for SES SMTP relay functionality."""
import os
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _clear_smtp_host():
    """Ensure SMTP_HOST is clean before/after each test."""
    old = os.environ.pop('SMTP_HOST', None)
    yield
    if old is not None:
        os.environ['SMTP_HOST'] = old
    else:
        os.environ.pop('SMTP_HOST', None)


@pytest.fixture(autouse=True)
def _reset_ses():
    """Reset SES module state between tests."""
    from ministack.services import ses
    ses.reset()


# ---------------------------------------------------------------------------
# _parse_smtp_host
# ---------------------------------------------------------------------------

def test_parse_smtp_host_not_set():
    from ministack.services.ses import _parse_smtp_host
    assert _parse_smtp_host() is None


def test_parse_smtp_host_with_port():
    os.environ['SMTP_HOST'] = '127.0.0.1:1025'
    from ministack.services.ses import _parse_smtp_host
    assert _parse_smtp_host() == ('127.0.0.1', 1025)


def test_parse_smtp_host_without_port():
    os.environ['SMTP_HOST'] = 'mail.example.com'
    from ministack.services.ses import _parse_smtp_host
    assert _parse_smtp_host() == ('mail.example.com', 25)


def test_parse_smtp_host_hostname_with_port():
    os.environ['SMTP_HOST'] = 'smtp.gmail.com:587'
    from ministack.services.ses import _parse_smtp_host
    assert _parse_smtp_host() == ('smtp.gmail.com', 587)


# ---------------------------------------------------------------------------
# _build_mime_message
# ---------------------------------------------------------------------------

def _parse_mime(msg_str):
    """Parse a MIME message string back for assertion."""
    from email import message_from_string
    return message_from_string(msg_str)


def test_build_mime_text_only():
    from ministack.services.ses import _build_mime_message
    result = _build_mime_message(
        'from@test.com', ['to@test.com'], [], [],
        'Subject', 'body text', '', 'msg-001',
    )
    msg = _parse_mime(result)
    assert msg['Subject'] == 'Subject'
    assert msg['From'] == 'from@test.com'
    assert msg['To'] == 'to@test.com'
    assert msg.get_content_type() == 'text/plain'


def test_build_mime_html_only():
    from ministack.services.ses import _build_mime_message
    result = _build_mime_message(
        'from@test.com', ['to@test.com'], [], [],
        'Subject', '', '<b>html</b>', 'msg-002',
    )
    msg = _parse_mime(result)
    assert msg.get_content_type() == 'text/html'


def test_build_mime_multipart():
    from ministack.services.ses import _build_mime_message
    result = _build_mime_message(
        'from@test.com', ['to@test.com'], ['cc@test.com'], [],
        'Subject', 'text', '<b>html</b>', 'msg-003',
    )
    msg = _parse_mime(result)
    assert msg.get_content_type() == 'multipart/alternative'
    assert msg['Cc'] == 'cc@test.com'


# ---------------------------------------------------------------------------
# _smtp_relay
# ---------------------------------------------------------------------------

def test_smtp_relay_skipped_when_no_host():
    from ministack.services.ses import _smtp_relay
    with patch('ministack.services.ses.smtplib.SMTP') as mock_cls:
        _smtp_relay('from@test.com', ['to@test.com'], 'message')
        mock_cls.assert_not_called()


def test_smtp_relay_sends_when_host_set():
    os.environ['SMTP_HOST'] = '127.0.0.1:1025'
    from ministack.services.ses import _smtp_relay
    mock_smtp = MagicMock()
    with patch('ministack.services.ses.smtplib.SMTP', return_value=mock_smtp) as mock_cls:
        mock_smtp.__enter__ = MagicMock(return_value=mock_smtp)
        mock_smtp.__exit__ = MagicMock(return_value=False)
        _smtp_relay('from@test.com', ['to@test.com'], 'message body')
        mock_cls.assert_called_once_with('127.0.0.1', 1025)
        mock_smtp.sendmail.assert_called_once_with(
            'from@test.com', ['to@test.com'], 'message body',
        )


def test_smtp_relay_error_is_logged_not_raised():
    os.environ['SMTP_HOST'] = '127.0.0.1:1025'
    from ministack.services.ses import _smtp_relay
    with patch('ministack.services.ses.smtplib.SMTP', side_effect=ConnectionRefusedError):
        # Should not raise
        _smtp_relay('from@test.com', ['to@test.com'], 'message')


# ---------------------------------------------------------------------------
# SendEmail with SMTP relay
# ---------------------------------------------------------------------------

def test_send_email_relays(monkeypatch):
    monkeypatch.setenv('SMTP_HOST', '127.0.0.1:1025')
    from ministack.services.ses import _send_email
    mock_smtp = MagicMock()
    with patch('ministack.services.ses.smtplib.SMTP', return_value=mock_smtp):
        mock_smtp.__enter__ = MagicMock(return_value=mock_smtp)
        mock_smtp.__exit__ = MagicMock(return_value=False)
        params = {
            'Source': ['sender@example.com'],
            'Destination.ToAddresses.member.1': ['to@example.com'],
            'Destination.CcAddresses.member.1': ['cc@example.com'],
            'Message.Subject.Data': ['Test Subject'],
            'Message.Body.Text.Data': ['Hello'],
            'Message.Body.Html.Data': ['<b>Hello</b>'],
        }
        status, headers, body = _send_email(params)
        assert status == 200
        mock_smtp.sendmail.assert_called_once()
        call_args = mock_smtp.sendmail.call_args
        assert call_args[0][0] == 'sender@example.com'
        assert set(call_args[0][1]) == {'to@example.com', 'cc@example.com'}
        msg = _parse_mime(call_args[0][2])
        assert msg['Subject'] == 'Test Subject'
        assert msg.get_content_type() == 'multipart/alternative'


def test_send_email_no_relay_without_host():
    from ministack.services.ses import _send_email
    with patch('ministack.services.ses.smtplib.SMTP') as mock_cls:
        params = {
            'Source': ['sender@example.com'],
            'Destination.ToAddresses.member.1': ['to@example.com'],
            'Message.Subject.Data': ['Test'],
            'Message.Body.Text.Data': ['body'],
        }
        status, _, _ = _send_email(params)
        assert status == 200
        mock_cls.assert_not_called()


# ---------------------------------------------------------------------------
# SendRawEmail with SMTP relay
# ---------------------------------------------------------------------------

def test_send_raw_email_relays(monkeypatch):
    monkeypatch.setenv('SMTP_HOST', 'localhost:2525')
    from ministack.services.ses import _send_raw_email
    mock_smtp = MagicMock()
    with patch('ministack.services.ses.smtplib.SMTP', return_value=mock_smtp):
        mock_smtp.__enter__ = MagicMock(return_value=mock_smtp)
        mock_smtp.__exit__ = MagicMock(return_value=False)
        raw_msg = (
            'From: raw@example.com\r\n'
            'To: dest@example.com\r\n'
            'Subject: Raw Test\r\n'
            '\r\n'
            'Raw body'
        )
        params = {
            'Source': ['raw@example.com'],
            'Destinations.member.1': ['dest@example.com'],
            'RawMessage.Data': [raw_msg],
        }
        status, _, _ = _send_raw_email(params)
        assert status == 200
        mock_smtp.sendmail.assert_called_once()
        call_args = mock_smtp.sendmail.call_args
        assert call_args[0][0] == 'raw@example.com'
        assert 'dest@example.com' in call_args[0][1]


# ---------------------------------------------------------------------------
# SendTemplatedEmail with SMTP relay
# ---------------------------------------------------------------------------

def test_send_templated_email_relays(monkeypatch):
    monkeypatch.setenv('SMTP_HOST', 'localhost:1025')
    from ministack.services.ses import _send_templated_email, _templates
    _templates['MyTemplate'] = {
        'TemplateName': 'MyTemplate',
        'SubjectPart': 'Hello {{name}}',
        'TextPart': 'Hi {{name}}',
        'HtmlPart': '<b>Hi {{name}}</b>',
    }
    mock_smtp = MagicMock()
    with patch('ministack.services.ses.smtplib.SMTP', return_value=mock_smtp):
        mock_smtp.__enter__ = MagicMock(return_value=mock_smtp)
        mock_smtp.__exit__ = MagicMock(return_value=False)
        params = {
            'Source': ['tmpl@example.com'],
            'Destination.ToAddresses.member.1': ['to@example.com'],
            'Template': ['MyTemplate'],
            'TemplateData': ['{"name": "World"}'],
        }
        status, _, _ = _send_templated_email(params)
        assert status == 200
        mock_smtp.sendmail.assert_called_once()
        msg = _parse_mime(mock_smtp.sendmail.call_args[0][2])
        assert 'Hello World' in msg['Subject']
