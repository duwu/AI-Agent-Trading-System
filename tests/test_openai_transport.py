"""The shared API must retry transient errors and expose genuine failures."""

import json
from unittest.mock import Mock

import pytest
import requests

from ai_agent.openai_analyzer import AIStreamError, ProxyOpenAIAnalyzer


@pytest.fixture
def analyzer(monkeypatch):
    monkeypatch.setenv("OPENAI_PROVIDER", "proxy")
    monkeypatch.setenv("OPENAI_READ_TIMEOUT", "180")
    monkeypatch.setenv("OPENAI_MAX_RETRIES", "2")
    monkeypatch.setenv("OPENAI_SEND_TEMPERATURE", "false")
    monkeypatch.setenv("OPENAI_STREAM", "false")
    monkeypatch.setattr("ai_agent.openai_analyzer.time.sleep", lambda _: None)
    return ProxyOpenAIAnalyzer(api_key="test-key", base_url="https://example.invalid/v1")


def response(status=200, content="真实模型内容"):
    result = requests.Response()
    result.status_code = status
    result._content = json.dumps({"choices": [{"message": {"content": content}}]}).encode()
    result._content_consumed = True
    return result


def test_timeout_then_success_uses_configured_endpoint_and_timeout(analyzer, monkeypatch):
    post = Mock(side_effect=[requests.ReadTimeout(), response()])
    monkeypatch.setattr("ai_agent.openai_analyzer.requests.post", post)
    assert analyzer.request_chat_completion("分析资产") == "真实模型内容"
    assert post.call_count == 2
    assert post.call_args.args[0] == "https://example.invalid/v1/chat/completions"
    assert post.call_args.kwargs["timeout"] == (10, 180)
    assert post.call_args.kwargs["headers"]["Authorization"] == "Bearer test-key"


@pytest.mark.parametrize("status", [429, 502, 503, 524])
def test_transient_status_recovers(analyzer, monkeypatch, status):
    post = Mock(side_effect=[response(status), response()])
    monkeypatch.setattr("ai_agent.openai_analyzer.requests.post", post)
    assert analyzer.request_chat_completion("prompt") == "真实模型内容"
    assert post.call_count == 2


def test_authentication_failure_does_not_retry_or_fabricate_result(analyzer, monkeypatch):
    post = Mock(return_value=response(401))
    monkeypatch.setattr("ai_agent.openai_analyzer.requests.post", post)
    with pytest.raises(RuntimeError, match="HTTP 401"):
        analyzer.request_chat_completion("prompt")
    assert post.call_count == 1


def test_repeated_timeout_is_bounded_and_reports_no_secret(analyzer, monkeypatch):
    post = Mock(side_effect=requests.ReadTimeout("sensitive upstream debug"))
    monkeypatch.setattr("ai_agent.openai_analyzer.requests.post", post)
    with pytest.raises(RuntimeError, match="ReadTimeout") as error:
        analyzer.request_chat_completion("prompt")
    assert post.call_count == 3
    assert "sensitive" not in str(error.value)
    assert "test-key" not in str(error.value)


def test_empty_content_is_explicit_failure(analyzer, monkeypatch):
    monkeypatch.setattr("ai_agent.openai_analyzer.requests.post", Mock(return_value=response(content="")))
    with pytest.raises(RuntimeError, match="空内容"):
        analyzer.request_chat_completion("prompt")


def test_stream_assembles_only_answer_and_ignores_heartbeats(analyzer, monkeypatch):
    monkeypatch.setenv("OPENAI_STREAM", "true")
    result = Mock(status_code=200, ok=True, headers={"Content-Type": "text/event-stream; charset=utf-8"})
    result.iter_lines.return_value = [
        b': heartbeat', b'',
        b'data: {"choices":[{"delta":{"reasoning_content":"private reasoning"}}]}',
        b'data: {"choices":[{"delta":{"content":"hello "}}]}',
        b'data: {"choices":[{"delta":{"content":"world"},"finish_reason":"stop"}]}',
        b'data: [DONE]',
    ]
    post = Mock(return_value=result)
    monkeypatch.setattr("ai_agent.openai_analyzer.requests.post", post)
    assert analyzer.request_chat_completion("prompt") == "hello world"
    assert post.call_args.kwargs["json"]["stream"] is True
    result.close.assert_called_once()


def test_incomplete_stream_is_not_a_valid_answer():
    result = Mock()
    result.iter_lines.return_value = [b'data: {"choices":[{"delta":{"content":"partial"}}]}']
    with pytest.raises(requests.ConnectionError):
        ProxyOpenAIAnalyzer._read_chat_stream(result)


def test_heartbeat_stream_obeys_total_deadline(monkeypatch):
    result = Mock()
    result.iter_lines.return_value = [b': heartbeat']
    monkeypatch.setattr("ai_agent.openai_analyzer.time.monotonic", lambda: 11)
    with pytest.raises(requests.ReadTimeout):
        ProxyOpenAIAnalyzer._read_chat_stream(result, deadline=10)


def test_stream_gateway_error_is_retryable_but_auth_error_is_not():
    result = Mock()
    result.iter_lines.return_value = [b'data: {"error":{"type":"server_error","code":"upstream_timeout","message":"private details"}}']
    with pytest.raises(AIStreamError) as error:
        ProxyOpenAIAnalyzer._read_chat_stream(result)
    assert error.value.retryable
    assert "private" not in str(error.value)
    assert not AIStreamError({"type": "authentication_error", "code": "invalid_api_key"}).retryable
