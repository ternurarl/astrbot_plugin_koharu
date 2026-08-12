"""集成测试:KoharuClient × httpx.MockTransport(全部请求走 mock,不访问网络)。"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from conftest import KoharuClientFactory
from koharu_client import (
    KoharuApiError,
    KoharuTimeoutError,
    LLMLoadOptions,
    LLMTarget,
)


async def test_wait_until_ready_immediate(koharu_client_factory: KoharuClientFactory) -> None:
    """/meta 200 → 立即返回 MetaInfo。"""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"name": "koharu", "version": "1.0"})

    async with koharu_client_factory(handler) as client:
        meta = await client.wait_until_ready(timeout_seconds=5, interval_seconds=0.01)
    assert meta.get("name") == "koharu"
    assert meta.get("version") == "1.0"


async def test_wait_until_ready_retries_after_503(koharu_client_factory: KoharuClientFactory) -> None:
    """503 重试后 200 → 成功返回,期间共发出 2 次请求。"""
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(503, text="koharu still starting")
        return httpx.Response(200, json={"name": "koharu"})

    async with koharu_client_factory(handler) as client:
        meta = await client.wait_until_ready(timeout_seconds=5, interval_seconds=0.01)
    assert meta.get("name") == "koharu"
    assert calls == 2


async def test_wait_until_ready_times_out(koharu_client_factory: KoharuClientFactory) -> None:
    """一直失败(503)→ 超时抛 KoharuTimeoutError。"""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="starting")

    with pytest.raises(KoharuTimeoutError):
        async with koharu_client_factory(handler) as client:
            await client.wait_until_ready(timeout_seconds=0.05, interval_seconds=0.01)


@pytest.mark.parametrize("key", ["id", "projectId", "project_id"])
async def test_create_project_id_variants(koharu_client_factory: KoharuClientFactory, key: str) -> None:
    """create_project 响应含 id/projectId/project_id 任一变体都能解析出项目 id。"""
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={key: "proj-42"})

    async with koharu_client_factory(handler) as client:
        project = await client.create_project("my-project")
    assert captured[0].method == "POST"
    assert captured[0].url.path == "/api/v1/projects"
    body: dict[str, object] = json.loads(captured[0].content)
    assert body == {"name": "my-project"}
    assert (
        project.get("id") or project.get("projectId") or project.get("project_id")
    ) == "proj-42"


async def test_create_project_non_dict_response_empty(koharu_client_factory: KoharuClientFactory) -> None:
    """create_project 响应非 dict → 归一化为空 ProjectInfo。"""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=["not-a-dict"])

    async with koharu_client_factory(handler) as client:
        project = await client.create_project("p")
    assert project == {}


async def test_create_pages_multipart_with_replace(koharu_client_factory: KoharuClientFactory, tmp_path: Path) -> None:
    """create_pages:multipart 请求体含文件内容与文件名,且带 replace=true 参数。"""
    page = tmp_path / "page1.png"
    page.write_bytes(b"fake-png-bytes-1")
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"pages": [{"id": "page-1"}]})

    async with koharu_client_factory(handler) as client:
        result = await client.create_pages([str(page)], replace=True)

    request = captured[0]
    assert request.method == "POST"
    assert request.url.path == "/api/v1/pages"
    assert request.headers["content-type"].startswith("multipart/form-data")
    assert b"page1.png" in request.content
    assert b"fake-png-bytes-1" in request.content
    assert b"replace" in request.content
    assert result == {"pages": [{"id": "page-1"}]}


async def test_create_pages_non_dict_response_raises(koharu_client_factory: KoharuClientFactory, tmp_path: Path) -> None:
    """create_pages 响应非 dict → KoharuApiError。"""
    page = tmp_path / "page1.png"
    page.write_bytes(b"x")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=["junk"])

    with pytest.raises(KoharuApiError, match="unexpected response"):
        async with koharu_client_factory(handler) as client:
            await client.create_pages([str(page)], replace=True)


async def test_start_pipeline_duplicate_field_names(koharu_client_factory: KoharuClientFactory) -> None:
    """start_pipeline:请求 body 同时写双下划线与驼峰字段。"""
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"operationId": "op-9"})

    async with koharu_client_factory(handler) as client:
        operation_id = await client.start_pipeline(
            ["detect", "ocr"],
            target_language="Simplified Chinese",
            system_prompt="be careful",
            default_font="Noto Sans SC:500",
        )

    assert operation_id == "op-9"
    request = captured[0]
    assert request.method == "POST"
    assert request.url.path == "/api/v1/pipelines"
    body: dict[str, object] = json.loads(request.content)
    assert body["steps"] == ["detect", "ocr"]
    assert body["target_language"] == "Simplified Chinese"
    assert body["targetLanguage"] == "Simplified Chinese"
    assert body["system_prompt"] == "be careful"
    assert body["systemPrompt"] == "be careful"
    assert body["default_font"] == "Noto Sans SC:500"
    assert body["defaultFont"] == "Noto Sans SC:500"


async def test_start_pipeline_missing_operation_id_raises(koharu_client_factory: KoharuClientFactory) -> None:
    """start_pipeline 响应缺 operationId → KoharuApiError。"""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "queued"})

    with pytest.raises(KoharuApiError, match="operationId"):
        async with koharu_client_factory(handler) as client:
            await client.start_pipeline(["detect"])


async def test_start_pipeline_non_dict_response_raises(koharu_client_factory: KoharuClientFactory) -> None:
    """start_pipeline 响应非 dict → KoharuApiError。"""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=["not-a-dict"])

    with pytest.raises(KoharuApiError, match="operationId"):
        async with koharu_client_factory(handler) as client:
            await client.start_pipeline(["detect"])


async def test_wait_operation_polls_until_finished(koharu_client_factory: KoharuClientFactory) -> None:
    """wait_operation:进行中 → 轮询 → finished 返回;dict 形响应({operations})也可用。"""
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        status = "processing" if calls == 1 else "finished"
        return httpx.Response(200, json={"operations": [{"id": "op-1", "status": status}]})

    async with koharu_client_factory(handler) as client:
        operation = await client.wait_operation(
            "op-1",
            timeout_seconds=5,
            interval_seconds=0.01,
        )
    assert operation.get("status") == "finished"
    assert calls == 2


async def test_wait_operation_failed_status_raises(koharu_client_factory: KoharuClientFactory) -> None:
    """wait_operation:failed 状态 → KoharuApiError,含错误信息。"""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=[{"id": "op-1", "status": "failed", "error": "model crashed"}],
        )

    with pytest.raises(KoharuApiError, match="model crashed"):
        async with koharu_client_factory(handler) as client:
            await client.wait_operation("op-1", timeout_seconds=5, interval_seconds=0.01)


async def test_wait_operation_times_out(koharu_client_factory: KoharuClientFactory) -> None:
    """wait_operation:一直进行中 → 超时抛 KoharuTimeoutError。"""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[{"id": "op-1", "status": "processing"}])

    with pytest.raises(KoharuTimeoutError):
        async with koharu_client_factory(handler) as client:
            await client.wait_operation(
                "op-1",
                timeout_seconds=0.05,
                interval_seconds=0.005,
            )


async def test_export_project_returns_bytes_and_content_type(koharu_client_factory: KoharuClientFactory) -> None:
    """export_project:返回 (bytes, content-type)。"""
    captured: list[httpx.Request] = []
    png_bytes = b"\x89PNG\r\n\x1a\nfake-image-data"

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, content=png_bytes, headers={"content-type": "image/png"})

    async with koharu_client_factory(handler) as client:
        content, content_type = await client.export_project("rendered")

    assert content == png_bytes
    assert content_type == "image/png"
    assert captured[0].method == "POST"
    assert captured[0].url.path == "/api/v1/projects/current/export"
    body: dict[str, object] = json.loads(captured[0].content)
    assert body == {"format": "rendered"}


async def test_load_llm_body_structure(koharu_client_factory: KoharuClientFactory) -> None:
    """load_llm:body 含 target/options 结构,走 PUT /llm/current,期望 204。"""
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(204)

    target: LLMTarget = {
        "kind": "provider",
        "providerId": "openai-compatible",
        "modelId": "gpt-4",
    }
    options: LLMLoadOptions = {
        "temperature": 0.7,
        "maxTokens": 512,
        "customSystemPrompt": "translate carefully",
    }
    async with koharu_client_factory(handler) as client:
        await client.load_llm(target, options=options)

    request = captured[0]
    assert request.method == "PUT"
    assert request.url.path == "/api/v1/llm/current"
    body: dict[str, object] = json.loads(request.content)
    assert body["target"] == target
    assert body["options"] == options


async def test_load_llm_without_options(koharu_client_factory: KoharuClientFactory) -> None:
    """load_llm:无 options 时 body 只含 target。"""
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(204)

    async with koharu_client_factory(handler) as client:
        target: LLMTarget = {"kind": "local", "modelId": "qwen"}
        await client.load_llm(target)

    body: dict[str, object] = json.loads(captured[0].content)
    assert body == {"target": {"kind": "local", "modelId": "qwen"}}


async def test_non_2xx_raises_koharu_api_error_with_body(koharu_client_factory: KoharuClientFactory) -> None:
    """非 2xx 状态码 → KoharuApiError,含响应体片段。"""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="internal boom: model missing")

    with pytest.raises(KoharuApiError) as exc_info:
        async with koharu_client_factory(handler) as client:
            await client.get_meta()
    message = str(exc_info.value)
    assert "HTTP 500" in message
    assert "internal boom: model missing" in message


@pytest.mark.parametrize(
    ("payload", "expected_count"),
    [
        ([{"id": "a"}, {"id": "b"}], 2),
        ({"operations": [{"id": "c"}]}, 1),
        ({"jobs": [{"id": "d"}]}, 1),
        ({"x": {"id": "e"}, "y": {"id": "f"}}, 2),
        ({"operations": "junk"}, 0),
        ("not-json-object", 0),
    ],
    ids=[
        "bare-list",
        "dict-with-operations-key",
        "dict-with-jobs-key",
        "dict-with-dict-values",
        "dict-with-junk-values",
        "non-dict",
    ],
)
async def test_list_operations_normalization(
    koharu_client_factory: KoharuClientFactory,
    payload: object,
    expected_count: int,
) -> None:
    """list_operations:list / {operations} / {jobs} / dict 值 / 垃圾响应都能归一化。"""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    async with koharu_client_factory(handler) as client:
        operations = await client.list_operations()
    assert len(operations) == expected_count


async def test_get_llm_current_non_dict_normalized(koharu_client_factory: KoharuClientFactory) -> None:
    """get_llm_current 响应非 dict → 归一化为 {}。"""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=["junk"])

    async with koharu_client_factory(handler) as client:
        state = await client.get_llm_current()
    assert state == {}
