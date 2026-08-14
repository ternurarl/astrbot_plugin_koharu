from __future__ import annotations

import asyncio
import json
import mimetypes
import os
import time
import zipfile
from collections.abc import AsyncIterator, Iterable, Sequence
from pathlib import Path
from typing import BinaryIO, Literal, TypeAlias, TypedDict, cast

import httpx

try:
    from astrbot.api import logger
except Exception:  # pragma: no cover - used when the client is run standalone.
    import logging

    logger = logging.getLogger(__name__)


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}

# 幂等删除/卸载接口的宽容响应码(200/202/204 成功,400/404/409 视为已不存在)。
_LENIENT_DELETE_STATUS = {200, 202, 204, 400, 404, 409}


# --- JSON / payload types ---------------------------------------------------------

JsonValue: TypeAlias = (
    str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
)
"""A JSON-serializable value."""

FilesPayload: TypeAlias = (
    list[tuple[str, tuple[str, BinaryIO, str]]]
    | dict[str, tuple[str, BinaryIO, str]]
)
"""Multipart payloads accepted by httpx: ``(field, (filename, file, content_type))``."""


class KoharuApiError(RuntimeError):
    """Raised when Koharu returns an error or a pipeline job fails."""


class KoharuTimeoutError(TimeoutError):
    """Raised when Koharu does not complete an operation in time."""


# --- Response types ---------------------------------------------------------------

class MetaInfo(TypedDict, total=False):
    """Shape of GET /meta."""

    name: str
    version: str
    appVersion: str


class ProjectInfo(TypedDict, total=False):
    """Shape of a Koharu project (0.66 项目无 id,身份标识为 name)。

    保留 id/projectId/project_id 仅用于 extract_project_id 对旧响应的兼容读取。
    """

    id: str
    projectId: str
    project_id: str
    name: str
    revision: str
    activePage: str
    pages: list[JsonValue]


class OperationInfo(TypedDict, total=False):
    """Shape of a Koharu operation (pipeline job)."""

    id: str
    operationId: str
    jobId: str
    status: str
    error: str


class OperationStartResponse(TypedDict, total=False):
    """Shape of the response returned when an operation is started."""

    operationId: str


class LLMCurrentState(TypedDict, total=False):
    """Shape of GET /llm/current (0.66 无 status 字段,模型翻译时懒加载)。"""

    model: dict[str, JsonValue]
    targetLanguage: str
    instructions: str


class PipelineConfig(TypedDict, total=False):
    """Shape of the ``pipeline`` section of the 0.66 Koharu config."""

    detection: JsonValue
    ocr: JsonValue
    translation: JsonValue
    inpainting: JsonValue


class KoharuConfig(TypedDict, total=False):
    """Shape of GET /config (only the fields the plugin consumes)."""

    pipeline: PipelineConfig


class ProjectsResponse(TypedDict, total=False):
    """Best-effort shape of GET /projects (not consumed by the plugin)."""

    projects: list[ProjectInfo]


class CatalogResponse(TypedDict, total=False):
    """Best-effort shape of GET /llm/catalog (not consumed by the plugin)."""

    models: list["LLMModelInfo"]


class LLMModelInfo(TypedDict, total=False):
    id: str
    providerId: str
    name: str


class SSEEvent(TypedDict, total=False):
    """A parsed server-sent event emitted by :meth:`KoharuClient.iter_events`."""

    id: str
    event: str
    raw: str
    data: JsonValue


# --- Request body types -----------------------------------------------------------

class ProjectCreateBody(TypedDict):
    name: str


class ExportBody(TypedDict, total=False):
    format: str
    pages: list[str]


class LLMTargetProvider(TypedDict):
    kind: Literal["provider"]
    providerId: str
    modelId: str


class LLMTargetLocal(TypedDict):
    kind: Literal["local"]
    modelId: str


LLMTarget: TypeAlias = LLMTargetProvider | LLMTargetLocal
"""A provider or local LLM to load via :meth:`KoharuClient.load_llm`."""


class LLMLoadOptions(TypedDict, total=False):
    customSystemPrompt: str


class ProviderSecretBody(TypedDict, total=False):
    secret: str


class PatchBody(TypedDict, total=False):
    """Opaque config patch payload, passed through to the Koharu API.

    0.66 PATCH /config 是顶层稀疏合并,section 整段替换。
    """

    pipeline: PipelineConfig


class KoharuClient:
    """Async wrapper for Koharu HTTP API v1."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 60.0,
        connect_timeout: float = 10.0,
    ) -> None:
        normalized = base_url.rstrip("/")
        if not normalized.endswith("/api/v1"):
            normalized = f"{normalized}/api/v1"
        self.base_url = normalized
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout, connect=connect_timeout),
        )

    async def __aenter__(self) -> "KoharuClient":
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _request(
        self,
        method: str,
        path: str,
        *,
        expected_status: Iterable[int] = range(200, 300),
        json: JsonValue | None = None,
        files: FilesPayload | None = None,
        data: dict[str, str] | None = None,
        content: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        started = time.monotonic()
        logger.debug("[koharu-client] request %s %s", method, path)
        response = await self._client.request(
            method,
            path,
            json=json,
            files=files,
            data=data,
            content=content,
            headers=headers,
        )
        elapsed_ms = int((time.monotonic() - started) * 1000)
        logger.debug(
            "[koharu-client] response %s %s status=%s elapsed_ms=%s",
            method,
            path,
            response.status_code,
            elapsed_ms,
        )
        if response.status_code not in expected_status:
            detail = response.text[:1000]
            raise KoharuApiError(
                f"Koharu API {method} {path} failed: "
                f"HTTP {response.status_code}: {detail}"
            )
        return response

    async def _json(
        self,
        method: str,
        path: str,
        *,
        expected_status: Iterable[int] = range(200, 300),
        json: JsonValue | None = None,
        files: FilesPayload | None = None,
        data: dict[str, str] | None = None,
        content: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> JsonValue:
        response = await self._request(
            method,
            path,
            expected_status=expected_status,
            json=json,
            files=files,
            data=data,
            content=content,
            headers=headers,
        )
        if response.status_code == 204 or not response.content:
            return None
        return cast(JsonValue, response.json())

    async def wait_until_ready(
        self,
        *,
        timeout_seconds: float = 60.0,
        interval_seconds: float = 1.0,
    ) -> MetaInfo:
        deadline = time.monotonic() + timeout_seconds
        last_error: Exception | None = None
        logger.debug("[koharu-client] wait_until_ready timeout_seconds=%s", timeout_seconds)
        while time.monotonic() < deadline:
            try:
                data = await self._json("GET", "/meta", expected_status={200})
                return cast(MetaInfo, data) if isinstance(data, dict) else {}
            except Exception as exc:  # Koharu may still be starting.
                last_error = exc
            await asyncio.sleep(interval_seconds)
        detail = f": {last_error}" if last_error else ""
        raise KoharuTimeoutError(f"Koharu is not ready within {timeout_seconds}s{detail}")

    # Meta
    async def get_meta(self) -> MetaInfo:
        data = await self._json("GET", "/meta")
        return cast(MetaInfo, data) if isinstance(data, dict) else {}

    # Projects
    async def list_projects(self) -> ProjectsResponse:
        data = await self._json("GET", "/projects")
        return cast(ProjectsResponse, data) if isinstance(data, dict) else {}

    async def create_project(self, name: str) -> ProjectInfo:
        body: dict[str, JsonValue] = {"name": name}
        data = await self._json("POST", "/projects", json=body)
        return cast(ProjectInfo, data) if isinstance(data, dict) else {}

    async def open_project(self, project_name: str) -> ProjectInfo:
        body: dict[str, JsonValue] = {"name": project_name}
        data = await self._json("PUT", "/projects/current", json=body)
        return cast(ProjectInfo, data) if isinstance(data, dict) else {}

    async def close_project(self) -> None:
        await self._json("DELETE", "/projects/current")

    async def close_project_if_any(self) -> bool:
        response = await self._request(
            "DELETE",
            "/projects/current",
            expected_status=_LENIENT_DELETE_STATUS,
        )
        return 200 <= response.status_code < 300

    async def delete_project(self, project_name: str) -> None:
        """按名称删除项目(0.66 的 DELETE /projects/{name})。"""
        await self._json("DELETE", f"/projects/{project_name}")

    async def delete_project_if_possible(self, project_name: str) -> bool:
        response = await self._request(
            "DELETE",
            f"/projects/{project_name}",
            expected_status=_LENIENT_DELETE_STATUS,
        )
        return 200 <= response.status_code < 300

    async def export_project(
        self,
        export_format: str = "rendered",
        *,
        pages: list[str] | None = None,
    ) -> tuple[bytes, str]:
        body: dict[str, JsonValue] = {"format": export_format}
        if pages:
            body["pages"] = cast(JsonValue, pages)
        response = await self._request("POST", "/projects/current/export", json=body)
        return response.content, response.headers.get("content-type", "")

    # Pages
    async def create_pages(
        self,
        image_paths: Sequence[str | os.PathLike[str]],
    ) -> ProjectInfo:
        """上传图片追加为当前项目页面(0.66: POST /projects/current/pages)。

        响应是 ProjectInfo(页列表在 pages 键)。0.66 无 replace 语义,直接追加。
        """
        opened: list[BinaryIO] = []
        try:
            files: list[tuple[str, tuple[str, BinaryIO, str]]] = []
            for image_path in image_paths:
                path = Path(image_path)
                file_obj = path.open("rb")
                opened.append(file_obj)
                content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
                files.append(("files", (path.name, file_obj, content_type)))
            data = await self._json("POST", "/projects/current/pages", files=files)
            if not isinstance(data, dict):
                raise KoharuApiError(
                    f"Koharu create_pages returned an unexpected response: {data!r}"
                )
            return cast(ProjectInfo, data)
        finally:
            for file_obj in opened:
                file_obj.close()

    async def get_page_thumbnail(self, page_id: str) -> bytes:
        response = await self._request("GET", f"/pages/{page_id}/thumbnail")
        return response.content

    # Pipelines
    async def start_pipeline(self, steps: list[str]) -> str:
        """启动 0.66 流水线,请求体为 Operation/Scope。

        steps 为 ["full"]（或空）时执行全部阶段；legacy 0.61 步骤名
        （detector/segmenter/ocr/translator/inpainter 等）映射为 0.66 的
        stages（renderer 忽略）。0.66 不再接受 target_language / system_prompt
        / default_font，语言由服务端配置决定。
        """
        body: dict[str, JsonValue] = {
            "operation": {"operation": "full"},
            "scope": {"scope": "project"},
        }
        if steps and not (len(steps) == 1 and steps[0].strip().lower() == "full"):
            stages = _pipeline_stages_from_steps(steps)
            if not stages:
                raise KoharuApiError(
                    "无法从 pipeline steps 映射出任何 0.66 阶段。"
                    "请配置 full 或 detection/ocr/translation/inpainting。"
                )
            body["operation"] = cast(
                JsonValue,
                {"operation": "stages", "stages": cast(JsonValue, stages)},
            )
        logger.debug("[koharu-client] start_pipeline body=%s", body)
        data = await self._json("POST", "/pipelines", json=body)
        return _extract_operation_id(data)

    # Operations
    async def list_operations(self) -> list[OperationInfo]:
        data = await self._json("GET", "/operations")
        return _normalize_operation_list(data)

    async def cancel_operation(self, operation_id: str) -> OperationInfo:
        data = await self._json("DELETE", f"/operations/{operation_id}")
        return cast(OperationInfo, data) if isinstance(data, dict) else {}

    async def wait_operation(
        self,
        operation_id: str,
        *,
        timeout_seconds: float = 900.0,
        interval_seconds: float = 2.0,
    ) -> OperationInfo:
        deadline = time.monotonic() + timeout_seconds
        last_seen: OperationInfo | None = None
        last_logged_status: str | None = None
        logger.debug(
            "[koharu-client] wait_operation operation_id=%s timeout_seconds=%s interval_seconds=%s",
            operation_id,
            timeout_seconds,
            interval_seconds,
        )
        while time.monotonic() < deadline:
            operations = await self.list_operations()
            current = _find_operation(operations, operation_id)
            if current:
                last_seen = current
                status = str(current.get("status", "")).lower()
                if status != last_logged_status:
                    logger.debug(
                        "[koharu-client] operation status operation_id=%s status=%s state=%s",
                        operation_id,
                        status,
                        current,
                    )
                    last_logged_status = status
                if status in {"finished", "completed", "complete", "succeeded", "success", "done"}:
                    return current
                if status in {"failed", "error", "cancelled", "canceled"}:
                    raise KoharuApiError(
                        f"Koharu operation {operation_id} failed: "
                        f"{current.get('error') or current}"
                    )
            await asyncio.sleep(interval_seconds)
        raise KoharuTimeoutError(
            f"Koharu operation {operation_id} did not finish within "
            f"{timeout_seconds}s. Last state: {last_seen}"
        )

    # LLM control
    async def get_llm_current(self) -> LLMCurrentState:
        data = await self._json("GET", "/llm/current")
        return cast(LLMCurrentState, data) if isinstance(data, dict) else {}

    async def load_llm(
        self,
        target: LLMTarget,
        *,
        options: LLMLoadOptions | None = None,
    ) -> None:
        """选择翻译模型(0.66: PUT /llm/current,204 即完成,模型翻译时懒加载)。

        0.66 的 ModelSelection 必填 provider/model/vision：远程 provider 按
        catalog 约定 vision=false（deepseek 等文本服务商），local 为 true。
        options.customSystemPrompt 映射为 instructions；temperature/maxTokens
        由 0.66 的 pipeline.translation.generation 配置管理，这里不再接受。
        """
        if target["kind"] == "provider":
            model: dict[str, JsonValue] = {
                "provider": target["providerId"],
                "model": target["modelId"],
                "vision": False,
            }
        else:
            model = {
                "provider": "local",
                "model": target["modelId"],
                "vision": True,
            }
        body: dict[str, JsonValue] = {"model": model}
        instructions = options.get("customSystemPrompt") if options else None
        if instructions:
            body["instructions"] = instructions
        await self._json("PUT", "/llm/current", json=body, expected_status={204})
    async def unload_llm(self) -> None:
        """重置翻译模型为默认 local(0.66 语义;--gpu 下会弄坏翻译,主流程不要调用)。"""
        await self._request(
            "DELETE",
            "/llm/current",
            expected_status=_LENIENT_DELETE_STATUS,
        )

    async def get_llm_catalog(self) -> CatalogResponse:
        data = await self._json("GET", "/llm/catalog")
        return cast(CatalogResponse, data) if isinstance(data, dict) else {}

    # Config
    async def get_config(self) -> KoharuConfig:
        data = await self._json("GET", "/config")
        return cast(KoharuConfig, data) if isinstance(data, dict) else {}

    async def patch_config(self, patch: PatchBody) -> PatchBody:
        data = await self._json("PATCH", "/config", json=cast(JsonValue, patch))
        return cast(PatchBody, data) if isinstance(data, dict) else {}

    async def set_provider_secret(self, provider_id: str, secret: str) -> None:
        """写入服务商密钥(0.66: {"secret": ...} 存 keyring,容器重启后需重放)。"""
        body: dict[str, JsonValue] = {"secret": secret}
        await self._request(
            "PUT",
            f"/config/providers/{provider_id}/secret",
            json=body,
        )

    async def clear_provider_secret(self, provider_id: str) -> None:
        await self._request("DELETE", f"/config/providers/{provider_id}/secret")

    # Events
    async def iter_events(
        self,
        *,
        last_event_id: str | None = None,
    ) -> AsyncIterator[SSEEvent]:
        headers = {"Last-Event-ID": last_event_id} if last_event_id else None
        async with self._client.stream("GET", "/events", headers=headers) as response:
            response.raise_for_status()
            event_name = ""
            event_id = ""
            data_lines: list[str] = []
            async for line in response.aiter_lines():
                if line == "":
                    if data_lines:
                        raw = "\n".join(data_lines)
                        yield {
                            "id": event_id,
                            "event": event_name,
                            "raw": raw,
                            "data": _parse_event_data(raw),
                        }
                    event_name = ""
                    event_id = ""
                    data_lines = []
                    continue
                if line.startswith(":"):
                    continue
                field, _, value = line.partition(":")
                value = value[1:] if value.startswith(" ") else value
                if field == "id":
                    event_id = value
                elif field == "event":
                    event_name = value
                elif field == "data":
                    data_lines.append(value)

    async def get_pipeline_steps_from_config(self) -> list[str]:
        """从 0.66 /config 读取 pipeline 阶段。

        0.66 是固定阶段流水线（detection/ocr/translation/inpainting），
        配置存在即返回 ["full"]（对应 Operation::Full，即正确语义）。
        """
        config = await self.get_config()
        pipeline = config.get("pipeline")
        if pipeline is None:
            return []
        return ["full"]


def extract_project_id(project: ProjectInfo) -> str | None:
    """从项目响应中提取项目标识(0.66 项目无 id,身份标识为 name)。"""
    return (
        project.get("id")
        or project.get("projectId")
        or project.get("project_id")
        or project.get("name")
    )


# 0.66 阶段(snake_case),按 Stage::ALL 顺序;legacy 步骤名据此映射。
_PIPELINE_STAGE_ORDER = ("detection", "ocr", "translation", "inpainting")

# 0.61 角色词 → 0.66 阶段;render 在 0.66 中不存在,映射为 None(忽略)。
_LEGACY_STAGE_KEYWORDS: tuple[tuple[str, str | None], ...] = (
    ("detect", "detection"),
    ("segment", "detection"),
    ("font", "detection"),
    ("ocr", "ocr"),
    ("translat", "translation"),
    ("inpaint", "inpainting"),
    ("render", None),
)


def _pipeline_stages_from_steps(steps: list[str]) -> list[str]:
    """把 legacy/引擎 id 步骤名映射为 0.66 阶段,按 Stage::ALL 顺序去重。"""
    stages: list[str] = []
    for step in steps:
        token = step.strip().lower()
        if token in _PIPELINE_STAGE_ORDER:
            stage: str | None = token
        else:
            stage = next(
                (
                    mapped
                    for keyword, mapped in _LEGACY_STAGE_KEYWORDS
                    if keyword in token
                ),
                None,
            )
        if stage is None:
            logger.warning("[koharu-client] ignoring unknown pipeline step %r", step)
            continue
        if stage not in stages:
            stages.append(stage)
    return stages


def _extract_operation_id(data: JsonValue) -> str:
    """从启动操作的响应中提取 operationId,缺失时抛 KoharuApiError。"""
    if not isinstance(data, dict):
        raise KoharuApiError(f"Koharu did not return operationId: {data!r}")
    operation_id = cast(OperationStartResponse, data).get("operationId")
    if not operation_id:
        raise KoharuApiError(f"Koharu did not return operationId: {data!r}")
    return str(operation_id)


def _parse_event_data(raw: str) -> JsonValue:
    try:
        return cast(JsonValue, json.loads(raw))
    except json.JSONDecodeError:
        return raw


def _normalize_operation_list(data: JsonValue) -> list[OperationInfo]:
    if isinstance(data, list):
        return [cast(OperationInfo, item) for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        for key in ("operations", "jobs", "items"):
            value = data.get(key)
            if isinstance(value, list):
                return [cast(OperationInfo, item) for item in value if isinstance(item, dict)]
        return [cast(OperationInfo, value) for value in data.values() if isinstance(value, dict)]
    return []


def _find_operation(
    operations: list[OperationInfo],
    operation_id: str,
) -> OperationInfo | None:
    for operation in operations:
        for value in (
            operation.get("id", ""),
            operation.get("operationId", ""),
            operation.get("jobId", ""),
        ):
            if str(value) == operation_id:
                return operation
    return None


def save_exported_images(
    content: bytes,
    content_type: str,
    output_dir: str | os.PathLike[str],
    *,
    base_name: str = "translated",
) -> list[str]:
    """Save Koharu rendered export bytes and return image file paths."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if "zip" in content_type.lower() or content.startswith(b"PK\x03\x04"):
        return _save_images_from_zip(content, output_path)

    suffix = _suffix_from_content_type(content_type) or ".png"
    target = output_path / f"{base_name}{suffix}"
    target.write_bytes(content)
    return [str(target)]


def _save_images_from_zip(content: bytes, output_dir: Path) -> list[str]:
    archive_path = output_dir / "koharu-rendered.zip"
    archive_path.write_bytes(content)
    saved: list[str] = []
    with zipfile.ZipFile(archive_path) as archive:
        for index, member in enumerate(archive.infolist(), start=1):
            if member.is_dir():
                continue
            suffix = Path(member.filename).suffix.lower()
            if suffix not in IMAGE_EXTENSIONS:
                continue
            target = output_dir / f"{index:03d}-{Path(member.filename).name}"
            with archive.open(member) as source:
                target.write_bytes(source.read())
            saved.append(str(target))
    if not saved:
        raise KoharuApiError("Koharu export zip did not contain image files")
    return saved


def _suffix_from_content_type(content_type: str) -> str | None:
    content_type = content_type.split(";", 1)[0].strip().lower()
    if content_type == "image/jpeg":
        return ".jpg"
    if content_type.startswith("image/"):
        suffix = mimetypes.guess_extension(content_type)
        return suffix or f".{content_type.removeprefix('image/')}"
    return None
