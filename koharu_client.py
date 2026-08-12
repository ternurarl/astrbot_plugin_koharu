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
    """Shape of a Koharu project."""

    id: str
    projectId: str
    project_id: str


class OperationInfo(TypedDict, total=False):
    """Shape of a Koharu operation (pipeline / download job)."""

    id: str
    operationId: str
    jobId: str
    status: str
    error: str


class OperationStartResponse(TypedDict, total=False):
    """Shape of the response returned when an operation is started."""

    operationId: str


class PageCreateResponse(TypedDict, total=False):
    """Shape of the response of page creation endpoints."""

    pages: list[str]


class LLMCurrentState(TypedDict, total=False):
    """Shape of GET /llm/current."""

    status: str
    kind: str
    providerId: str
    modelId: str


class PipelineConfig(TypedDict, total=False):
    """Shape of the ``pipeline`` section of the Koharu config."""

    detector: str
    fontDetector: str
    font_detector: str
    segmenter: str
    bubbleSegmenter: str
    bubble_segmenter: str
    ocr: str
    translator: str
    inpainter: str
    renderer: str


class KoharuConfig(TypedDict, total=False):
    """Shape of GET /config (only the fields the plugin consumes)."""

    pipeline: PipelineConfig


class EnginesResponse(TypedDict, total=False):
    """Best-effort shape of GET /engines (not consumed by the plugin)."""

    engines: list["EngineInfo"]


class EngineInfo(TypedDict, total=False):
    id: str
    name: str


class FontsResponse(TypedDict, total=False):
    """Best-effort shape of GET /fonts (not consumed by the plugin)."""

    fonts: list["FontInfo"]


class FontInfo(TypedDict, total=False):
    id: str
    name: str


class GoogleFontsResponse(TypedDict, total=False):
    """Best-effort shape of GET /google-fonts (not consumed by the plugin)."""

    fonts: list["GoogleFontInfo"]


class GoogleFontInfo(TypedDict, total=False):
    id: str
    family: str


class ProjectsResponse(TypedDict, total=False):
    """Best-effort shape of GET /projects (not consumed by the plugin)."""

    projects: list[ProjectInfo]


class SceneResponse(TypedDict, total=False):
    """Best-effort shape of GET /scene.json (not consumed by the plugin)."""

    version: int
    pages: list[JsonValue]


class PageLayerInfo(TypedDict, total=False):
    """Best-effort shape of a page image layer (not consumed by the plugin)."""

    id: str
    pageId: str
    kind: str


class DownloadsResponse(TypedDict, total=False):
    """Best-effort shape of GET /downloads (not consumed by the plugin)."""

    downloads: list["DownloadInfo"]


class DownloadInfo(TypedDict, total=False):
    id: str
    modelId: str
    status: str


class CatalogResponse(TypedDict, total=False):
    """Best-effort shape of GET /llm/catalog (not consumed by the plugin)."""

    providers: list["LLMProviderInfo"]
    models: list["LLMModelInfo"]


class LLMProviderInfo(TypedDict, total=False):
    id: str
    name: str


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


class PagePathsBody(TypedDict, total=False):
    paths: list[str]
    replace: bool


class RegionSpec(TypedDict, total=False):
    """Opaque region specification, passed through to the pipeline request."""


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
    temperature: float
    maxTokens: int
    customSystemPrompt: str


class ProviderSecretBody(TypedDict, total=False):
    apiKey: str


class DownloadStartBody(TypedDict, total=False):
    modelId: str


class HistoryOp(TypedDict, total=False):
    """Opaque history operation payload, passed through to the Koharu API."""

    op: str
    data: JsonValue


class PatchBody(TypedDict, total=False):
    """Opaque config patch payload, passed through to the Koharu API."""

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

    async def get_engines(self) -> EnginesResponse:
        data = await self._json("GET", "/engines")
        return cast(EnginesResponse, data) if isinstance(data, dict) else {}

    # Fonts
    async def get_fonts(self) -> FontsResponse:
        data = await self._json("GET", "/fonts")
        return cast(FontsResponse, data) if isinstance(data, dict) else {}

    async def get_google_fonts(self) -> GoogleFontsResponse:
        data = await self._json("GET", "/google-fonts")
        return cast(GoogleFontsResponse, data) if isinstance(data, dict) else {}

    async def fetch_google_font(self, family: str) -> None:
        await self._request("POST", f"/google-fonts/{family}/fetch")

    async def get_google_font_file(self, family: str, file_name: str) -> bytes:
        response = await self._request("GET", f"/google-fonts/{family}/{file_name}")
        return response.content

    # Projects
    async def list_projects(self) -> ProjectsResponse:
        data = await self._json("GET", "/projects")
        return cast(ProjectsResponse, data) if isinstance(data, dict) else {}

    async def create_project(self, name: str) -> ProjectInfo:
        body: dict[str, JsonValue] = {"name": name}
        data = await self._json("POST", "/projects", json=body)
        return cast(ProjectInfo, data) if isinstance(data, dict) else {}

    async def import_project_archive(self, archive_path: str | os.PathLike[str]) -> ProjectInfo:
        path = Path(archive_path)
        with path.open("rb") as file_obj:
            files: FilesPayload = {"file": (path.name, file_obj, "application/octet-stream")}
            data = await self._json("POST", "/projects/import", files=files)
        return cast(ProjectInfo, data) if isinstance(data, dict) else {}

    async def open_project(self, project_id: str) -> ProjectInfo:
        body: dict[str, JsonValue] = {"id": project_id}
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

    async def delete_project(self, project_id: str) -> None:
        await self._json("DELETE", f"/projects/{project_id}")

    async def delete_project_if_possible(self, project_id: str) -> bool:
        response = await self._request(
            "DELETE",
            f"/projects/{project_id}",
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
        *,
        replace: bool = False,
    ) -> PageCreateResponse:
        opened: list[BinaryIO] = []
        try:
            files: list[tuple[str, tuple[str, BinaryIO, str]]] = []
            for image_path in image_paths:
                path = Path(image_path)
                file_obj = path.open("rb")
                opened.append(file_obj)
                content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
                files.append(("files", (path.name, file_obj, content_type)))
            data: dict[str, str] | None = {"replace": "true"} if replace else None
            response_data = await self._json("POST", "/pages", files=files, data=data)
            if not isinstance(response_data, dict):
                raise KoharuApiError(
                    f"Koharu create_pages returned an unexpected response: {response_data!r}"
                )
            return cast(PageCreateResponse, response_data)
        finally:
            for file_obj in opened:
                file_obj.close()

    async def create_pages_from_paths(
        self,
        image_paths: Sequence[str | os.PathLike[str]],
        *,
        replace: bool = False,
    ) -> PageCreateResponse:
        body: dict[str, JsonValue] = {
            "paths": cast(JsonValue, [str(Path(path)) for path in image_paths]),
            "replace": replace,
        }
        data = await self._json("POST", "/pages/from-paths", json=body)
        if not isinstance(data, dict):
            raise KoharuApiError(
                f"Koharu create_pages_from_paths returned an unexpected response: {data!r}"
            )
        return cast(PageCreateResponse, data)

    async def add_page_image_layer(
        self,
        page_id: str,
        image_path: str | os.PathLike[str],
    ) -> PageLayerInfo:
        path = Path(image_path)
        with path.open("rb") as file_obj:
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            files: FilesPayload = {"file": (path.name, file_obj, content_type)}
            data = await self._json("POST", f"/pages/{page_id}/image-layers", files=files)
        return cast(PageLayerInfo, data) if isinstance(data, dict) else {}

    async def upsert_page_mask(
        self,
        page_id: str,
        role: str,
        png_bytes: bytes,
    ) -> None:
        await self._request(
            "PUT",
            f"/pages/{page_id}/masks/{role}",
            content=png_bytes,
            headers={"content-type": "image/png"},
        )

    async def get_page_thumbnail(self, page_id: str) -> bytes:
        response = await self._request("GET", f"/pages/{page_id}/thumbnail")
        return response.content

    # Scene and blobs
    async def get_scene_json(self) -> SceneResponse:
        data = await self._json("GET", "/scene.json")
        return cast(SceneResponse, data) if isinstance(data, dict) else {}

    async def get_scene_bin(self) -> tuple[bytes, str | None]:
        response = await self._request("GET", "/scene.bin")
        return response.content, response.headers.get("x-koharu-epoch")

    async def get_blob(self, blob_hash: str) -> bytes:
        response = await self._request("GET", f"/blobs/{blob_hash}")
        return response.content

    # History
    async def apply_history_op(self, op: HistoryOp) -> HistoryOp:
        data = await self._json("POST", "/history/apply", json=cast(JsonValue, op))
        return cast(HistoryOp, data) if isinstance(data, dict) else {}

    async def undo_history(self) -> HistoryOp:
        data = await self._json("POST", "/history/undo")
        return cast(HistoryOp, data) if isinstance(data, dict) else {}

    async def redo_history(self) -> HistoryOp:
        data = await self._json("POST", "/history/redo")
        return cast(HistoryOp, data) if isinstance(data, dict) else {}

    # Pipelines
    async def start_pipeline(
        self,
        steps: list[str],
        *,
        pages: list[str] | None = None,
        region: RegionSpec | None = None,
        target_language: str | None = None,
        system_prompt: str | None = None,
        default_font: str | None = None,
    ) -> str:
        body: dict[str, JsonValue] = {"steps": cast(JsonValue, steps)}
        if pages:
            body["pages"] = cast(JsonValue, pages)
        if region:
            body["region"] = cast(JsonValue, region)
        if target_language:
            body["target_language"] = target_language
            body["targetLanguage"] = target_language
        if system_prompt:
            body["system_prompt"] = system_prompt
            body["systemPrompt"] = system_prompt
        if default_font:
            body["default_font"] = default_font
            body["defaultFont"] = default_font
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

    # Downloads
    async def list_downloads(self) -> DownloadsResponse:
        data = await self._json("GET", "/downloads")
        return cast(DownloadsResponse, data) if isinstance(data, dict) else {}

    async def start_download(self, model_id: str) -> str:
        body: dict[str, JsonValue] = {"modelId": model_id}
        data = await self._json("POST", "/downloads", json=body)
        return _extract_operation_id(data)

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
        body: dict[str, JsonValue] = {"target": cast(JsonValue, target)}
        if options:
            body["options"] = cast(JsonValue, options)
        await self._json("PUT", "/llm/current", json=body, expected_status={204})

    async def unload_llm(self) -> None:
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
        body: dict[str, JsonValue] = {"apiKey": secret}
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
        config = await self.get_config()
        pipeline = config.get("pipeline")
        if pipeline is None:
            return []
        steps: list[str] = []
        for value in (
            pipeline.get("detector"),
            pipeline.get("fontDetector") or pipeline.get("font_detector"),
            pipeline.get("segmenter"),
            pipeline.get("bubbleSegmenter") or pipeline.get("bubble_segmenter"),
            pipeline.get("ocr"),
            pipeline.get("translator"),
            pipeline.get("inpainter"),
            pipeline.get("renderer"),
        ):
            if value and value not in steps:
                steps.append(value)
        return steps


def extract_project_id(project: ProjectInfo) -> str | None:
    """从项目响应中提取项目 ID(兼容 id/projectId/project_id 三种键名)。"""
    return project.get("id") or project.get("projectId") or project.get("project_id")


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
