from __future__ import annotations

import asyncio
import json
import mimetypes
import os
import time
import zipfile
from collections.abc import AsyncIterator, Iterable
from pathlib import Path
from typing import Any

import httpx

try:
    from astrbot.api import logger
except Exception:  # pragma: no cover - used when the client is run standalone.
    import logging

    logger = logging.getLogger(__name__)


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}


class KoharuApiError(RuntimeError):
    """Raised when Koharu returns an error or a pipeline job fails."""


class KoharuTimeoutError(TimeoutError):
    """Raised when Koharu does not complete an operation in time."""


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
        **kwargs: Any,
    ) -> httpx.Response:
        started = time.monotonic()
        logger.debug("[koharu-client] request %s %s", method, path)
        response = await self._client.request(method, path, **kwargs)
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
        **kwargs: Any,
    ) -> Any:
        response = await self._request(
            method,
            path,
            expected_status=expected_status,
            **kwargs,
        )
        if response.status_code == 204 or not response.content:
            return None
        return response.json()

    async def wait_until_ready(
        self,
        *,
        timeout_seconds: float = 60.0,
        interval_seconds: float = 1.0,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_seconds
        last_error: Exception | None = None
        logger.debug("[koharu-client] wait_until_ready timeout_seconds=%s", timeout_seconds)
        while time.monotonic() < deadline:
            try:
                response = await self._client.get("/meta")
                logger.debug(
                    "[koharu-client] wait_until_ready /meta status=%s",
                    response.status_code,
                )
                if response.status_code == 200:
                    return response.json()
                if response.status_code != 503:
                    response.raise_for_status()
            except Exception as exc:  # Koharu may still be starting.
                last_error = exc
            await asyncio.sleep(interval_seconds)
        detail = f": {last_error}" if last_error else ""
        raise KoharuTimeoutError(f"Koharu is not ready within {timeout_seconds}s{detail}")

    # Meta
    async def get_meta(self) -> dict[str, Any]:
        return await self._json("GET", "/meta")

    async def get_engines(self) -> dict[str, Any]:
        return await self._json("GET", "/engines")

    # Fonts
    async def get_fonts(self) -> Any:
        return await self._json("GET", "/fonts")

    async def get_google_fonts(self) -> Any:
        return await self._json("GET", "/google-fonts")

    async def fetch_google_font(self, family: str) -> Any:
        return await self._json("POST", f"/google-fonts/{family}/fetch")

    async def get_google_font_file(self, family: str, file_name: str) -> bytes:
        response = await self._request("GET", f"/google-fonts/{family}/{file_name}")
        return response.content

    # Projects
    async def list_projects(self) -> Any:
        return await self._json("GET", "/projects")

    async def create_project(self, name: str) -> dict[str, Any]:
        return await self._json("POST", "/projects", json={"name": name})

    async def import_project_archive(self, archive_path: str | os.PathLike[str]) -> Any:
        path = Path(archive_path)
        with path.open("rb") as file_obj:
            files = {"file": (path.name, file_obj, "application/octet-stream")}
            return await self._json("POST", "/projects/import", files=files)

    async def open_project(self, project_id: str) -> Any:
        return await self._json("PUT", "/projects/current", json={"id": project_id})

    async def close_project(self) -> None:
        await self._json("DELETE", "/projects/current")

    async def close_project_if_any(self) -> bool:
        response = await self._request(
            "DELETE",
            "/projects/current",
            expected_status={200, 202, 204, 400, 404, 409},
        )
        return 200 <= response.status_code < 300

    async def export_project(
        self,
        export_format: str = "rendered",
        *,
        pages: list[str] | None = None,
    ) -> tuple[bytes, str]:
        body: dict[str, Any] = {"format": export_format}
        if pages:
            body["pages"] = pages
        response = await self._request("POST", "/projects/current/export", json=body)
        return response.content, response.headers.get("content-type", "")

    # Pages
    async def create_pages(
        self,
        image_paths: list[str | os.PathLike[str]],
        *,
        replace: bool = False,
    ) -> Any:
        opened: list[Any] = []
        try:
            files = []
            for image_path in image_paths:
                path = Path(image_path)
                file_obj = path.open("rb")
                opened.append(file_obj)
                content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
                files.append(("files", (path.name, file_obj, content_type)))
            data = {"replace": "true"} if replace else None
            return await self._json("POST", "/pages", files=files, data=data)
        finally:
            for file_obj in opened:
                file_obj.close()

    async def create_pages_from_paths(
        self,
        image_paths: list[str | os.PathLike[str]],
        *,
        replace: bool = False,
    ) -> Any:
        return await self._json(
            "POST",
            "/pages/from-paths",
            json={"paths": [str(Path(path)) for path in image_paths], "replace": replace},
        )

    async def add_page_image_layer(
        self,
        page_id: str,
        image_path: str | os.PathLike[str],
    ) -> Any:
        path = Path(image_path)
        with path.open("rb") as file_obj:
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            files = {"file": (path.name, file_obj, content_type)}
            return await self._json("POST", f"/pages/{page_id}/image-layers", files=files)

    async def upsert_page_mask(
        self,
        page_id: str,
        role: str,
        png_bytes: bytes,
    ) -> Any:
        return await self._json(
            "PUT",
            f"/pages/{page_id}/masks/{role}",
            content=png_bytes,
            headers={"content-type": "image/png"},
        )

    async def get_page_thumbnail(self, page_id: str) -> bytes:
        response = await self._request("GET", f"/pages/{page_id}/thumbnail")
        return response.content

    # Scene and blobs
    async def get_scene_json(self) -> dict[str, Any]:
        return await self._json("GET", "/scene.json")

    async def get_scene_bin(self) -> tuple[bytes, str | None]:
        response = await self._request("GET", "/scene.bin")
        return response.content, response.headers.get("x-koharu-epoch")

    async def get_blob(self, blob_hash: str) -> bytes:
        response = await self._request("GET", f"/blobs/{blob_hash}")
        return response.content

    # History
    async def apply_history_op(self, op: dict[str, Any]) -> dict[str, Any]:
        return await self._json("POST", "/history/apply", json=op)

    async def undo_history(self) -> dict[str, Any]:
        return await self._json("POST", "/history/undo")

    async def redo_history(self) -> dict[str, Any]:
        return await self._json("POST", "/history/redo")

    # Pipelines
    async def start_pipeline(
        self,
        steps: list[str],
        *,
        pages: list[str] | None = None,
        region: dict[str, Any] | None = None,
        target_language: str | None = None,
        system_prompt: str | None = None,
        default_font: str | None = None,
    ) -> str:
        body: dict[str, Any] = {"steps": steps}
        if pages:
            body["pages"] = pages
        if region:
            body["region"] = region
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
        operation_id = data.get("operationId") if isinstance(data, dict) else None
        if not operation_id:
            raise KoharuApiError(f"Koharu did not return operationId: {data!r}")
        return str(operation_id)

    # Operations
    async def list_operations(self) -> list[dict[str, Any]]:
        data = await self._json("GET", "/operations")
        return _normalize_operation_list(data)

    async def cancel_operation(self, operation_id: str) -> Any:
        return await self._json("DELETE", f"/operations/{operation_id}")

    async def wait_operation(
        self,
        operation_id: str,
        *,
        timeout_seconds: float = 900.0,
        interval_seconds: float = 2.0,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_seconds
        last_seen: dict[str, Any] | None = None
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
    async def list_downloads(self) -> Any:
        return await self._json("GET", "/downloads")

    async def start_download(self, model_id: str) -> str:
        data = await self._json("POST", "/downloads", json={"modelId": model_id})
        operation_id = data.get("operationId") if isinstance(data, dict) else None
        if not operation_id:
            raise KoharuApiError(f"Koharu did not return operationId: {data!r}")
        return str(operation_id)

    # LLM control
    async def get_llm_current(self) -> Any:
        return await self._json("GET", "/llm/current")

    async def load_llm(
        self,
        target: dict[str, Any],
        *,
        options: dict[str, Any] | None = None,
    ) -> None:
        body: dict[str, Any] = {"target": dict(target)}
        if options:
            body["options"] = options
        await self._json("PUT", "/llm/current", json=body, expected_status={204})

    async def unload_llm(self) -> None:
        await self._request(
            "DELETE",
            "/llm/current",
            expected_status={200, 202, 204, 400, 404, 409},
        )

    async def get_llm_catalog(self) -> Any:
        return await self._json("GET", "/llm/catalog")

    # Config
    async def get_config(self) -> dict[str, Any]:
        return await self._json("GET", "/config")

    async def patch_config(self, patch: dict[str, Any]) -> Any:
        return await self._json("PATCH", "/config", json=patch)

    async def set_provider_secret(self, provider_id: str, secret: str) -> Any:
        return await self._json(
            "PUT",
            f"/config/providers/{provider_id}/secret",
            json={"apiKey": secret},
        )

    async def clear_provider_secret(self, provider_id: str) -> Any:
        return await self._json("DELETE", f"/config/providers/{provider_id}/secret")

    # Events
    async def iter_events(
        self,
        *,
        last_event_id: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
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
        pipeline = config.get("pipeline", {}) if isinstance(config, dict) else {}
        if not isinstance(pipeline, dict):
            return []
        ordered_keys = [
            ("detector",),
            ("fontDetector", "font_detector"),
            ("segmenter",),
            ("bubbleSegmenter", "bubble_segmenter"),
            ("ocr",),
            ("translator",),
            ("inpainter",),
            ("renderer",),
        ]
        steps: list[str] = []
        for aliases in ordered_keys:
            value = next(
                (
                    pipeline[key]
                    for key in aliases
                    if isinstance(pipeline.get(key), str) and pipeline.get(key)
                ),
                None,
            )
            if isinstance(value, str) and value and value not in steps:
                steps.append(value)
        return steps


def _parse_event_data(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _normalize_operation_list(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        for key in ("operations", "jobs", "items"):
            value = data.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        return [value for value in data.values() if isinstance(value, dict)]
    return []


def _find_operation(
    operations: list[dict[str, Any]],
    operation_id: str,
) -> dict[str, Any] | None:
    for operation in operations:
        for key in ("id", "operationId", "jobId"):
            if str(operation.get(key, "")) == operation_id:
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

