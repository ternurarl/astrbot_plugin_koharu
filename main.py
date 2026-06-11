from __future__ import annotations

import asyncio
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, register
import astrbot.api.message_components as Comp
from astrbot.core.utils.session_waiter import SessionController, session_waiter

try:
    from astrbot.core.utils.astrbot_path import get_astrbot_data_path
except Exception:  # pragma: no cover - compatibility fallback for older AstrBot.
    get_astrbot_data_path = None

try:
    from .koharu_client import KoharuApiError, KoharuClient, save_exported_images
except ImportError:  # AstrBot may load plugin files without package context.
    from koharu_client import KoharuApiError, KoharuClient, save_exported_images


PLUGIN_NAME = "astrbot_plugin_koharu"


@register(
    PLUGIN_NAME,
    "ABCwewe+CodeX",
    "使用 Koharu HTTP API 翻译聊天中的漫画图片。",
    "1.3.0",
)
class KoharuMangaTranslatorPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig | None = None):
        super().__init__(context)
        self.config = config or {}
        self._translate_lock = asyncio.Lock()
        self._data_dir = self._resolve_data_dir()
        self._queue_semaphore = asyncio.Semaphore(self._int_conf("queue_depth") + 1)

    async def initialize(self) -> None:
        self._data_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(
            "[koharu-plugin] initialized data_dir=%s api_base=%s target_language=%s",
            self._data_dir,
            self._str_conf("koharu_api_base_url"),
            self._str_conf("target_language"),
        )
        self._cleanup_output_cache()

    @filter.command("漫画翻译", alias={"manga_translate", "manga-translate"})
    async def manga_translate(
        self,
        event: AstrMessageEvent,
        target_language: str = "",
    ):
        """Translate manga image(s) with Koharu. Usage: /漫画翻译 [target_language] + image(s)."""

        event.stop_event()
        logger.info(
            "[koharu-plugin] command triggered sender=%s session=%s message=%r",
            event.get_sender_id(),
            event.get_session_id(),
            event.message_str,
        )
        target_language = (target_language or self._str_conf("target_language")).strip()
        image_paths = await self._extract_image_paths(event)
        logger.debug(
            "[koharu-plugin] command image extraction done count=%d target_language=%s",
            len(image_paths),
            target_language,
        )

        if image_paths:
            if self._queue_semaphore.locked():
                logger.info("[koharu-plugin] queue full; rejecting immediate request")
                await event.send(
                    event.plain_result(
                        f"翻译队列已满（最大等待 {self._int_conf('queue_depth')} 个），请稍后再试。"
                    )
                )
                return
            await self._queue_semaphore.acquire()
            logger.info("[koharu-plugin] sending accepted message before translation")
            await event.send(
                event.plain_result(
                    f"已收到 {len(image_paths)} 张图片，开始调用 Koharu 翻译为 {target_language}。"
                )
            )
            logger.info("[koharu-plugin] accepted message sent; starting translation")
            try:
                output_paths = await self._translate_images(image_paths, target_language)
            except Exception as exc:
                logger.exception("Koharu manga translation failed")
                await event.send(event.plain_result(f"漫画翻译失败：{exc}"))
                return
            finally:
                self._release_queue()
            logger.info(
                "[koharu-plugin] translation finished; sending output count=%d",
                len(output_paths),
            )
            await event.send(self._build_image_result(event, output_paths))
            self._cleanup_current_outputs_if_needed(output_paths)
            self._cleanup_output_cache()
            return

        await event.send(
            event.plain_result(
                f"请在 {self._int_conf('wait_image_timeout_seconds')} 秒内发送需要翻译的漫画图片。"
                "发送“取消”可退出。"
            )
        )

        @session_waiter(
            timeout=self._int_conf("wait_image_timeout_seconds"),
            record_history_chains=False,
        )
        async def wait_for_images(
            controller: SessionController,
            next_event: AstrMessageEvent,
        ) -> None:
            next_event.stop_event()
            logger.debug(
                "[koharu-plugin] waiter received event sender=%s session=%s message=%r",
                next_event.get_sender_id(),
                next_event.get_session_id(),
                next_event.message_str,
            )
            if (next_event.message_str or "").strip().lower() in {"取消", "退出", "cancel"}:
                await next_event.send(next_event.plain_result("已取消漫画翻译。"))
                controller.stop()
                return

            next_image_paths = await self._extract_image_paths(next_event)
            logger.debug(
                "[koharu-plugin] waiter image extraction done count=%d",
                len(next_image_paths),
            )
            if not next_image_paths:
                logger.debug("[koharu-plugin] waiter got no images; keep waiting")
                await next_event.send(
                    next_event.plain_result("未检测到图片，请重新发送图片或发送“取消”。")
                )
                controller.keep(
                    timeout=self._int_conf("wait_image_timeout_seconds"),
                    reset_timeout=True,
                )
                return

            await next_event.send(
                next_event.plain_result(
                    f"已收到 {len(next_image_paths)} 张图片，开始调用 Koharu 翻译为 {target_language}。"
                )
            )
            if self._queue_semaphore.locked():
                logger.info("[koharu-plugin] queue full; rejecting waiter request")
                await next_event.send(
                    next_event.plain_result(
                        f"翻译队列已满（最大等待 {self._int_conf('queue_depth')} 个），请稍后再试。"
                    )
                )
                controller.stop()
                return
            await self._queue_semaphore.acquire()
            try:
                logger.info("[koharu-plugin] waiter starting translation")
                output_paths = await self._translate_images(next_image_paths, target_language)
                logger.info(
                    "[koharu-plugin] waiter translation finished; sending output count=%d",
                    len(output_paths),
                )
                await next_event.send(self._build_image_result(next_event, output_paths))
                self._cleanup_current_outputs_if_needed(output_paths)
                self._cleanup_output_cache()
            except Exception as exc:
                logger.exception("Koharu manga translation failed")
                await next_event.send(next_event.plain_result(f"漫画翻译失败：{exc}"))
            finally:
                self._release_queue()
                controller.stop()

        try:
            logger.debug("[koharu-plugin] registering session waiter for image input")
            await wait_for_images(event)
        except TimeoutError:
            logger.info("[koharu-plugin] waiter timeout")
            await event.send(event.plain_result("等待图片超时，已退出漫画翻译。"))

    async def _translate_images(
        self,
        image_paths: list[str],
        target_language: str,
    ) -> list[str]:
        logger.debug(
            "[koharu-plugin] translate requested image_count=%d target_language=%s",
            len(image_paths),
            target_language,
        )
        max_images = self._int_conf("max_images_per_request")
        if max_images > 0 and len(image_paths) > max_images:
            raise ValueError(f"单次最多支持 {max_images} 张图片，请减少图片数量后重试。")

        logger.debug("[koharu-plugin] waiting for translate lock")
        async with self._translate_lock:
            request_id = f"{int(time.time())}-{uuid.uuid4().hex[:8]}"
            output_dir = self._data_dir / "outputs" / request_id
            project_name = f"astrbot-koharu-{request_id}"
            logger.debug(
                "[koharu-plugin] translate lock acquired request_id=%s project=%s output_dir=%s api_base=%s",
                request_id,
                project_name,
                output_dir,
                self._str_conf("koharu_api_base_url"),
            )

            async with KoharuClient(
                self._str_conf("koharu_api_base_url"),
                timeout=float(self._int_conf("http_timeout_seconds")),
                connect_timeout=float(self._int_conf("http_connect_timeout_seconds")),
            ) as client:
                logger.debug("[koharu-plugin] waiting koharu ready")
                await client.wait_until_ready(
                    timeout_seconds=float(self._int_conf("koharu_ready_timeout_seconds"))
                )
                logger.debug("[koharu-plugin] closing existing koharu project before creating a new one")
                closed_existing = await client.close_project_if_any()
                logger.debug(
                    "[koharu-plugin] close existing project result closed=%s",
                    closed_existing,
                )
                logger.debug("[koharu-plugin] koharu ready; creating project")
                project = await client.create_project(project_name)
                logger.debug("[koharu-plugin] project created response=%s", project)
                project_id = (
                    project.get("id")
                    or project.get("projectId")
                    or project.get("project_id")
                )
                logger.debug("[koharu-plugin] project_id=%s", project_id)
                try:
                    cached_image_paths, upload_cache_dir = self._cache_ordered_upload_images(
                        image_paths
                    )
                    try:
                        logger.debug(
                            "[koharu-plugin] uploading pages count=%d cache_dir=%s paths=%s",
                            len(cached_image_paths),
                            upload_cache_dir,
                            [_safe_path(path) for path in cached_image_paths],
                        )
                        pages = await client.create_pages(cached_image_paths, replace=True)
                    finally:
                        self._delete_upload_cache(upload_cache_dir)
                    logger.debug("[koharu-plugin] pages uploaded response=%s", pages)
                    await self._maybe_load_llm(client)
                    steps = await self._resolve_pipeline_steps(client)
                    logger.debug("[koharu-plugin] resolved pipeline steps=%s", steps)
                    if not steps:
                        raise KoharuApiError(
                            "未能确定 Koharu pipeline steps。请在插件配置中填写 "
                            "pipeline_steps，或先在 Koharu 配置中选择 pipeline 引擎。"
                        )
                    logger.info("[koharu-plugin] starting pipeline")
                    operation_id = await client.start_pipeline(
                        steps,
                        target_language=target_language,
                        system_prompt=self._str_conf("system_prompt") or None,
                        default_font=self._str_conf("default_font") or None,
                    )
                    logger.info("[koharu-plugin] pipeline started operation_id=%s", operation_id)
                    operation = await client.wait_operation(
                        operation_id,
                        timeout_seconds=float(self._int_conf("pipeline_timeout_seconds")),
                        interval_seconds=float(self._int_conf("operation_poll_interval_seconds")),
                    )
                    logger.info("[koharu-plugin] pipeline completed operation=%s", operation)
                    logger.debug("[koharu-plugin] exporting rendered project")
                    content, content_type = await client.export_project("rendered")
                    logger.debug(
                        "[koharu-plugin] export received bytes=%d content_type=%s",
                        len(content),
                        content_type,
                    )
                    output_paths = save_exported_images(
                        content,
                        content_type,
                        output_dir,
                        base_name="translated",
                    )
                    output_paths = self._compress_output_images_if_enabled(
                        output_paths,
                        output_dir,
                    )
                    logger.debug(
                        "[koharu-plugin] export saved output_count=%d paths=%s",
                        len(output_paths),
                        [_safe_path(path) for path in output_paths],
                    )
                    return output_paths
                finally:
                    if self._bool_conf("close_project_after_export"):
                        try:
                            logger.debug("[koharu-plugin] closing koharu project")
                            await client.close_project()
                            logger.debug("[koharu-plugin] koharu project closed")
                        except Exception as exc:
                            logger.warning(f"Failed to close Koharu project: {exc}")
                    if self._bool_conf("delete_project_after_export") and project_id:
                        try:
                            logger.debug(
                                "[koharu-plugin] deleting koharu project project_id=%s",
                                project_id,
                            )
                            await client.delete_project(str(project_id))
                            logger.debug("[koharu-plugin] koharu project deleted")
                        except Exception as exc:
                            logger.warning(f"Failed to delete Koharu project: {exc}")

    async def _maybe_load_llm(self, client: KoharuClient) -> None:
        if not self._bool_conf("auto_load_llm"):
            logger.debug("[koharu-plugin] auto_load_llm disabled; skip llm load")
            return

        model_id = self._str_conf("llm_model_id").strip()
        if not model_id:
            logger.debug("[koharu-plugin] auto_load_llm enabled but llm_model_id empty; skip llm load")
            return

        llm_kind = self._str_conf("llm_kind").strip().lower()
        logger.debug(
            "[koharu-plugin] loading llm kind=%s provider=%s model=%s",
            llm_kind,
            self._str_conf("llm_provider_id"),
            model_id,
        )
        if llm_kind == "provider":
            provider_id = self._str_conf("llm_provider_id").strip()
            if not provider_id:
                raise ValueError("auto_load_llm 已启用，但 llm_provider_id 为空。")
            target = {
                "kind": "provider",
                "providerId": provider_id,
                "modelId": model_id,
            }
        elif llm_kind == "local":
            target = {"kind": "local", "modelId": model_id}
        else:
            raise ValueError("llm_kind 只能是 local 或 provider。")

        options: dict[str, Any] = {}
        temperature = self._float_conf("llm_temperature")
        if temperature >= 0:
            options["temperature"] = temperature
        max_tokens = self._int_conf("llm_max_tokens")
        if max_tokens > 0:
            options["maxTokens"] = max_tokens
        custom_prompt = self._str_conf("llm_custom_system_prompt").strip()
        if custom_prompt:
            options["customSystemPrompt"] = custom_prompt

        logger.debug("[koharu-plugin] unloading current llm before load")
        await client.unload_llm()
        logger.debug("[koharu-plugin] sending llm load request target=%s options=%s", target, options)
        await client.load_llm(target, options=options or None)
        await self._wait_llm_ready(client)
        logger.debug("[koharu-plugin] llm ready")

    async def _wait_llm_ready(self, client: KoharuClient) -> None:
        deadline = time.monotonic() + float(self._int_conf("llm_load_timeout_seconds"))
        last_state: Any = None
        while time.monotonic() < deadline:
            last_state = await client.get_llm_current()
            status = ""
            if isinstance(last_state, dict):
                status = str(last_state.get("status", "")).lower()
                logger.debug("[koharu-plugin] llm current status=%s state=%s", status, last_state)
                if status in {"loaded", "ready", "running"}:
                    return
                if status in {"failed", "error"}:
                    raise KoharuApiError(f"Koharu LLM load failed: {last_state}")
            await asyncio.sleep(1)
        raise TimeoutError(f"Koharu LLM did not become ready. Last state: {last_state}")

    async def _resolve_pipeline_steps(self, client: KoharuClient) -> list[str]:
        configured = self._str_conf("pipeline_steps").strip()
        if configured:
            steps = [step.strip() for step in configured.split(",") if step.strip()]
            logger.debug("[koharu-plugin] using configured pipeline_steps=%s", steps)
            return steps
        logger.debug("[koharu-plugin] pipeline_steps empty; reading koharu /config")
        steps = await client.get_pipeline_steps_from_config()
        logger.debug("[koharu-plugin] pipeline steps from koharu config=%s", steps)
        return steps

    async def _extract_image_paths(self, event: AstrMessageEvent) -> list[str]:
        paths: list[str] = []
        messages = event.get_messages()
        logger.debug(
            "[koharu-plugin] extracting images from message_chain component_count=%d component_types=%s",
            len(messages),
            [type(component).__name__ for component in messages],
        )
        for component in messages:
            await self._collect_image_paths(component, paths)
        deduped = _dedupe(paths)
        logger.debug(
            "[koharu-plugin] extracted image paths count=%d paths=%s",
            len(deduped),
            [_safe_path(path) for path in deduped],
        )
        return deduped

    async def _collect_image_paths(self, component: Any, paths: list[str]) -> None:
        if isinstance(component, Comp.Image):
            logger.debug(
                "[koharu-plugin] converting image component file=%r url=%r path=%r",
                getattr(component, "file", None),
                getattr(component, "url", None),
                getattr(component, "path", None),
            )
            path = await component.convert_to_file_path()
            logger.debug("[koharu-plugin] image component converted path=%s", _safe_path(path))
            paths.append(path)
            return

        # Some adapters put quoted message chains under Reply.chain.
        nested_chain = getattr(component, "chain", None)
        if isinstance(nested_chain, list):
            logger.debug(
                "[koharu-plugin] walking nested chain component=%s nested_count=%d",
                type(component).__name__,
                len(nested_chain),
            )
            for nested in nested_chain:
                await self._collect_image_paths(nested, paths)

    def _cache_ordered_upload_images(self, image_paths: list[str]) -> tuple[list[str], Path]:
        upload_cache_dir = self._data_dir / "uploads" / uuid.uuid4().hex
        cached_paths: list[str] = []
        try:
            upload_cache_dir.mkdir(parents=True, exist_ok=False)
            for index, image_path in enumerate(image_paths, start=1):
                source = Path(image_path)
                suffix = source.suffix or ".jpg"
                target = upload_cache_dir / f"{index}{suffix}"
                shutil.copy2(source, target)
                cached_paths.append(str(target))
            logger.debug(
                "[koharu-plugin] cached ordered upload images dir=%s paths=%s",
                upload_cache_dir,
                [_safe_path(path) for path in cached_paths],
            )
            return cached_paths, upload_cache_dir
        except Exception:
            self._delete_upload_cache(upload_cache_dir)
            raise

    def _delete_upload_cache(self, upload_cache_dir: Path) -> None:
        try:
            if upload_cache_dir.exists():
                shutil.rmtree(upload_cache_dir)
                logger.debug("[koharu-plugin] deleted upload cache dir=%s", upload_cache_dir)
        except Exception as exc:
            logger.warning(
                "[koharu-plugin] failed to delete upload cache dir=%s error=%s",
                upload_cache_dir,
                exc,
            )

    def _compress_output_images_if_enabled(
        self,
        output_paths: list[str],
        output_dir: Path,
    ) -> list[str]:
        if not self._bool_conf("compress_return_images"):
            return output_paths

        image_format = self._str_conf("return_image_format").strip().lower()
        if image_format not in {"jpg", "jpeg", "webp"}:
            logger.warning(
                "[koharu-plugin] invalid return_image_format=%r; fallback to webp",
                image_format,
            )
            image_format = "webp"

        extension = ".jpg" if image_format in {"jpg", "jpeg"} else ".webp"
        pillow_format = "JPEG" if extension == ".jpg" else "WEBP"
        quality = min(100, max(1, self._int_conf("return_image_quality")))

        logger.info(
            "[koharu-plugin] compressing return images count=%d format=%s quality=%d",
            len(output_paths),
            extension.lstrip("."),
            quality,
        )
        compressed_paths: list[str] = []
        for index, output_path in enumerate(output_paths, start=1):
            source = Path(output_path)
            target = output_dir / f"{source.stem}.compressed-{index}{extension}"
            try:
                _compress_image(source, target, pillow_format, quality)
                compressed_paths.append(str(target))
                try:
                    source.unlink()
                except OSError as exc:
                    logger.debug(
                        "[koharu-plugin] failed to delete uncompressed output path=%s error=%s",
                        source,
                        exc,
                    )
            except Exception as exc:
                logger.warning(
                    "[koharu-plugin] failed to compress output image path=%s error=%s; "
                    "using original image",
                    source,
                    exc,
                )
                compressed_paths.append(str(source))
        return compressed_paths

    def _build_image_result(self, event: AstrMessageEvent, output_paths: list[str]):
        logger.info(
            "[koharu-plugin] building image result output_count=%d paths=%s",
            len(output_paths),
            [_safe_path(path) for path in output_paths],
        )
        if len(output_paths) == 1:
            return event.image_result(output_paths[0]).stop_event()

        max_send = self._int_conf("max_send_images")
        selected = output_paths if max_send <= 0 else output_paths[:max_send]
        chain: list[Comp.BaseMessageComponent] = [
            Comp.Plain(f"Koharu 翻译完成，共 {len(output_paths)} 张。")
        ]
        if max_send > 0 and len(output_paths) > max_send:
            chain.append(Comp.Plain(f"当前配置最多发送 {max_send} 张，以下为前 {max_send} 张。"))
        for path in selected:
            chain.append(Comp.Image.fromFileSystem(path))
        return event.chain_result(chain).stop_event()

    def _cleanup_current_outputs_if_needed(self, output_paths: list[str]) -> None:
        if self._str_conf("result_retention_policy") != "none":
            return
        outputs_root = (self._data_dir / "outputs").resolve()
        for output_path in output_paths:
            path = Path(output_path)
            try:
                resolved = path.resolve()
                if not _is_relative_to(resolved, outputs_root):
                    logger.warning(
                        "[koharu-plugin] skip deleting output outside cache path=%s",
                        resolved,
                    )
                    continue
                if resolved.exists():
                    resolved.unlink()
                    logger.debug("[koharu-plugin] deleted non-retained output=%s", resolved)
                self._remove_empty_parents(resolved.parent, outputs_root)
            except Exception as exc:
                logger.warning(
                    "[koharu-plugin] failed to delete non-retained output %s: %s",
                    path,
                    exc,
                )

    def _cleanup_output_cache(self) -> None:
        policy = self._str_conf("result_retention_policy")
        outputs_root = self._data_dir / "outputs"
        if policy == "forever" or not outputs_root.exists():
            return
        if policy == "none":
            retention_seconds = 0
        else:
            retention_days = max(0, self._int_conf("result_retention_days"))
            retention_seconds = retention_days * 86400

        cutoff = time.time() - retention_seconds
        for child in outputs_root.iterdir():
            try:
                if child.stat().st_mtime > cutoff:
                    continue
                if child.is_dir():
                    shutil.rmtree(child)
                    logger.debug("[koharu-plugin] deleted expired output directory=%s", child)
                else:
                    child.unlink()
                    logger.debug("[koharu-plugin] deleted expired output file=%s", child)
            except Exception as exc:
                logger.warning(
                    "[koharu-plugin] failed to delete cached output %s: %s",
                    child,
                    exc,
                )

    def _remove_empty_parents(self, start: Path, stop: Path) -> None:
        current = start.resolve()
        stop = stop.resolve()
        while _is_relative_to(current, stop) and current != stop:
            try:
                current.rmdir()
            except OSError:
                return
            current = current.parent

    def _resolve_data_dir(self) -> Path:
        if get_astrbot_data_path is not None:
            return Path(get_astrbot_data_path()) / "plugin_data" / PLUGIN_NAME
        return Path.cwd() / "data" / "plugin_data" / PLUGIN_NAME

    def _str_conf(self, key: str) -> str:
        value = self.config.get(key, DEFAULT_CONFIG[key])
        return str(value)

    def _int_conf(self, key: str) -> int:
        value = self.config.get(key, DEFAULT_CONFIG[key])
        try:
            return int(value)
        except (TypeError, ValueError):
            return int(DEFAULT_CONFIG[key])

    def _float_conf(self, key: str) -> float:
        value = self.config.get(key, DEFAULT_CONFIG[key])
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(DEFAULT_CONFIG[key])

    def _bool_conf(self, key: str) -> bool:
        value = self.config.get(key, DEFAULT_CONFIG[key])
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on", "是"}

    def _release_queue(self) -> None:
        """Release a queue slot."""
        self._queue_semaphore.release()

    async def terminate(self) -> None:
        pass


DEFAULT_CONFIG: dict[str, Any] = {
    "koharu_api_base_url": "http://127.0.0.1:7331/api/v1",
    "target_language": "Simplified Chinese",
    "pipeline_steps": "",
    "system_prompt": "",
    "default_font": "Noto Sans SC:500",
    "auto_load_llm": False,
    "llm_kind": "provider",
    "llm_provider_id": "openai-compatible",
    "llm_model_id": "",
    "llm_temperature": -1.0,
    "llm_max_tokens": 0,
    "llm_custom_system_prompt": "",
    "llm_load_timeout_seconds": 180,
    "wait_image_timeout_seconds": 120,
    "koharu_ready_timeout_seconds": 60,
    "pipeline_timeout_seconds": 900,
    "operation_poll_interval_seconds": 2,
    "http_timeout_seconds": 120,
    "http_connect_timeout_seconds": 10,
    "max_images_per_request": 20,
    "max_send_images": 0,
    "queue_depth": 3,
    "compress_return_images": False,
    "return_image_format": "webp",
    "return_image_quality": 85,
    "close_project_after_export": True,
    "delete_project_after_export": True,
    "result_retention_policy": "days",
    "result_retention_days": 7,
}


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _safe_path(path: str) -> str:
    try:
        return str(Path(path))
    except Exception:
        return str(path)


def _compress_image(source: Path, target: Path, image_format: str, quality: int) -> None:
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise RuntimeError("Pillow is required when compress_return_images is enabled") from exc

    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image)
        if image_format == "JPEG":
            image = _prepare_jpeg_image(image)
            image.save(
                target,
                "JPEG",
                quality=quality,
                optimize=True,
                progressive=True,
            )
            return

        if image.mode not in {"RGB", "RGBA"}:
            image = image.convert("RGBA" if _image_has_alpha(image) else "RGB")

        image.save(
            target,
            "WEBP",
            quality=quality,
            method=6,
        )


def _prepare_jpeg_image(image: Any) -> Any:
    from PIL import Image

    if not _image_has_alpha(image):
        return image.convert("RGB")

    rgba = image.convert("RGBA")
    background = Image.new("RGB", rgba.size, (255, 255, 255))
    background.paste(rgba, mask=rgba.getchannel("A"))
    return background


def _image_has_alpha(image: Any) -> bool:
    return image.mode in {"RGBA", "LA"} or (
        image.mode == "P" and "transparency" in image.info
    )


def _is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False
