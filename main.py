from __future__ import annotations

import asyncio
import copy
import shutil
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict, cast
from urllib.parse import urlparse

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, register
import astrbot.api.message_components as Comp
from astrbot.core.utils.session_waiter import SessionController, session_waiter

if TYPE_CHECKING:
    from PIL import Image as PILImage

try:
    from astrbot.core.utils.astrbot_path import get_astrbot_data_path
except Exception:  # pragma: no cover - compatibility fallback for older AstrBot.
    get_astrbot_data_path = None

try:
    from .koharu_client import (
        AppConfig,
        KoharuApiError,
        KoharuClient,
        LLMLoadOptions,
        LLMTarget,
        PatchBody,
        extract_project_id,
        save_exported_images,
    )
except ImportError:  # AstrBot may load plugin files without package context.
    from koharu_client import (
        AppConfig,
        KoharuApiError,
        KoharuClient,
        LLMLoadOptions,
        LLMTarget,
        PatchBody,
        extract_project_id,
        save_exported_images,
    )

try:
    from .onebot_client import ForwardNodeContent, QuotedMessageReadError, QuotedMessageReader
except ImportError:  # AstrBot may load plugin files without package context.
    from onebot_client import ForwardNodeContent, QuotedMessageReadError, QuotedMessageReader


PLUGIN_NAME = "astrbot_plugin_koharu"


@dataclass
class ForwardNode:
    """转发记录中一个含图节点，image_indices 指向 QuotedBatch.image_paths 的下标。"""

    uin: str
    name: str
    image_indices: list[int]


@dataclass
class QuotedBatch:
    """一次翻译请求的提取结果。forward_nodes 为 None 表示非转发；[] 表示转发但无图节点。"""

    image_paths: list[str]
    forward_nodes: list[ForwardNode] | None = None


@register(
    PLUGIN_NAME,
    "ABCwewe+CodeX",
    "使用 Koharu HTTP API 翻译聊天中的漫画图片。",
    "1.6.2",
)
class KoharuMangaTranslatorPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig | None = None):
        super().__init__(context)
        self.config: PluginConfig = cast(PluginConfig, config or {})
        self._translate_lock = asyncio.Lock()
        self._config_lock = asyncio.Lock()
        self._data_dir = self._resolve_data_dir()
        self._queue_semaphore = asyncio.Semaphore(self._int_conf("queue_depth") + 1)
        self._startup_task: asyncio.Task[None] | None = None

    async def initialize(self) -> None:
        self._data_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(
            "[koharu-plugin] initialized data_dir=%s api_base=%s target_language=%s",
            self._data_dir,
            self._str_conf("koharu_api_base_url"),
            self._str_conf("target_language"),
        )
        self._cleanup_output_cache()
        # 后台应用一次持久化配置；task 引用保存在实例上便于 terminate 取消。
        self._startup_task = asyncio.create_task(self._apply_config_on_startup())

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
        batch = await self._try_extract_image_batch(event)
        if batch is None:
            return
        has_quote = _contains_quote(event.get_messages())
        logger.debug(
            "[koharu-plugin] command image extraction done count=%d forward=%s",
            len(batch.image_paths),
            batch.forward_nodes is not None,
        )

        if batch.image_paths:
            await self._run_translation(event, batch, target_language)
            return

        if has_quote:
            await event.send(
                event.plain_result(
                    "未能从被引用消息中提取到图片，请引用带图片的消息或直接发送图片。"
                )
            )
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

            next_batch = await self._try_extract_image_batch(next_event)
            if next_batch is None:
                controller.stop()
                return
            logger.debug(
                "[koharu-plugin] waiter image extraction done count=%d",
                len(next_batch.image_paths),
            )
            if not next_batch.image_paths:
                logger.debug("[koharu-plugin] waiter got no images; keep waiting")
                await next_event.send(
                    next_event.plain_result("未检测到图片，请重新发送图片或发送“取消”。")
                )
                controller.keep(
                    timeout=self._int_conf("wait_image_timeout_seconds"),
                    reset_timeout=True,
                )
                return

            await self._run_translation(next_event, next_batch, target_language)
            controller.stop()

        try:
            logger.debug("[koharu-plugin] registering session waiter for image input")
            await wait_for_images(event)
        except TimeoutError:
            logger.info("[koharu-plugin] waiter timeout")
            await event.send(event.plain_result("等待图片超时，已退出漫画翻译。"))

    @filter.command("koharu-config")
    async def koharu_config(self, event: AstrMessageEvent):
        """手动重放 Koharu 持久化配置与密钥（管线模型/提供商/字体/语言/密钥）。"""

        event.stop_event()
        logger.info(
            "[koharu-plugin] koharu-config command triggered sender=%s session=%s",
            event.get_sender_id(),
            event.get_session_id(),
        )
        await event.send(event.plain_result("正在应用 Koharu 持久化配置与密钥..."))
        try:
            async with KoharuClient(
                self._str_conf("koharu_api_base_url"),
                timeout=float(self._int_conf("http_timeout_seconds")),
                connect_timeout=float(self._int_conf("http_connect_timeout_seconds")),
            ) as client:
                await client.wait_until_ready(
                    timeout_seconds=float(self._int_conf("koharu_ready_timeout_seconds"))
                )
                result = await self._apply_config_once(client)
        except Exception as exc:
            logger.exception("koharu-config apply failed")
            await event.send(event.plain_result(f"Koharu 配置应用失败：{exc}"))
            return
        patched = (
            "、".join(result.patched_sections)
            if result.patched_sections
            else "无（与现有配置一致）"
        )
        secrets = "、".join(result.replayed_secrets) if result.replayed_secrets else "无"
        await event.send(
            event.plain_result(
                f"Koharu 配置已应用。PATCH section：{patched}；密钥重放：{secrets}"
            )
        )

    async def _run_translation(
        self,
        event: AstrMessageEvent,
        batch: QuotedBatch,
        target_language: str,
    ) -> None:
        """按队列语义执行一次翻译并发送结果（转发场景输出合并转发记录）。"""
        image_count = len(batch.image_paths)
        if self._queue_semaphore.locked():
            logger.info("[koharu-plugin] queue full; rejecting translation request")
            await event.send(
                event.plain_result(
                    f"翻译队列已满（最大等待 {self._int_conf('queue_depth')} 个），请稍后再试。"
                )
            )
            return
        await self._queue_semaphore.acquire()
        try:
            logger.info("[koharu-plugin] sending accepted message before translation")
            forward_prefix = "转发记录中的 " if batch.forward_nodes is not None else " "
            confirm_text = (
                f"已收到{forward_prefix}{image_count} 张图片，"
                f"开始调用 Koharu 翻译为 {display_language(target_language)}。"
            )
            await event.send(event.plain_result(confirm_text))
            logger.info("[koharu-plugin] accepted message sent; starting translation")
            try:
                output_paths = await self._translate_images(batch.image_paths, target_language)
            except Exception as exc:
                logger.exception("Koharu manga translation failed")
                await event.send(event.plain_result(f"漫画翻译失败：{exc}"))
                return
            logger.info(
                "[koharu-plugin] translation finished; sending output count=%d",
                len(output_paths),
            )
            if batch.forward_nodes is not None:
                await self._send_forward_result(event, batch, output_paths)
            else:
                await self._send_one_by_one(event, output_paths)
            self._cleanup_current_outputs_if_needed(output_paths)
            self._cleanup_output_cache()
        finally:
            self._release_queue()

    async def _send_forward_result(
        self,
        event: AstrMessageEvent,
        batch: QuotedBatch,
        output_paths: list[str],
    ) -> None:
        """按原聊天记录格式（合并转发）发送译文图，只保留含图节点，无任何提示文字。"""
        max_send = self._int_conf("max_send_images")
        budget = max_send if max_send > 0 else len(output_paths)
        nodes: list[Comp.Node] = []
        for node in batch.forward_nodes or []:
            if budget <= 0:
                break
            images: list[Comp.BaseMessageComponent] = []
            for index in node.image_indices:
                if budget <= 0:
                    break
                if index >= len(output_paths):
                    logger.warning(
                        "[koharu-plugin] forward node image index out of range "
                        "index=%d output_count=%d",
                        index,
                        len(output_paths),
                    )
                    continue
                images.append(_image_from_path(output_paths[index]))
                budget -= 1
            if images:
                nodes.append(Comp.Node(uin=node.uin, name=node.name, content=images))
        if not nodes:
            logger.warning(
                "[koharu-plugin] no image nodes to send in forward result output_count=%d",
                len(output_paths),
            )
            return
        await event.send(event.chain_result([Comp.Nodes(nodes)]).stop_event())

    async def _send_one_by_one(self, event: AstrMessageEvent, output_paths: list[str]) -> None:
        """非转发场景：翻译结果逐张单独发送，无提示文字。"""
        max_send = self._int_conf("max_send_images")
        selected = output_paths if max_send <= 0 else output_paths[:max_send]
        for path in selected:
            await event.send(event.image_result(path))

    async def _try_extract_image_batch(self, event: AstrMessageEvent) -> QuotedBatch | None:
        """提取图片批次;读取被引用消息失败时向用户播报错误并返回 None。"""
        try:
            return await self._extract_image_batch(event)
        except QuotedMessageReadError as exc:
            logger.info(
                "[koharu-plugin] failed to read quoted message content error=%s",
                exc,
            )
            await event.send(event.plain_result(str(exc)))
            return None

    async def _extract_image_batch(self, event: AstrMessageEvent) -> QuotedBatch:
        """提取当前消息中的图片；引用消息或合并转发按引用场景处理。"""
        messages = event.get_messages()
        logger.debug(
            "[koharu-plugin] extracting images from message_chain component_count=%d component_types=%s",
            len(messages),
            [type(component).__name__ for component in messages],
        )
        reader = QuotedMessageReader(self.context)
        raw_paths: list[str] = []
        pending_nodes: list[tuple[str, str, int, int]] = []
        is_forward = False

        for component in messages:
            if isinstance(component, Comp.Image):
                path = await self._try_convert_image_path(component)
                if path is not None:
                    raw_paths.append(path)
                continue
            if isinstance(component, Comp.Reply):
                chain = component.chain
                if chain:
                    for nested in chain:
                        if isinstance(nested, Comp.Image):
                            path = await self._try_convert_image_path(nested)
                            if path is not None:
                                raw_paths.append(path)
                        elif isinstance(nested, Comp.Forward):
                            is_forward = True
                            await self._collect_forward_node_images(
                                reader,
                                event,
                                nested.id,
                                raw_paths,
                                pending_nodes,
                            )
                    continue
                if component.id:
                    await self._collect_quoted_fallback_images(
                        reader,
                        event,
                        component.id,
                        raw_paths,
                    )
                continue
            if isinstance(component, Comp.Forward):
                is_forward = True
                await self._collect_forward_node_images(
                    reader,
                    event,
                    component.id,
                    raw_paths,
                    pending_nodes,
                )

        unique_paths, index_map = _dedupe_mapped(raw_paths)
        forward_nodes = [
            ForwardNode(
                uin=uin,
                name=name,
                image_indices=[
                    index_map[index] for index in range(start, start + count)
                ],
            )
            for uin, name, start, count in pending_nodes
        ]
        logger.debug(
            "[koharu-plugin] extracted image paths count=%d paths=%s",
            len(unique_paths),
            [_safe_path(path) for path in unique_paths],
        )
        return QuotedBatch(
            image_paths=unique_paths,
            forward_nodes=forward_nodes if is_forward else None,
        )

    async def _collect_forward_node_images(
        self,
        reader: QuotedMessageReader,
        event: AstrMessageEvent,
        forward_id: str,
        raw_paths: list[str],
        pending_nodes: list[tuple[str, str, int, int]],
    ) -> None:
        """读取合并转发记录，收集各含图节点的图片路径（单图失败跳过）。

        整体读取失败时抛 QuotedMessageReadError，由调用方提示用户。
        """
        contents: list[ForwardNodeContent] = await reader.fetch_forward(
            event,
            forward_id,
        )
        for content in contents:
            node_paths: list[str] = []
            for component in content.components:
                if not isinstance(component, Comp.Image):
                    continue
                path = await self._try_convert_image_path(component)
                if path is not None:
                    node_paths.append(path)
            if not node_paths:
                continue
            start = len(raw_paths)
            raw_paths.extend(node_paths)
            pending_nodes.append((content.uin, content.name, start, len(node_paths)))

    async def _collect_quoted_fallback_images(
        self,
        reader: QuotedMessageReader,
        event: AstrMessageEvent,
        message_id: str | int,
        raw_paths: list[str],
    ) -> None:
        """Reply.chain 为空时按被引用消息 ID 拉取消息内容兜底（结果不算转发）。

        整体读取失败时抛 QuotedMessageReadError，由调用方提示用户。
        """
        quoted_components = await reader.fetch_quoted_message(
            event,
            str(message_id),
        )
        for component in quoted_components:
            if isinstance(component, Comp.Image):
                path = await self._try_convert_image_path(component)
                if path is not None:
                    raw_paths.append(path)

    async def _convert_image_path(self, component: Comp.Image) -> str:
        logger.debug(
            "[koharu-plugin] converting image component file=%r url=%r path=%r",
            component.file,
            component.url,
            component.path,
        )
        path = await component.convert_to_file_path()
        logger.debug("[koharu-plugin] image component converted path=%s", _safe_path(path))
        return path

    async def _try_convert_image_path(self, component: Comp.Image) -> str | None:
        """转换图片为本地路径;单张失败时记录日志并返回 None(不拖垮整批)。"""
        try:
            return await self._convert_image_path(component)
        except Exception as exc:
            logger.warning(
                "[koharu-plugin] failed to convert image component error=%s",
                exc,
            )
            return None

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
                await self._ensure_config_applied(client)
                logger.debug("[koharu-plugin] closing existing koharu project before creating a new one")
                closed_existing = await client.close_project_if_any()
                logger.debug(
                    "[koharu-plugin] close existing project result closed=%s",
                    closed_existing,
                )
                logger.debug("[koharu-plugin] koharu ready; creating project")
                project = await client.create_project(project_name)
                logger.debug("[koharu-plugin] project created response=%s", project)
                project_name_from_response = extract_project_id(project)
                logger.debug(
                    "[koharu-plugin] project identity=%s",
                    project_name_from_response,
                )
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
                        pages = await client.create_pages(cached_image_paths)
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
                    operation_id = await client.start_pipeline(steps)
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
                    if (
                        self._bool_conf("delete_project_after_export")
                        and project_name_from_response
                    ):
                        try:
                            logger.debug(
                                "[koharu-plugin] deleting koharu project name=%s",
                                project_name_from_response,
                            )
                            await client.delete_project(str(project_name_from_response))
                            logger.debug("[koharu-plugin] koharu project deleted")
                        except Exception as exc:
                            logger.warning(f"Failed to delete Koharu project: {exc}")

    async def _apply_config_on_startup(self) -> None:
        """插件启动后后台应用一次持久化配置（等 Koharu 就绪；失败重试，不阻塞启动）。"""
        for attempt in range(1, 4):
            try:
                async with KoharuClient(
                    self._str_conf("koharu_api_base_url"),
                    timeout=float(self._int_conf("http_timeout_seconds")),
                    connect_timeout=float(self._int_conf("http_connect_timeout_seconds")),
                ) as client:
                    await client.wait_until_ready(
                        timeout_seconds=float(self._int_conf("koharu_ready_timeout_seconds"))
                    )
                    result = await self._apply_config_once(client)
                    logger.info(
                        "[koharu-plugin] persistent config applied on startup "
                        "patched=%s secrets=%s",
                        result.patched_sections,
                        result.replayed_secrets,
                    )
                return
            except Exception as exc:
                logger.warning(
                    "[koharu-plugin] startup config apply attempt %d failed: %s",
                    attempt,
                    exc,
                )
                await asyncio.sleep(5 * attempt)

    async def _ensure_config_applied(self, client: KoharuClient) -> None:
        """翻译前确保持久化配置已应用；失败仅告警，不阻断翻译。"""
        try:
            result = await self._apply_config_once(client)
            if result.patched_sections or result.replayed_secrets:
                logger.debug(
                    "[koharu-plugin] applied persistent config patched=%s secrets=%s",
                    result.patched_sections,
                    result.replayed_secrets,
                )
        except Exception as exc:
            logger.warning("[koharu-plugin] failed to apply persistent config: %s", exc)

    async def _apply_config_once(self, client: KoharuClient) -> ConfigApplyResult:
        """GET /config 全量 → 组装期望值 → 差异 section 整段 PATCH → 重放密钥。

        用 _config_lock 与启动任务/手动指令串行化，避免与翻译流程并发 PATCH
        造成「翻译前一刻服务端模型被改回插件默认」的竞态。
        """
        async with self._config_lock:
            current = await client.get_config()
            expected = build_expected_config(current, self.config)
            patched: list[str] = []
            for section in ("pipeline", "providers", "typesetting"):
                if section in config_differs(current, expected):
                    section_value = cast(dict[str, object], expected).get(section)
                    patch = cast(PatchBody, {section: section_value})
                    await client.patch_config(patch)
                    patched.append(section)
            replayed = await self._replay_provider_secrets(client)
            return ConfigApplyResult(patched_sections=patched, replayed_secrets=replayed)

    async def _replay_provider_secrets(self, client: KoharuClient) -> list[str]:
        """把插件配置里非空的 provider 密钥写入服务端 keyring（幂等，重启即丢需重放）。"""
        replayed: list[str] = []
        for provider_id, secret in self._provider_secrets().items():
            if secret.strip():
                await client.set_provider_secret(provider_id, secret.strip())
                replayed.append(provider_id)
        return replayed

    def _provider_secrets(self) -> dict[str, str]:
        # 旧配置可能缺 provider_secrets 键或嵌套值为 None，统一兜底。
        raw = self.config.get("provider_secrets") or {}
        result: dict[str, str] = {}
        for key, value in raw.items():
            stripped = str(value or "").strip()
            if stripped:
                result[str(key)] = stripped
        # 独立的 openai-compatible key 字段优先于 provider_secrets 里的同名项。
        direct_key = str(self.config.get("openai_compatible_api_key") or "").strip()
        if direct_key:
            result["openai-compatible"] = direct_key
        return result

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
        target: LLMTarget
        if llm_kind == "provider":
            provider_id = self._str_conf("llm_provider_id").strip()
            if not provider_id:
                raise ValueError("auto_load_llm 已启用，但 llm_provider_id 为空。")
            target = {
                "kind": "provider",
                "providerId": provider_id,
                "modelId": model_id,
                # 与持久化配置共用 vision：多模态模型置 true，否则 PUT 会覆盖
                # PATCH 写入的 vision 为硬编码 false。
                "vision": self._bool_conf("translation_vision"),
            }
        elif llm_kind == "local":
            target = {"kind": "local", "modelId": model_id}
            quantization = self._str_conf("translation_quantization").strip()
            if quantization:
                target["quantization"] = quantization
        else:
            raise ValueError("llm_kind 只能是 local 或 provider。")

        options: LLMLoadOptions = {}
        custom_prompt = self._str_conf("llm_custom_system_prompt").strip()
        if custom_prompt:
            # 0.66 的 PUT /llm/current 只接受 instructions;temperature/maxTokens
            # 由服务端 pipeline.translation.generation 配置管理。
            options["customSystemPrompt"] = custom_prompt

        logger.debug("[koharu-plugin] sending llm load request target=%s options=%s", target, options)
        # 0.66: PUT /llm/current 返回 204 即完成(模型翻译时懒加载),无 status 可轮询。
        await client.load_llm(target, options=options or None)
        logger.debug("[koharu-plugin] llm selected")

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

    def _raw_config_value(self, key: str) -> str | int | float | bool:
        """取配置值；缺失时回退到默认值。默认值缺失视为配置键错误。"""
        value = self.config.get(key)
        if value is not None:
            return value
        default = DEFAULT_CONFIG.get(key)
        if default is None:
            raise KeyError(f"unknown plugin config key: {key}")
        return default

    def _str_conf(self, key: str) -> str:
        return str(self._raw_config_value(key))

    def _int_conf(self, key: str) -> int:
        value = self._raw_config_value(key)
        try:
            return int(value)
        except (TypeError, ValueError):
            default = DEFAULT_CONFIG.get(key)
            if default is None:
                raise
            return int(default)

    def _float_conf(self, key: str) -> float:
        value = self._raw_config_value(key)
        try:
            return float(value)
        except (TypeError, ValueError):
            default = DEFAULT_CONFIG.get(key)
            if default is None:
                raise
            return float(default)

    def _bool_conf(self, key: str) -> bool:
        value = self._raw_config_value(key)
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on", "是"}

    def _release_queue(self) -> None:
        """Release a queue slot."""
        self._queue_semaphore.release()

    async def terminate(self) -> None:
        task = self._startup_task
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


class PluginConfig(TypedDict):
    """插件配置键值类型，与 _conf_schema.json 保持一致。"""

    koharu_api_base_url: str
    target_language: str
    pipeline_steps: str
    system_prompt: str
    auto_load_llm: bool
    llm_kind: str
    llm_provider_id: str
    llm_model_id: str
    llm_temperature: float
    llm_max_tokens: int
    llm_custom_system_prompt: str
    wait_image_timeout_seconds: int
    koharu_ready_timeout_seconds: int
    pipeline_timeout_seconds: int
    operation_poll_interval_seconds: int
    http_timeout_seconds: int
    http_connect_timeout_seconds: int
    max_images_per_request: int
    max_send_images: int
    queue_depth: int
    compress_return_images: bool
    return_image_format: str
    return_image_quality: int
    close_project_after_export: bool
    delete_project_after_export: bool
    result_retention_policy: str
    result_retention_days: int
    # --- Koharu 0.66 持久化配置（PATCH /config） ---
    pipeline_ocr_model: str
    pipeline_inpainting_model: str
    pipeline_inpainting_prompt: str
    pipeline_inpainting_negative_prompt: str
    pipeline_detection_text_threshold: float
    pipeline_detection_bubble_threshold: float
    pipeline_detection_panel_threshold: float
    translation_provider: str
    translation_model: str
    translation_quantization: str
    translation_vision: bool
    openai_compatible_base_url: str
    openai_compatible_api_key: str
    openai_compatible_vision: bool
    lm_studio_base_url: str
    deepl_base_url: str
    font_families: str
    provider_secrets: dict[str, str]


DEFAULT_CONFIG: PluginConfig = {
    "koharu_api_base_url": "http://koharu-headless:4000/api/v1",
    "target_language": "zh-CN",
    "pipeline_steps": "full",
    "system_prompt": "",
    "auto_load_llm": False,
    "llm_kind": "provider",
    "llm_provider_id": "openai-compatible",
    "llm_model_id": "",
    "llm_temperature": -1.0,
    "llm_max_tokens": 0,
    "llm_custom_system_prompt": "",
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
    # --- Koharu 0.66 持久化配置默认值（留空=不覆盖服务端对应字段） ---
    "pipeline_ocr_model": "",
    "pipeline_inpainting_model": "",
    "pipeline_inpainting_prompt": "",
    "pipeline_inpainting_negative_prompt": "",
    "pipeline_detection_text_threshold": -1.0,
    "pipeline_detection_bubble_threshold": -1.0,
    "pipeline_detection_panel_threshold": -1.0,
    "translation_provider": "deepseek",
    "translation_model": "deepseek-v4-flash",
    "translation_quantization": "",
    "translation_vision": False,
    "openai_compatible_base_url": "",
    "openai_compatible_api_key": "",
    "openai_compatible_vision": False,
    "lm_studio_base_url": "",
    "deepl_base_url": "",
    "font_families": "CCWildWords,Adobe 黑体 Std",
    "provider_secrets": {},
}


def _dedupe_mapped(paths: list[str]) -> tuple[list[str], dict[int, int]]:
    """去重保序，返回（唯一列表, 原位置 → 唯一下标映射）。

    重复路径映射到首次出现的唯一下标（同一张图只翻译一次），
    保证 index_map 对每个原位置都有值。
    """
    first_index: dict[str, int] = {}
    unique: list[str] = []
    index_map: dict[int, int] = {}
    for index, path in enumerate(paths):
        if path not in first_index:
            first_index[path] = len(unique)
            unique.append(path)
        index_map[index] = first_index[path]
    return unique, index_map


def _contains_quote(messages: list[Comp.BaseMessageComponent]) -> bool:
    """消息链顶层是否含引用(Reply)或合并转发(Forward)组件。"""
    return any(isinstance(component, (Comp.Reply, Comp.Forward)) for component in messages)


def _safe_path(path: str) -> str:
    try:
        return str(Path(path))
    except Exception:
        return str(path)


def _image_from_path(path: str) -> Comp.Image:
    """将本地图片路径构造为 Comp.Image（SDK 未类型化成员的边界 helper，唯一 cast 点之一）。"""
    from_file_system = getattr(Comp.Image, "fromFileSystem")
    return cast(Comp.Image, from_file_system(path))


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


def _prepare_jpeg_image(image: PILImage.Image) -> PILImage.Image:
    from PIL import Image

    if not _image_has_alpha(image):
        return image.convert("RGB")

    rgba = image.convert("RGBA")
    background = Image.new("RGB", rgba.size, (255, 255, 255))
    background.paste(rgba, mask=rgba.getchannel("A"))
    return background


def _image_has_alpha(image: PILImage.Image) -> bool:
    return image.mode in {"RGBA", "LA"} or (
        image.mode == "P" and "transparency" in image.info
    )


def _is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False


# --- Koharu 0.66 持久化配置（语言目录 / config 组装）-----------------------------

# 服务端支持的 provider id（koharu_translator::Provider，wire 名）。
_PROVIDER_IDS: tuple[str, ...] = (
    "local",
    "atlas-cloud",
    "openai",
    "gemini",
    "claude",
    "deepseek",
    "openai-compatible",
    "openrouter",
    "lm-studio",
    "deepl",
    "google-cloud-translation",
    "caiyun",
)

# 0.66 服务端语言目录（koharu_translator::Language，canonical tag → 显示名）。
# 展示用英文名（与语言.rs 的 to_string 一致）。
_LANGUAGE_NAME_BY_TAG: dict[str, str] = {
    "zh-CN": "Simplified Chinese",
    "en-US": "English",
    "fr-FR": "French",
    "pt-PT": "Portuguese",
    "pt-BR": "Brazilian Portuguese",
    "es-ES": "Spanish",
    "ja-JP": "Japanese",
    "tr-TR": "Turkish",
    "ru-RU": "Russian",
    "ar-SA": "Arabic",
    "ko-KR": "Korean",
    "th-TH": "Thai",
    "it-IT": "Italian",
    "de-DE": "German",
    "vi-VN": "Vietnamese",
    "ms-MY": "Malay",
    "id-ID": "Indonesian",
    "fil-PH": "Filipino",
    "hi-IN": "Hindi",
    "zh-TW": "Traditional Chinese",
    "pl-PL": "Polish",
    "cs-CZ": "Czech",
    "nl-NL": "Dutch",
    "km-KH": "Khmer",
    "my-MM": "Burmese",
    "fa-IR": "Persian",
    "gu-IN": "Gujarati",
    "ur-PK": "Urdu",
    "te-IN": "Telugu",
    "mr-IN": "Marathi",
    "he-IL": "Hebrew",
    "bn-BD": "Bengali",
    "bg-BG": "Bulgarian",
    "ta-IN": "Tamil",
    "uk-UA": "Ukrainian",
    "bo-CN": "Tibetan",
    "kk-KZ": "Kazakh",
    "mn-MN": "Mongolian",
    "ug-CN": "Uyghur",
    "yue-HK": "Cantonese",
    "be-BY": "Belarusian",
    "hu-HU": "Hungarian",
}

# canonical tag / 别名（serialize 接受串）/ 显示名 → canonical tag（全部 lowercase）。
_LANGUAGE_TAG_BY_ALIAS: dict[str, str] = {
    **{tag.lower(): tag for tag in _LANGUAGE_NAME_BY_TAG},
    **{name.lower(): tag for tag, name in _LANGUAGE_NAME_BY_TAG.items()},
    # 别名（language.rs 的 serialize 额外串）
    "zh": "zh-CN",
    "zh-hans": "zh-CN",
    "en": "en-US",
    "fr": "fr-FR",
    "pt": "pt-PT",
    "es": "es-ES",
    "ja": "ja-JP",
    "tr": "tr-TR",
    "ru": "ru-RU",
    "ar": "ar-SA",
    "ko": "ko-KR",
    "th": "th-TH",
    "it": "it-IT",
    "de": "de-DE",
    "vi": "vi-VN",
    "ms": "ms-MY",
    "id": "id-ID",
    "fil": "fil-PH",
    "tl": "fil-PH",
    "hi": "hi-IN",
    "zh-hant": "zh-TW",
    "pl": "pl-PL",
    "cs": "cs-CZ",
    "nl": "nl-NL",
    "km": "km-KH",
    "my": "my-MM",
    "fa": "fa-IR",
    "gu": "gu-IN",
    "ur": "ur-PK",
    "te": "te-IN",
    "mr": "mr-IN",
    "he": "he-IL",
    "bn": "bn-BD",
    "bg": "bg-BG",
    "ta": "ta-IN",
    "uk": "uk-UA",
    "bo": "bo-CN",
    "kk": "kk-KZ",
    "mn": "mn-MN",
    "ug": "ug-CN",
    "yue": "yue-HK",
    "be": "be-BY",
    "hu": "hu-HU",
}

# 展示文案特例：中文界面直接用中文名。
_LANGUAGE_DISPLAY_ZH: dict[str, str] = {
    "zh-CN": "简体中文",
    "zh-TW": "繁体中文",
}


def normalize_language(raw: str) -> str | None:
    """把配置值规范化为 0.66 服务端接受的 BCP47 tag。

    接受 canonical tag / 别名（zh、zh-Hans）/ 显示名（"Simplified Chinese"），
    大小写不敏感；无法识别返回 None（调用方跳过覆盖，避免 PATCH 422）。
    """
    lowered = raw.strip().lower()
    return _LANGUAGE_TAG_BY_ALIAS.get(lowered)


def display_language(raw: str) -> str:
    """把语言配置值转换为用户可读的展示文案（zh-CN → 简体中文）。"""
    tag = normalize_language(raw)
    if tag is None:
        return raw.strip()
    return _LANGUAGE_DISPLAY_ZH.get(tag) or _LANGUAGE_NAME_BY_TAG[tag]


@dataclass
class ConfigApplyResult:
    """一次持久化配置应用的结果。"""

    patched_sections: list[str]
    replayed_secrets: list[str]


def _cfg_str(cfg: Mapping[str, object], key: str, default: str = "") -> str:
    """安全读取字符串配置：None/非 str（旧配置缺键或值为 None）回落默认。"""
    value = cfg.get(key)
    return value if isinstance(value, str) else default


def _cfg_float(cfg: Mapping[str, object], key: str, default: float) -> float:
    """安全读取 float 配置：兼容数字与数字字符串，None/错型回落默认。"""
    value = cfg.get(key)
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return default
    return default


def _cfg_int(cfg: Mapping[str, object], key: str, default: int) -> int:
    """安全读取 int 配置：兼容数字与数字字符串，None/错型回落默认。"""
    value = cfg.get(key)
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return default
    return default


def _cfg_bool(cfg: Mapping[str, object], key: str, default: bool) -> bool:
    """安全读取 bool 配置：与 _bool_conf 语义一致（"false"/"0" 不是 True）。"""
    value = cfg.get(key)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "是"}
    return default


def _is_valid_url(raw: str) -> bool:
    """宽松 URL 校验：必须带 scheme 与 netloc（服务端 base_url 是 Url 类型，非法会 422）。"""
    try:
        parsed = urlparse(raw)
        return bool(parsed.scheme and parsed.netloc)
    except ValueError:
        return False


# u32::MAX（服务端 max_tokens 是 Option<u32>，超限 422）。
_U32_MAX = 4294967295


def build_expected_config(current: AppConfig, cfg: Mapping[str, object]) -> AppConfig:
    """从服务端现有 config 拷贝，覆盖插件已配置的字段。

    语义：配置项**留空/默认未配置值 = 不覆盖**服务端对应字段；**非空默认值
    （如 target_language=zh-CN、translation_provider=deepseek）即声明值**，
    会在启动/翻译前强制对齐服务端。0.66 PATCH /config 是「顶层稀疏、
    section 整段替换」：期望值必须以现有 config 为底，只改插件声明的字段。
    """
    expected = copy.deepcopy(current)
    pipeline = expected.get("pipeline") or {}
    expected["pipeline"] = pipeline

    # --- detection/ocr/inpainting 模型选择 ---
    ocr_model = _cfg_str(cfg, "pipeline_ocr_model").strip()
    if ocr_model:
        ocr = pipeline.get("ocr") or {}
        ocr["model"] = ocr_model
        pipeline["ocr"] = ocr
    inpainting_model = _cfg_str(cfg, "pipeline_inpainting_model").strip()
    if inpainting_model:
        inpainting = pipeline.get("inpainting") or {}
        inpainting["model"] = inpainting_model
        pipeline["inpainting"] = inpainting

    # --- processor（保留现有键，只覆盖配置的） ---
    processor = pipeline.get("processor") or {}
    pipeline["processor"] = processor
    text_threshold = _threshold_or_none(_cfg_float(cfg, "pipeline_detection_text_threshold", -1.0))
    bubble_threshold = _threshold_or_none(
        _cfg_float(cfg, "pipeline_detection_bubble_threshold", -1.0)
    )
    panel_threshold = _threshold_or_none(_cfg_float(cfg, "pipeline_detection_panel_threshold", -1.0))
    if any(v is not None for v in (text_threshold, bubble_threshold, panel_threshold)):
        detection_processor = processor.get("koharu-layout-rfdetr-seg-2xl") or {}
        if text_threshold is not None:
            detection_processor["text_threshold"] = text_threshold
        if bubble_threshold is not None:
            detection_processor["bubble_threshold"] = bubble_threshold
        if panel_threshold is not None:
            detection_processor["panel_threshold"] = panel_threshold
        processor["koharu-layout-rfdetr-seg-2xl"] = detection_processor
    inpainting_prompt = _cfg_str(cfg, "pipeline_inpainting_prompt").strip()
    negative_prompt = _cfg_str(cfg, "pipeline_inpainting_negative_prompt").strip()
    if inpainting_prompt:
        for key in ("flux2-klein", "rorem-mixed"):
            model_processor = processor.get(key) or {}
            model_processor["prompt"] = inpainting_prompt
            processor[key] = model_processor
    if negative_prompt:
        rorem_processor = processor.get("rorem-mixed") or {}
        rorem_processor["negative_prompt"] = negative_prompt
        processor["rorem-mixed"] = rorem_processor

    # --- translation ---
    translation = pipeline.get("translation") or {}
    pipeline["translation"] = translation
    provider = _cfg_str(cfg, "translation_provider").strip()
    model_name = _cfg_str(cfg, "translation_model").strip()
    # 自定义端点三件套：填了 openai-compatible 的 base_url 或 api_key 即强制
    # 切换翻译到该提供商（无需手动改 translation_provider）。
    compatible_base_url = _cfg_str(cfg, "openai_compatible_base_url").strip()
    compatible_api_key = _cfg_str(cfg, "openai_compatible_api_key").strip()
    if compatible_base_url or compatible_api_key:
        provider = "openai-compatible"
    # ModelSelection 的 provider/vision 必填：provider 不在白名单或 model 为空时
    # 不重建（避免 provider:"" 触发服务端 422）。
    if provider in _PROVIDER_IDS and model_name:
        translation["model"] = {
            "provider": provider,
            "model": model_name,
            "quantization": _cfg_str(cfg, "translation_quantization").strip() or None,
            "vision": _cfg_bool(cfg, "translation_vision", False),
        }
    target_language = normalize_language(_cfg_str(cfg, "target_language"))
    if target_language is not None:
        translation["target_language"] = target_language
    instructions = _cfg_str(cfg, "system_prompt").strip()
    if instructions:
        translation["instructions"] = instructions
    generation = translation.get("generation") or {}
    temperature = _float_or_none(_cfg_float(cfg, "llm_temperature", -1.0))
    if temperature is not None:
        # 服务端 f32 反序列化对超大值 422，钳制到合理范围。
        generation["temperature"] = min(2.0, max(0.0, temperature))
    max_tokens = _int_or_none(_cfg_int(cfg, "llm_max_tokens", 0))
    if max_tokens is not None:
        generation["max_tokens"] = min(_U32_MAX, max_tokens)
    if generation:
        translation["generation"] = generation

    # --- providers（保留现有 settings，只覆盖配置的） ---
    providers = expected.get("providers") or {}
    expected["providers"] = providers
    if compatible_base_url and _is_valid_url(compatible_base_url):
        settings = providers.get("openai-compatible") or {}
        settings["base_url"] = compatible_base_url
        providers["openai-compatible"] = settings
    # bool 配置无法留空：始终覆盖（默认 false，与服务端默认一致）。
    settings = providers.get("openai-compatible") or {}
    settings["vision"] = _cfg_bool(cfg, "openai_compatible_vision", False)
    providers["openai-compatible"] = settings
    lm_studio_base_url = _cfg_str(cfg, "lm_studio_base_url").strip()
    if lm_studio_base_url and _is_valid_url(lm_studio_base_url):
        settings = providers.get("lm-studio") or {}
        settings["base_url"] = lm_studio_base_url
        providers["lm-studio"] = settings
    deepl_base_url = _cfg_str(cfg, "deepl_base_url").strip()
    if deepl_base_url and _is_valid_url(deepl_base_url):
        settings = providers.get("deepl") or {}
        settings["base_url"] = deepl_base_url
        providers["deepl"] = settings

    # --- typesetting（字体，合并保留现有键） ---
    font_families = _cfg_str(cfg, "font_families").strip()
    if font_families:
        families = [item.strip() for item in font_families.split(",") if item.strip()]
        if families:
            typesetting = expected.get("typesetting") or {}
            typesetting["font_families"] = families
            expected["typesetting"] = typesetting

    return expected


def _float_or_none(value: object) -> float | None:
    """float 配置值：>0 才视为已配置（-1/0/清空均表示不覆盖，见 _float_or_none 语义）。"""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if number > 0 else None
    return None


def _threshold_or_none(value: float) -> float | None:
    """检测阈值：服务端校验 0.0..=1.0（运行时才报错），这里直接钳制到合法范围。"""
    if value > 0:
        return min(1.0, max(0.0, value))
    return None


def _int_or_none(value: object) -> int | None:
    """int 配置值：>0 才视为已配置（0 语义与 llm_max_tokens 一致）。"""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value) if int(value) > 0 else None
    return None


def config_differs(current: AppConfig, expected: AppConfig) -> set[str]:
    """按 section 比较，返回有差异的 section 名（pipeline/providers/typesetting）。

    None 与 {} 视为等价：服务端缺失 section 且期望无内容时不触发空 PATCH
    （空 section PATCH 会把服务端该 section 重置为默认值）。
    """
    return {
        section
        for section in ("pipeline", "providers", "typesetting")
        if (current.get(section) or {}) != (expected.get(section) or {})
    }
