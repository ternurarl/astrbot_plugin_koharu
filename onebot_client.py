"""OneBot (aiocqhttp) 统一封装:读取被引用消息与合并转发消息。

按项目"统一封装"约定,所有 OneBot API 调用集中在本模块,main.py 不直接
接触 bot 对象与原始 dict。本模块只做"原始数据 → AstrBot 组件"的转换,
不负责图片下载/翻译——那是 main.py 的职责。

被引用消息与合并转发记录的内容只提取 image 段(用户已确认:只保留有图
节点、丢弃原文文本),其他段(text/face/at 等)一律忽略。
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, TypedDict, cast, runtime_checkable

import astrbot.api.message_components as Comp

try:
    from astrbot.api import logger
except Exception:  # pragma: no cover - 独立运行时的兜底。
    logger = logging.getLogger(__name__)

from astrbot.api.event import AstrMessageEvent
from astrbot.api.star import Context

__all__ = [
    "OneBotBot",
    "OneBotSegmentData",
    "OneBotSegment",
    "ForwardNodeData",
    "ForwardNodePayload",
    "ForwardMessageResponse",
    "QuotedMessageResponse",
    "ForwardNodeContent",
    "QuotedMessageReader",
    "QuotedMessageReadError",
]


class QuotedMessageReadError(RuntimeError):
    """读取被引用消息 / 合并转发消息失败(含平台不支持)。消息可直接展示给用户。"""



@runtime_checkable
class OneBotBot(Protocol):
    """aiocqhttp CQHttp 的最小类型面(避免依赖 aiocqhttp 包)。"""

    async def call_action(self, action: str, **params: object) -> object: ...


class OneBotSegmentData(TypedDict, total=False):
    text: str
    file: str
    url: str
    path: str
    file_unique: str


class OneBotSegment(TypedDict, total=False):
    type: str
    data: OneBotSegmentData


class ForwardNodeData(TypedDict, total=False):
    user_id: str
    nickname: str
    content: list[OneBotSegment]


class ForwardNodePayload(TypedDict, total=False):
    type: str
    data: ForwardNodeData


class ForwardMessageResponse(TypedDict, total=False):
    messages: list[ForwardNodePayload]


class QuotedMessageResponse(TypedDict, total=False):
    message: list[OneBotSegment]


@dataclass
class ForwardNodeContent:
    uin: str  # 节点发送者 QQ 号(字符串)
    name: str  # 节点发送者昵称
    components: list[Comp.BaseMessageComponent]  # 该节点内容转换后的 AstrBot 组件(只含 Comp.Image)


class QuotedMessageReader:
    """通过 OneBot API 读取被引用消息与合并转发消息。"""

    def __init__(self, context: Context) -> None:
        self.context = context

    async def fetch_forward(
        self,
        event: AstrMessageEvent,
        forward_id: str,
    ) -> list[ForwardNodeContent]:
        """读取合并转发记录,返回按原始顺序排列的、含图片的节点列表。

        无图节点会被整体跳过;单个节点解析失败只跳过该节点,不导致整体失败。
        """
        logger.debug("[onebot-client] get_forward_msg forward_id=%s", forward_id)
        result = await self._call_action(
            event,
            "get_forward_msg",
            "读取转发消息失败",
            message_id=forward_id,
        )
        # 边界单点:外部 JSON → 命名 TypedDict。
        response = cast(ForwardMessageResponse, result)
        messages = response.get("messages")
        if not isinstance(messages, list):
            logger.warning("[onebot-client] get_forward_msg 响应缺少 messages 字段")
            return []
        nodes: list[ForwardNodeContent] = []
        for index, node in enumerate(messages):
            content = _parse_forward_node(node)
            if content is not None:
                nodes.append(content)
            else:
                logger.warning(
                    "[onebot-client] 转发节点 %s 解析失败或无图片,已跳过", index
                )
        logger.debug(
            "[onebot-client] get_forward_msg forward_id=%s nodes=%s",
            forward_id,
            len(nodes),
        )
        return nodes

    async def fetch_quoted_message(
        self,
        event: AstrMessageEvent,
        message_id: str,
    ) -> list[Comp.BaseMessageComponent]:
        """读取被引用消息内容,返回其中全部 image 段转换后的 AstrBot 组件。

        引用过旧消息时 OneBot 适配器的 Reply.chain 可能为空,此方法通过
        get_msg 兜底读取原始消息。
        """
        # OneBot 的 message_id 在部分实现中是 int,优先转 int 传参;转失败就传原值。
        try:
            message_id_param: str | int = int(message_id)
        except ValueError:
            message_id_param = message_id
        logger.debug("[onebot-client] get_msg message_id=%s", message_id)
        result = await self._call_action(
            event,
            "get_msg",
            "读取被引用消息失败",
            message_id=message_id_param,
        )
        # 边界单点:外部 JSON → 命名 TypedDict。
        response = cast(QuotedMessageResponse, result)
        segments = response.get("message")
        if not isinstance(segments, list):
            raise QuotedMessageReadError("读取被引用消息失败:响应缺少 message 字段")
        return _extract_images(segments)

    async def _call_action(
        self,
        event: AstrMessageEvent,
        action: str,
        error_prefix: str,
        **params: object,
    ) -> dict[str, object]:
        """执行 OneBot API 调用,统一把失败包装为可展示给用户的 QuotedMessageReadError。"""
        bot = self._resolve_bot(event)
        try:
            result = await bot.call_action(action, **params)
        except Exception as exc:
            raise QuotedMessageReadError(f"{error_prefix}:{exc}") from exc
        if not isinstance(result, dict):
            raise QuotedMessageReadError(f"{error_prefix}:响应格式异常(非 dict)")
        return cast(dict[str, object], result)

    def _resolve_bot(self, event: AstrMessageEvent) -> OneBotBot:
        # get_platform_id 在 AstrBot SDK 中无返回类型注解(在本地 SDK 中可推断为 str),
        # 此处做显式注解边界处理。
        platform_id: str = event.get_platform_id()
        inst = self.context.get_platform_inst(platform_id)
        if inst is None:
            raise QuotedMessageReadError("当前平台不支持读取转发消息内容。")
        # aiocqhttp 适配器实例的 OneBot API 对象是 .bot(CQHttp),而非适配器本身。
        bot = getattr(inst, "bot", None)
        if not isinstance(bot, OneBotBot):
            raise QuotedMessageReadError("当前平台不支持读取转发消息内容。")
        return bot


def _parse_forward_node(node: object) -> ForwardNodeContent | None:
    """防御解析单个转发节点;结构异常或无图片时返回 None。

    参数声明为 object:运行时数据未经验证(TypedDict 不支持 isinstance 检查,
    只能先查 dict 再在边界 cast 到命名 TypedDict)。
    """
    if not isinstance(node, dict):
        logger.warning("[onebot-client] 转发节点不是 dict,已跳过")
        return None
    # 边界:外部 JSON → 命名 TypedDict。
    payload = cast(ForwardNodePayload, node)
    data = payload.get("data")
    if not isinstance(data, dict):
        logger.warning("[onebot-client] 转发节点 data 不是 dict,已跳过")
        return None
    user_id = data.get("user_id")
    nickname = data.get("nickname")
    uin = str(user_id) if user_id is not None else ""
    name = str(nickname) if nickname is not None else ""
    content = data.get("content")
    if not isinstance(content, list):
        logger.warning("[onebot-client] 转发节点 content 不是 list,已跳过")
        return None
    components = _extract_images(content)
    if not components:
        return None
    return ForwardNodeContent(uin=uin, name=name, components=components)


def _extract_images(segments: Sequence[object]) -> list[Comp.BaseMessageComponent]:
    """从消息段序列中提取全部 image 段转换后的组件(其他段忽略)。"""
    components: list[Comp.BaseMessageComponent] = []
    for segment in segments:
        image = _extract_image_segment(segment)
        if image is not None:
            components.append(image)
    return components


def _extract_image_segment(segment: object) -> Comp.Image | None:
    """只提取 image 段;其他段(text/face/at 等)与畸形段一律忽略。"""
    if not isinstance(segment, dict):
        return None
    # 边界:外部 JSON → 命名 TypedDict。
    seg = cast(OneBotSegment, segment)
    if seg.get("type") != "image":
        return None
    data = seg.get("data")
    if not isinstance(data, dict):
        return None
    return _image_from_segment_data(data)


def _image_from_segment_data(data: OneBotSegmentData) -> Comp.Image | None:
    """从图片段 data 中取 file/url/path/file_unique 存在键构造 Comp.Image。

    只传存在的键;全部缺失(或为空串)则返回 None,该段被跳过。
    Comp.Image 构造器要求 file 位置参数,file 键缺失时传 None(其
    convert_to_file_path 会优先使用 url)。
    """
    kwargs: dict[str, str] = {}
    for key in ("file", "url", "path", "file_unique"):
        value = data.get(key)
        if isinstance(value, str) and value:
            kwargs[key] = value
    if not kwargs:
        return None
    file_value = kwargs.pop("file", None)
    return Comp.Image(file=file_value, **kwargs)
