"""集成测试:QuotedMessageReader × FakeBot(get_forward_msg / get_msg)。"""

from __future__ import annotations

import pytest

import astrbot.api.message_components as Comp

from _fakes import (
    FakeBot,
    FakeCtx,
    FakeEvent,
    FakeInst,
    as_context,
    as_event,
    forward_node_payload,
    forward_response,
    image_file_uri,
    image_segment,
    text_segment,
)
from onebot_client import QuotedMessageReadError, QuotedMessageReader


def _reader(bot: FakeBot) -> tuple[QuotedMessageReader, FakeEvent]:
    ctx = FakeCtx(FakeInst(bot))
    return QuotedMessageReader(as_context(ctx)), FakeEvent([])


async def test_fetch_forward_parses_image_nodes() -> None:
    """正常解析:混合段只取 image,int/str user_id 都转字符串,纯文本/空节点跳过。"""
    bot = FakeBot()
    bot.preset(
        "get_forward_msg",
        forward_response(
            [
                forward_node_payload(
                    12345,
                    "alice",
                    [
                        text_segment("hello"),
                        image_segment(image_file_uri("/tmp/a.png")),
                        image_segment(image_file_uri("/tmp/b.png")),
                    ],
                ),
                forward_node_payload(
                    "67890",
                    "bob",
                    [image_segment(image_file_uri("/tmp/c.png"))],
                ),
                forward_node_payload(111, "text-only", [text_segment("no images")]),
                forward_node_payload(222, "empty", []),
            ]
        ),
    )
    reader, event = _reader(bot)
    contents = await reader.fetch_forward(as_event(event), "fwd-1")
    assert [content.uin for content in contents] == ["12345", "67890"]
    assert [content.name for content in contents] == ["alice", "bob"]
    assert [
        [
            component.file
            for component in content.components
            if isinstance(component, Comp.Image)
        ]
        for content in contents
    ] == [
        [image_file_uri("/tmp/a.png"), image_file_uri("/tmp/b.png")],
        [image_file_uri("/tmp/c.png")],
    ]
    assert bot.calls == [("get_forward_msg", {"message_id": "fwd-1"})]


async def test_fetch_forward_all_no_image_nodes_returns_empty() -> None:
    """全部节点无图片 → 返回空列表。"""
    bot = FakeBot()
    bot.preset(
        "get_forward_msg",
        forward_response(
            [
                forward_node_payload(1, "a", [text_segment("only text")]),
                forward_node_payload(2, "b", []),
            ]
        ),
    )
    reader, event = _reader(bot)
    contents = await reader.fetch_forward(as_event(event), "fwd-1")
    assert contents == []


async def test_fetch_forward_defensive_skips_malformed_nodes() -> None:
    """畸形节点(非 dict / 无 data / content 非 list)逐个跳过,不整体失败。"""
    bot = FakeBot()
    bot.preset(
        "get_forward_msg",
        forward_response(
            [
                "junk-node",
                {"type": "node"},
                {
                    "type": "node",
                    "data": {"user_id": 1, "nickname": "x", "content": "junk"},
                },
                {
                    "type": "node",
                    "data": {
                        "user_id": 2,
                        "nickname": "y",
                        "content": [image_segment(image_file_uri("/tmp/z.png"))],
                    },
                },
            ]
        ),
    )
    reader, event = _reader(bot)
    contents = await reader.fetch_forward(as_event(event), "fwd-1")
    assert [content.uin for content in contents] == ["2"]


async def test_fetch_forward_non_dict_response_raises() -> None:
    """响应非 dict → QuotedMessageReadError。"""
    bot = FakeBot()
    bot.preset("get_forward_msg", ["not", "a", "dict"])
    reader, event = _reader(bot)
    with pytest.raises(QuotedMessageReadError, match="非 dict"):
        await reader.fetch_forward(as_event(event), "fwd-1")


async def test_fetch_forward_missing_messages_field_returns_empty() -> None:
    """响应缺 messages 字段 → 返回空列表(不抛错)。"""
    bot = FakeBot()
    bot.preset("get_forward_msg", {"unexpected": 1})
    reader, event = _reader(bot)
    contents = await reader.fetch_forward(as_event(event), "fwd-1")
    assert contents == []


async def test_fetch_forward_call_action_error_wrapped() -> None:
    """call_action 抛异常 → 包装为 QuotedMessageReadError 且含原文。"""
    bot = FakeBot()
    bot.preset_error("get_forward_msg", RuntimeError("connection refused"))
    reader, event = _reader(bot)
    with pytest.raises(QuotedMessageReadError) as exc_info:
        await reader.fetch_forward(as_event(event), "fwd-1")
    assert "读取转发消息失败" in str(exc_info.value)
    assert "connection refused" in str(exc_info.value)


async def test_fetch_quoted_message_normal() -> None:
    """get_msg 正常:只返回 image 段;数字 message_id 以 int 传参。"""
    bot = FakeBot()
    bot.preset(
        "get_msg",
        {
            "message": [
                image_segment(image_file_uri("/tmp/a.png")),
                text_segment("ignored"),
                image_segment(image_file_uri("/tmp/b.png")),
            ]
        },
    )
    reader, event = _reader(bot)
    components = await reader.fetch_quoted_message(as_event(event), "42")
    assert len(components) == 2
    assert all(isinstance(component, Comp.Image) for component in components)
    assert [
        component.file for component in components if isinstance(component, Comp.Image)
    ] == [
        image_file_uri("/tmp/a.png"),
        image_file_uri("/tmp/b.png"),
    ]
    assert bot.calls == [("get_msg", {"message_id": 42})]


async def test_fetch_quoted_message_non_numeric_id_passed_as_str() -> None:
    """非数字 message_id 原样传字符串。"""
    bot = FakeBot()
    bot.preset("get_msg", {"message": [image_segment(image_file_uri("/tmp/a.png"))]})
    reader, event = _reader(bot)
    await reader.fetch_quoted_message(as_event(event), "abc-123")
    assert bot.calls == [("get_msg", {"message_id": "abc-123"})]


async def test_fetch_quoted_message_non_dict_response_raises() -> None:
    """响应非 dict → QuotedMessageReadError。"""
    bot = FakeBot()
    bot.preset("get_msg", [1, 2, 3])
    reader, event = _reader(bot)
    with pytest.raises(QuotedMessageReadError, match="非 dict"):
        await reader.fetch_quoted_message(as_event(event), "42")


async def test_fetch_quoted_message_missing_message_field_raises() -> None:
    """响应缺 message 字段 → QuotedMessageReadError。"""
    bot = FakeBot()
    bot.preset("get_msg", {"ok": True})
    reader, event = _reader(bot)
    with pytest.raises(QuotedMessageReadError, match="缺少 message 字段"):
        await reader.fetch_quoted_message(as_event(event), "42")


async def test_fetch_quoted_message_call_action_error_wrapped() -> None:
    """call_action 抛异常 → 包装为 QuotedMessageReadError 且含原文。"""
    bot = FakeBot()
    bot.preset_error("get_msg", RuntimeError("deep fail"))
    reader, event = _reader(bot)
    with pytest.raises(QuotedMessageReadError) as exc_info:
        await reader.fetch_quoted_message(as_event(event), "abc")
    assert "读取被引用消息失败" in str(exc_info.value)
    assert "deep fail" in str(exc_info.value)


async def test_fetch_forward_platform_unsupported_no_inst() -> None:
    """get_platform_inst 返回 None → QuotedMessageReadError("当前平台不支持…")。"""
    reader = QuotedMessageReader(as_context(FakeCtx(None)))
    event = FakeEvent([])
    with pytest.raises(QuotedMessageReadError, match="当前平台不支持"):
        await reader.fetch_forward(as_event(event), "fwd-1")


async def test_fetch_forward_platform_unsupported_inst_without_bot() -> None:
    """get_platform_inst 返回无 .bot 的对象 → QuotedMessageReadError("当前平台不支持…")。"""
    reader = QuotedMessageReader(as_context(FakeCtx(FakeInst(None))))
    event = FakeEvent([])
    with pytest.raises(QuotedMessageReadError, match="当前平台不支持"):
        await reader.fetch_forward(as_event(event), "fwd-1")


async def test_fetch_quoted_platform_unsupported_no_inst() -> None:
    """get_msg 兜底同样受平台支持检查约束。"""
    reader = QuotedMessageReader(as_context(FakeCtx(None)))
    event = FakeEvent([])
    with pytest.raises(QuotedMessageReadError, match="当前平台不支持"):
        await reader.fetch_quoted_message(as_event(event), "42")
