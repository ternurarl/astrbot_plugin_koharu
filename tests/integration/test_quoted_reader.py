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
    forward_message_node,
    forward_node_payload,
    forward_response,
    forward_segment,
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
    content = await reader.fetch_quoted_message(as_event(event), "42")
    assert len(content.images) == 2
    assert len(content.forward_nodes) == 0
    assert all(isinstance(component, Comp.Image) for component in content.images)
    assert [
        component.file for component in content.images if isinstance(component, Comp.Image)
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


async def test_fetch_quoted_message_forward_segment_expands() -> None:
    """被引用消息是合并转发(get_msg 返回 forward 占位段):展开为转发节点。"""
    bot = FakeBot()
    bot.preset(
        "get_msg",
        {"message": [text_segment("[聊天记录]"), forward_segment("fwd-9")]},
    )
    bot.preset(
        "get_forward_msg",
        forward_response(
            [
                forward_node_payload(
                    "10001",
                    "alice",
                    [image_segment(image_file_uri("/tmp/a.png"))],
                ),
                forward_node_payload(
                    "10002",
                    "bob",
                    [image_segment(image_file_uri("/tmp/b.png"))],
                ),
            ]
        ),
    )
    reader, event = _reader(bot)
    content = await reader.fetch_quoted_message(as_event(event), "42")
    assert content.images == []
    assert [node.uin for node in content.forward_nodes] == ["10001", "10002"]
    assert [node.name for node in content.forward_nodes] == ["alice", "bob"]
    assert [
        component.file
        for node in content.forward_nodes
        for component in node.components
        if isinstance(component, Comp.Image)
    ] == [
        image_file_uri("/tmp/a.png"),
        image_file_uri("/tmp/b.png"),
    ]
    assert bot.calls == [
        ("get_msg", {"message_id": 42}),
        ("get_forward_msg", {"message_id": "fwd-9"}),
    ]


async def test_fetch_quoted_message_inline_nodes_parsed() -> None:
    """get_msg 直接返回内联 node 段(部分实现内联展开转发):解析为转发节点。"""
    bot = FakeBot()
    bot.preset(
        "get_msg",
        {
            "message": [
                forward_node_payload(
                    "20001",
                    "carol",
                    [image_segment(image_file_uri("/tmp/c.png"))],
                ),
            ]
        },
    )
    reader, event = _reader(bot)
    content = await reader.fetch_quoted_message(as_event(event), "42")
    assert content.images == []
    assert len(content.forward_nodes) == 1
    assert content.forward_nodes[0].uin == "20001"
    assert content.forward_nodes[0].name == "carol"
    assert not bot.calls or bot.calls[-1][0] == "get_msg"


async def test_fetch_quoted_message_mixed_images_and_forward() -> None:
    """直接图片段与合并转发段共存:images 与 forward_nodes 分别返回。"""
    bot = FakeBot()
    bot.preset(
        "get_msg",
        {
            "message": [
                image_segment(image_file_uri("/tmp/d.png")),
                forward_segment("fwd-1"),
            ]
        },
    )
    bot.preset(
        "get_forward_msg",
        forward_response(
            [
                forward_node_payload(
                    "30001",
                    "dave",
                    [image_segment(image_file_uri("/tmp/e.png"))],
                ),
            ]
        ),
    )
    reader, event = _reader(bot)
    content = await reader.fetch_quoted_message(as_event(event), "42")
    assert [
        component.file for component in content.images if isinstance(component, Comp.Image)
    ] == [image_file_uri("/tmp/d.png")]
    assert len(content.forward_nodes) == 1
    assert content.forward_nodes[0].uin == "30001"


async def test_fetch_forward_nested_forward_expands() -> None:
    """转发节点内嵌套合并转发(节点内容为 forward 段):递归展开为独立节点。"""
    bot = FakeBot()
    bot.preset_sequence(
        "get_forward_msg",
        [
            forward_response(
                [
                    forward_node_payload(
                        "1",
                        "outer",
                        [
                            image_segment(image_file_uri("/tmp/o1.png")),
                            forward_segment("fwd-inner"),
                            image_segment(image_file_uri("/tmp/o2.png")),
                        ],
                    ),
                ]
            ),
            forward_response(
                [
                    forward_node_payload(
                        "2",
                        "inner",
                        [image_segment(image_file_uri("/tmp/i.png"))],
                    ),
                ]
            ),
        ],
    )
    reader, event = _reader(bot)
    contents = await reader.fetch_forward(as_event(event), "fwd-outer")
    # 顺序:outer 首图 → inner 节点 → outer 尾图(保持原始记录顺序)
    assert [(node.uin, node.name) for node in contents] == [
        ("1", "outer"),
        ("2", "inner"),
        ("1", "outer"),
    ]
    assert [
        component.file
        for node in contents
        for component in node.components
        if isinstance(component, Comp.Image)
    ] == [
        image_file_uri("/tmp/o1.png"),
        image_file_uri("/tmp/i.png"),
        image_file_uri("/tmp/o2.png"),
    ]
    assert bot.calls == [
        ("get_forward_msg", {"message_id": "fwd-outer"}),
        ("get_forward_msg", {"message_id": "fwd-inner"}),
    ]


async def test_fetch_forward_nested_forward_empty_skipped() -> None:
    """嵌套合并转发为空记录:外层节点图片保留,嵌套不产生节点。"""
    bot = FakeBot()
    bot.preset_sequence(
        "get_forward_msg",
        [
            forward_response(
                [
                    forward_node_payload(
                        "1",
                        "outer",
                        [
                            image_segment(image_file_uri("/tmp/o.png")),
                            forward_segment("fwd-empty"),
                        ],
                    ),
                ]
            ),
            forward_response([]),
        ],
    )
    reader, event = _reader(bot)
    contents = await reader.fetch_forward(as_event(event), "fwd-outer")
    assert len(contents) == 1
    assert contents[0].uin == "1"
    assert contents[0].name == "outer"


async def test_fetch_forward_nested_forward_missing_id_skipped() -> None:
    """嵌套 forward 段缺 id:跳过该段,不拖垮整个节点。"""
    bot = FakeBot()
    bot.preset(
        "get_forward_msg",
        forward_response(
            [
                forward_node_payload(
                    "1",
                    "outer",
                    [
                        image_segment(image_file_uri("/tmp/o.png")),
                        {"type": "forward", "data": {}},
                    ],
                ),
            ]
        ),
    )
    reader, event = _reader(bot)
    contents = await reader.fetch_forward(as_event(event), "fwd-outer")
    assert len(contents) == 1
    assert contents[0].uin == "1"
    assert bot.calls == [("get_forward_msg", {"message_id": "fwd-outer"})]


async def test_fetch_forward_nested_inline_node_expands() -> None:
    """节点内容内联 node 段:按内联节点自己的 uin/name 展开为独立节点。"""
    bot = FakeBot()
    bot.preset(
        "get_forward_msg",
        forward_response(
            [
                forward_node_payload(
                    "1",
                    "outer",
                    [
                        forward_node_payload(
                            "9",
                            "inline",
                            [image_segment(image_file_uri("/tmp/in.png"))],
                        ),
                    ],
                ),
            ]
        ),
    )
    reader, event = _reader(bot)
    contents = await reader.fetch_forward(as_event(event), "fwd-outer")
    assert [(node.uin, node.name) for node in contents] == [("9", "inline")]


async def test_fetch_forward_parses_napcat_message_nodes() -> None:
    """NapCat 形态:节点是完整消息对象(OB11Message,无 type/data 包装)。"""
    bot = FakeBot()
    bot.preset(
        "get_forward_msg",
        forward_response(
            [
                forward_message_node(
                    "10001",
                    "alice",
                    [
                        text_segment("hello"),
                        image_segment(image_file_uri("/tmp/a.png")),
                    ],
                ),
                forward_message_node(
                    67890,  # int user_id → "67890"
                    "bob",
                    [image_segment(image_file_uri("/tmp/b.png"))],
                ),
                forward_message_node("111", "text-only", [text_segment("no images")]),
            ]
        ),
    )
    reader, event = _reader(bot)
    contents = await reader.fetch_forward(as_event(event), "fwd-1")
    assert [content.uin for content in contents] == ["10001", "67890"]
    assert [content.name for content in contents] == ["alice", "bob"]
    assert [
        component.file
        for content in contents
        for component in content.components
        if isinstance(component, Comp.Image)
    ] == [
        image_file_uri("/tmp/a.png"),
        image_file_uri("/tmp/b.png"),
    ]
    assert bot.calls == [("get_forward_msg", {"message_id": "fwd-1"})]


async def test_fetch_forward_mixed_node_shapes() -> None:
    """标准 node 段与 NapCat 消息对象混合:两种形态都解析。"""
    bot = FakeBot()
    bot.preset(
        "get_forward_msg",
        forward_response(
            [
                forward_node_payload(
                    "1",
                    "std",
                    [image_segment(image_file_uri("/tmp/std.png"))],
                ),
                forward_message_node(
                    "2",
                    "nap",
                    [image_segment(image_file_uri("/tmp/nap.png"))],
                ),
            ]
        ),
    )
    reader, event = _reader(bot)
    contents = await reader.fetch_forward(as_event(event), "fwd-1")
    assert [(content.uin, content.name) for content in contents] == [
        ("1", "std"),
        ("2", "nap"),
    ]


async def test_fetch_forward_napcat_nested_forward_inline() -> None:
    """NapCat 形态嵌套:forward 段 data.content 已内联展开,直接解析不再拉取。"""
    bot = FakeBot()
    bot.preset(
        "get_forward_msg",
        forward_response(
            [
                forward_message_node(
                    "1",
                    "outer",
                    [
                        image_segment(image_file_uri("/tmp/o.png")),
                        {
                            "type": "forward",
                            "data": {
                                "id": "nested-msg-id",
                                "content": [
                                    forward_message_node(
                                        "2",
                                        "inner",
                                        [image_segment(image_file_uri("/tmp/i.png"))],
                                    ),
                                ],
                            },
                        },
                    ],
                ),
            ]
        ),
    )
    reader, event = _reader(bot)
    contents = await reader.fetch_forward(as_event(event), "fwd-1")
    assert [(content.uin, content.name) for content in contents] == [
        ("1", "outer"),
        ("2", "inner"),
    ]
    assert [
        component.file
        for content in contents
        for component in content.components
        if isinstance(component, Comp.Image)
    ] == [
        image_file_uri("/tmp/o.png"),
        image_file_uri("/tmp/i.png"),
    ]
    # 内联内容直接解析,不额外调用 get_forward_msg
    assert bot.calls == [("get_forward_msg", {"message_id": "fwd-1"})]


async def test_fetch_forward_standard_node_message_key() -> None:
    """标准 node 段但内容在 data.message 键(部分实现形态):兼容解析。"""
    bot = FakeBot()
    bot.preset(
        "get_forward_msg",
        forward_response(
            [
                {
                    "type": "node",
                    "data": {
                        "user_id": "7",
                        "nickname": "seven",
                        "message": [image_segment(image_file_uri("/tmp/m.png"))],
                    },
                },
            ]
        ),
    )
    reader, event = _reader(bot)
    contents = await reader.fetch_forward(as_event(event), "fwd-1")
    assert len(contents) == 1
    assert contents[0].uin == "7"
    assert contents[0].name == "seven"


async def test_fetch_forward_napcat_message_node_malformed_skipped() -> None:
    """NapCat 形态畸形对象(message 非 list / 非 dict)逐个跳过。"""
    bot = FakeBot()
    bot.preset(
        "get_forward_msg",
        forward_response(
            [
                {
                    "user_id": "1",
                    "sender": {"user_id": "1", "nickname": "bad"},
                    "message": "not-a-list",
                },
                "junk",
                forward_message_node("2", "good", [image_segment(image_file_uri("/tmp/g.png"))]),
            ]
        ),
    )
    reader, event = _reader(bot)
    contents = await reader.fetch_forward(as_event(event), "fwd-1")
    assert [content.uin for content in contents] == ["2"]


async def test_fetch_forward_nested_depth_capped() -> None:
    """嵌套过深:递归在深度上限处截断,不无限调用。"""
    depth = 12
    chain: list[object] = []
    for index in range(depth):
        chain.append(
            forward_response(
                [
                    forward_node_payload(
                        str(index),
                        f"n{index}",
                        [forward_segment(f"fwd-{index + 1}")],
                    ),
                ]
            )
        )
    bot = FakeBot()
    bot.preset_sequence("get_forward_msg", chain)
    reader, event = _reader(bot)
    contents = await reader.fetch_forward(as_event(event), "fwd-0")
    assert contents == []
    # 深度上限 _MAX_FORWARD_DEPTH=10:0..10 共 11 层会发起读取,更深处截断
    assert len(bot.calls) == 11


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
