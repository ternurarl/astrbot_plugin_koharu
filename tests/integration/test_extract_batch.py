"""集成测试:_extract_image_batch 的图片/引用/转发提取与去重。"""

from __future__ import annotations

import pytest

import astrbot.api.message_components as Comp

from _fakes import (
    FakeBot,
    FakeCtx,
    FakeEvent,
    FakeInst,
    extract_image_batch,
    forward_node_payload,
    forward_response,
    forward_segment,
    image_file_uri,
    image_from_path,
    image_segment,
    text_segment,
)
from conftest import MakePlugin
from onebot_client import QuotedMessageReadError


async def test_direct_images_preserve_order(make_plugin: MakePlugin) -> None:
    """普通消息直接携带多张图片:有序提取,非转发。"""
    plugin = make_plugin()
    event = FakeEvent(
        [
            image_from_path("/tmp/a.png"),
            image_from_path("/tmp/b.png"),
            image_from_path("/tmp/c.png"),
        ]
    )
    batch = await extract_image_batch(plugin, event)
    assert batch.image_paths == ["/tmp/a.png", "/tmp/b.png", "/tmp/c.png"]
    assert batch.forward_nodes is None


async def test_reply_chain_images_collected(make_plugin: MakePlugin) -> None:
    """引用消息 chain 含图片(混合文本段):只收集图片,非转发。"""
    plugin = make_plugin()
    reply = Comp.Reply(
        id="r1",
        chain=[
            image_from_path("/tmp/a.png"),
            Comp.Plain("some quoted text"),
            image_from_path("/tmp/b.png"),
        ],
    )
    batch = await extract_image_batch(plugin, FakeEvent([reply]))
    assert batch.image_paths == ["/tmp/a.png", "/tmp/b.png"]
    assert batch.forward_nodes is None


async def test_reply_chain_image_convert_failure_skipped(make_plugin: MakePlugin) -> None:
    """引用消息 chain 中单张图 convert 失败被跳过,其余图保留,异常不冒泡。"""
    plugin = make_plugin()
    reply = Comp.Reply(
        id="r1",
        chain=[
            image_from_path("/tmp/good.png"),
            Comp.Image(file="definitely-missing-file-xyz.png"),
        ],
    )
    batch = await extract_image_batch(plugin, FakeEvent([reply]))
    assert batch.image_paths == ["/tmp/good.png"]
    assert batch.forward_nodes is None


async def test_reply_chain_forward_nodes(make_plugin: MakePlugin) -> None:
    """引用消息 chain 含 Forward:节点 uin/name/图片下标按原始顺序正确。"""
    bot = FakeBot()
    bot.preset(
        "get_forward_msg",
        forward_response(
            [
                forward_node_payload(
                    12345,  # int user_id → "12345"
                    "alice",
                    [
                        text_segment("ignored text"),
                        image_segment(image_file_uri("/tmp/a.png")),
                        image_segment(image_file_uri("/tmp/b.png")),
                    ],
                ),
                forward_node_payload(
                    "67890",  # str user_id
                    "bob",
                    [image_segment(image_file_uri("/tmp/c.png"))],
                ),
            ]
        ),
    )
    plugin = make_plugin(context=FakeCtx(FakeInst(bot)))
    event = FakeEvent([Comp.Reply(id="r1", chain=[Comp.Forward(id="fwd-1")])])
    batch = await extract_image_batch(plugin, event)
    assert batch.forward_nodes is not None
    assert [node.uin for node in batch.forward_nodes] == ["12345", "67890"]
    assert [node.name for node in batch.forward_nodes] == ["alice", "bob"]
    assert [node.image_indices for node in batch.forward_nodes] == [[0, 1], [2]]
    assert batch.image_paths == ["/tmp/a.png", "/tmp/b.png", "/tmp/c.png"]


async def test_mixed_images_dedupe_and_index_mapping(make_plugin: MakePlugin) -> None:
    """顶层图片 + 引用转发混合;同一路径出现两次 → 唯一列表 2 项,下标映射正确。"""
    bot = FakeBot()
    bot.preset(
        "get_forward_msg",
        forward_response(
            [
                forward_node_payload(
                    "1",
                    "a",
                    [
                        image_segment(image_file_uri("/tmp/x.png")),
                        image_segment(image_file_uri("/tmp/y.png")),
                    ],
                ),
            ]
        ),
    )
    plugin = make_plugin(context=FakeCtx(FakeInst(bot)))
    event = FakeEvent(
        [
            image_from_path("/tmp/x.png"),
            Comp.Reply(id="r1", chain=[Comp.Forward(id="fwd-1")]),
        ]
    )
    batch = await extract_image_batch(plugin, event)
    # raw 顺序:顶层 x、转发 x、转发 y → 去重后唯一 2 项
    assert batch.image_paths == ["/tmp/x.png", "/tmp/y.png"]
    assert batch.forward_nodes is not None
    # 转发节点原始下标 1、2 → 映射到唯一列表下标 0、1
    assert batch.forward_nodes[0].image_indices == [0, 1]


async def test_reply_empty_chain_falls_back_to_get_msg(make_plugin: MakePlugin) -> None:
    """Reply.chain 为空且有 id:get_msg 兜底,结果非转发。"""
    bot = FakeBot()
    bot.preset(
        "get_msg",
        {
            "message": [
                image_segment(image_file_uri("/tmp/q.png")),
                text_segment("ignored"),
            ]
        },
    )
    plugin = make_plugin(context=FakeCtx(FakeInst(bot)))
    event = FakeEvent([Comp.Reply(id="42")])
    batch = await extract_image_batch(plugin, event)
    assert batch.image_paths == ["/tmp/q.png"]
    assert batch.forward_nodes is None
    # message_id 为数字字符串时以 int 传参
    assert bot.calls == [("get_msg", {"message_id": 42})]


async def test_reply_empty_chain_falls_back_to_forward(make_plugin: MakePlugin) -> None:
    """Reply.chain 为空但被引用消息是合并转发:get_msg 兜底展开,按转发输出。"""
    bot = FakeBot()
    bot.preset(
        "get_msg",
        {"message": [forward_segment("fwd-9")]},
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
    plugin = make_plugin(context=FakeCtx(FakeInst(bot)))
    event = FakeEvent([Comp.Reply(id="42")])
    batch = await extract_image_batch(plugin, event)
    assert batch.forward_nodes is not None
    assert [node.uin for node in batch.forward_nodes] == ["10001", "10002"]
    assert [node.name for node in batch.forward_nodes] == ["alice", "bob"]
    assert [node.image_indices for node in batch.forward_nodes] == [[0], [1]]
    assert batch.image_paths == ["/tmp/a.png", "/tmp/b.png"]
    assert bot.calls == [
        ("get_msg", {"message_id": 42}),
        ("get_forward_msg", {"message_id": "fwd-9"}),
    ]


async def test_reply_chain_only_placeholder_falls_back_to_forward(make_plugin: MakePlugin) -> None:
    """Reply.chain 只有占位文本(如合并转发渲染成的 "[聊天记录]"):落到 id 兜底展开转发。"""
    bot = FakeBot()
    bot.preset(
        "get_msg",
        {"message": [forward_segment("fwd-9")]},
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
            ]
        ),
    )
    plugin = make_plugin(context=FakeCtx(FakeInst(bot)))
    event = FakeEvent([Comp.Reply(id="42", chain=[Comp.Plain("[聊天记录]")])])
    batch = await extract_image_batch(plugin, event)
    assert batch.forward_nodes is not None
    assert batch.forward_nodes[0].uin == "10001"
    assert batch.forward_nodes[0].image_indices == [0]
    assert batch.image_paths == ["/tmp/a.png"]


async def test_reply_chain_image_plus_forward_no_fallback(make_plugin: MakePlugin) -> None:
    """Reply.chain 已含图片:不再触发 get_msg 兜底(id 存在也不拉取)。"""
    bot = FakeBot()
    plugin = make_plugin(context=FakeCtx(FakeInst(bot)))
    event = FakeEvent(
        [Comp.Reply(id="42", chain=[image_from_path("/tmp/a.png"), Comp.Plain("[聊天记录]")])]
    )
    batch = await extract_image_batch(plugin, event)
    assert batch.image_paths == ["/tmp/a.png"]
    assert batch.forward_nodes is None
    assert bot.calls == []


async def test_top_level_forward_with_nested_forward(make_plugin: MakePlugin) -> None:
    """转发记录节点内嵌套合并转发:嵌套记录图片按顺序展开为独立节点。"""
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
                            forward_segment("fwd-inner"),
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
    plugin = make_plugin(context=FakeCtx(FakeInst(bot)))
    event = FakeEvent([Comp.Forward(id="fwd-outer")])
    batch = await extract_image_batch(plugin, event)
    assert batch.forward_nodes is not None
    assert [(node.uin, node.name, node.image_indices) for node in batch.forward_nodes] == [
        ("1", "outer", [0]),
        ("2", "inner", [1]),
    ]
    assert batch.image_paths == ["/tmp/o.png", "/tmp/i.png"]


async def test_top_level_forward_and_direct_image_mixed(make_plugin: MakePlugin) -> None:
    """顶层 Comp.Forward 与顶层 Image 混合:转发节点 + 直接图片都保留。"""
    bot = FakeBot()
    bot.preset(
        "get_forward_msg",
        forward_response(
            [forward_node_payload("7", "seven", [image_segment(image_file_uri("/tmp/f.png"))])]
        ),
    )
    plugin = make_plugin(context=FakeCtx(FakeInst(bot)))
    event = FakeEvent([Comp.Forward(id="fwd-9"), image_from_path("/tmp/d.png")])
    batch = await extract_image_batch(plugin, event)
    assert batch.forward_nodes is not None
    assert batch.forward_nodes[0].uin == "7"
    assert batch.forward_nodes[0].image_indices == [0]
    assert batch.image_paths == ["/tmp/f.png", "/tmp/d.png"]


async def test_fetch_forward_error_propagates(make_plugin: MakePlugin) -> None:
    """fetch_forward 抛异常 → QuotedMessageReadError 冒泡,带原文。"""
    bot = FakeBot()
    bot.preset_error("get_forward_msg", RuntimeError("network down"))
    plugin = make_plugin(context=FakeCtx(FakeInst(bot)))
    event = FakeEvent([Comp.Forward(id="fwd-1")])
    with pytest.raises(QuotedMessageReadError) as exc_info:
        await extract_image_batch(plugin, event)
    assert "读取转发消息失败" in str(exc_info.value)
    assert "network down" in str(exc_info.value)


async def test_forward_node_single_image_convert_failure_skipped(make_plugin: MakePlugin) -> None:
    """转发节点内单张图 convert 失败被跳过,该节点保留其余图;全失败的节点整体跳过。"""
    bot = FakeBot()
    bot.preset(
        "get_forward_msg",
        forward_response(
            [
                forward_node_payload(
                    "1",
                    "a",
                    [
                        image_segment(image_file_uri("/tmp/good.png")),
                        image_segment("definitely-missing-file-xyz.png"),
                    ],
                ),
                forward_node_payload(
                    "2",
                    "b",
                    [image_segment("another-missing-file-xyz.png")],
                ),
            ]
        ),
    )
    plugin = make_plugin(context=FakeCtx(FakeInst(bot)))
    event = FakeEvent([Comp.Forward(id="fwd-1")])
    batch = await extract_image_batch(plugin, event)
    assert batch.image_paths == ["/tmp/good.png"]
    assert batch.forward_nodes is not None
    assert len(batch.forward_nodes) == 1
    assert batch.forward_nodes[0].uin == "1"
    assert batch.forward_nodes[0].name == "a"
    assert batch.forward_nodes[0].image_indices == [0]


async def test_extract_platform_unsupported_bubbles(make_plugin: MakePlugin) -> None:
    """平台不支持(无 .bot)→ 兜底 get_msg 前即抛 QuotedMessageReadError。"""
    plugin = make_plugin(context=FakeCtx(None))
    event = FakeEvent([Comp.Reply(id="42")])
    with pytest.raises(QuotedMessageReadError, match="当前平台不支持"):
        await extract_image_batch(plugin, event)
