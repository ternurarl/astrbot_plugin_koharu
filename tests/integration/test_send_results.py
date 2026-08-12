"""集成测试:_send_forward_result / _send_one_by_one 的截断与防御。"""

from __future__ import annotations

from pathlib import Path

import astrbot.api.message_components as Comp

from _fakes import (
    FakeEvent,
    image_file_uri,
    send_forward_result,
    send_one_by_one,
)
from conftest import MakePlugin
from main import ForwardNode, QuotedBatch


def _make_outputs(tmp_path: Path, names: list[str]) -> list[str]:
    """在 tmp_path 下创建真实输出文件并返回路径列表(供 Nodes.to_dict 读取)。"""
    outputs: list[str] = []
    for name in names:
        output_path = tmp_path / name
        output_path.write_bytes(b"fake-image-bytes")
        outputs.append(str(output_path))
    return outputs


async def test_forward_max_send_truncates_from_end(make_plugin: MakePlugin, tmp_path: Path) -> None:
    """转发:max_send_images 为总预算,超出部分从尾部节点丢弃。"""
    plugin = make_plugin(overrides={"max_send_images": 2})
    outputs = _make_outputs(tmp_path, ["o1.png", "o2.png", "o3.png", "o4.png"])
    batch = QuotedBatch(
        image_paths=["/tmp/a.png", "/tmp/b.png", "/tmp/c.png", "/tmp/d.png"],
        forward_nodes=[
            ForwardNode(uin="1", name="a", image_indices=[0, 1]),
            ForwardNode(uin="2", name="b", image_indices=[2, 3]),
        ],
    )
    event = FakeEvent([])
    await send_forward_result(plugin, event, batch, outputs)

    assert len(event.sent_chains) == 1
    chain = event.sent_chains[0]
    assert len(chain) == 1
    nodes_comp = chain[0]
    assert isinstance(nodes_comp, Comp.Nodes)
    # 预算 2 全部被第一个节点消耗,第二个节点(尾部)被丢弃
    assert [node.uin for node in nodes_comp.nodes] == ["1"]
    assert len(nodes_comp.nodes[0].content) == 2
    assert [
        comp.file for comp in nodes_comp.nodes[0].content if isinstance(comp, Comp.Image)
    ] == [
        image_file_uri(outputs[0]),
        image_file_uri(outputs[1]),
    ]


async def test_forward_max_send_truncates_mid_node(make_plugin: MakePlugin, tmp_path: Path) -> None:
    """转发:预算在节点中间耗尽 → 该节点保留前 N 张,后续节点全部丢弃。"""
    plugin = make_plugin(overrides={"max_send_images": 3})
    outputs = _make_outputs(tmp_path, ["o1.png", "o2.png", "o3.png", "o4.png", "o5.png"])
    batch = QuotedBatch(
        image_paths=["/tmp/a.png", "/tmp/b.png", "/tmp/c.png", "/tmp/d.png", "/tmp/e.png"],
        forward_nodes=[
            ForwardNode(uin="1", name="a", image_indices=[0, 1, 2, 3]),
            ForwardNode(uin="2", name="b", image_indices=[4]),
        ],
    )
    event = FakeEvent([])
    await send_forward_result(plugin, event, batch, outputs)

    nodes_comp = event.sent_chains[0][0]
    assert isinstance(nodes_comp, Comp.Nodes)
    assert [node.uin for node in nodes_comp.nodes] == ["1"]
    assert len(nodes_comp.nodes[0].content) == 3


async def test_forward_index_out_of_range_skipped_keeps_valid(make_plugin: MakePlugin, tmp_path: Path) -> None:
    """转发:节点图片下标越界被防御跳过,有效图仍保留并发送。"""
    plugin = make_plugin()  # max_send_images=0 → 预算 = 输出数
    outputs = _make_outputs(tmp_path, ["o1.png"])
    batch = QuotedBatch(
        image_paths=["/tmp/a.png"],
        forward_nodes=[ForwardNode(uin="1", name="a", image_indices=[0, 99])],
    )
    event = FakeEvent([])
    await send_forward_result(plugin, event, batch, outputs)

    assert len(event.sent_chains) == 1
    nodes_comp = event.sent_chains[0][0]
    assert isinstance(nodes_comp, Comp.Nodes)
    assert len(nodes_comp.nodes) == 1
    assert len(nodes_comp.nodes[0].content) == 1
    first_image = nodes_comp.nodes[0].content[0]
    assert isinstance(first_image, Comp.Image)
    assert first_image.file == image_file_uri(outputs[0])


async def test_forward_all_nodes_empty_no_send(make_plugin: MakePlugin, tmp_path: Path) -> None:
    """转发:所有节点图片下标越界(节点全空)→ 不发送任何消息。"""
    plugin = make_plugin()
    outputs = _make_outputs(tmp_path, ["o1.png", "o2.png"])
    batch = QuotedBatch(
        image_paths=["/tmp/a.png", "/tmp/b.png"],
        forward_nodes=[
            ForwardNode(uin="1", name="a", image_indices=[5, 6]),
            ForwardNode(uin="2", name="b", image_indices=[7]),
        ],
    )
    event = FakeEvent([])
    await send_forward_result(plugin, event, batch, outputs)

    assert event.sent_chains == []


async def test_forward_no_outputs_no_send(make_plugin: MakePlugin, tmp_path: Path) -> None:
    """转发:输出为空 → 不发送任何消息。"""
    plugin = make_plugin()
    batch = QuotedBatch(
        image_paths=["/tmp/a.png"],
        forward_nodes=[ForwardNode(uin="1", name="a", image_indices=[0])],
    )
    event = FakeEvent([])
    await send_forward_result(plugin, event, batch, [])

    assert event.sent_chains == []


async def test_one_by_one_max_send_takes_front(make_plugin: MakePlugin, tmp_path: Path) -> None:
    """非转发:max_send_images 截断取前 N 张。"""
    plugin = make_plugin(overrides={"max_send_images": 2})
    outputs = _make_outputs(tmp_path, ["o1.png", "o2.png", "o3.png", "o4.png"])
    event = FakeEvent([])
    await send_one_by_one(plugin, event, outputs)

    assert len(event.sent_chains) == 2
    for index, path in enumerate(outputs[:2]):
        chain = event.sent_chains[index]
        assert len(chain) == 1
        assert isinstance(chain[0], Comp.Image)
        assert chain[0].file == image_file_uri(path)


async def test_one_by_one_max_send_zero_sends_all(make_plugin: MakePlugin, tmp_path: Path) -> None:
    """非转发:max_send_images=0(不限) → 全部发送。"""
    plugin = make_plugin()
    outputs = _make_outputs(tmp_path, ["o1.png", "o2.png", "o3.png"])
    event = FakeEvent([])
    await send_one_by_one(plugin, event, outputs)

    assert len(event.sent_chains) == 3


async def test_one_by_one_empty_no_send(make_plugin: MakePlugin) -> None:
    """非转发:输出为空 → 不发送任何消息。"""
    plugin = make_plugin()
    event = FakeEvent([])
    await send_one_by_one(plugin, event, [])

    assert event.sent_chains == []
