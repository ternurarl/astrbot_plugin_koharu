# pyright: reportPrivateUsage=false
"""onebot_client.py 私有解析函数单元测试。

覆盖:_extract_image_segment / _image_from_segment_data / _parse_forward_node。
全部为纯 dict → AstrBot 组件 的转换,不涉及任何 OneBot API 调用。
"""

from __future__ import annotations

from astrbot.api.message_components import Image as CompImage

from onebot_client import (
    OneBotSegment,
    OneBotSegmentData,
    _extract_image_segment,
    _image_from_segment_data,
    _parse_forward_node,
)


# --- _extract_image_segment -----------------------------------------------------


def test_extract_image_segment_with_all_fields() -> None:
    segment: OneBotSegment = {
        "type": "image",
        "data": {
            "url": "http://example.com/1.png",
            "file": "cached://1.png",
            "path": "/tmp/1.png",
            "file_unique": "uniq1",
        },
    }
    image = _extract_image_segment(segment)
    assert image is not None
    assert isinstance(image, CompImage)
    assert image.url == "http://example.com/1.png"
    assert image.file == "cached://1.png"
    assert image.path == "/tmp/1.png"
    assert image.file_unique == "uniq1"


def test_extract_image_segment_with_url_only() -> None:
    segment: OneBotSegment = {"type": "image", "data": {"url": "http://example.com/2.png"}}
    image = _extract_image_segment(segment)
    assert image is not None
    assert isinstance(image, CompImage)
    assert image.url == "http://example.com/2.png"
    assert image.file is None  # file 键缺失


def test_extract_image_segment_text_and_face_return_none() -> None:
    text_segment: OneBotSegment = {"type": "text", "data": {"text": "hello"}}
    assert _extract_image_segment(text_segment) is None
    # face 段的 data 含 id 等 image 段没有的键,用原始 dict 形态传入。
    face_segment = {"type": "face", "data": {"id": 1}}
    assert _extract_image_segment(face_segment) is None


def test_extract_image_segment_non_dict_returns_none() -> None:
    assert _extract_image_segment("not a dict") is None
    assert _extract_image_segment(42) is None
    assert _extract_image_segment(None) is None
    assert _extract_image_segment(["image"]) is None


def test_extract_image_segment_missing_data_returns_none() -> None:
    assert _extract_image_segment({"type": "image"}) is None


def test_extract_image_segment_type_not_image_returns_none() -> None:
    record: OneBotSegment = {"type": "record", "data": {"file": "a.mp3"}}
    assert _extract_image_segment(record) is None


# --- _image_from_segment_data ---------------------------------------------------


def test_image_from_segment_data_url_only_file_is_none() -> None:
    data: OneBotSegmentData = {"url": "http://example.com/3.png"}
    image = _image_from_segment_data(data)
    assert image is not None
    assert isinstance(image, CompImage)
    assert image.file is None
    assert image.url == "http://example.com/3.png"


def test_image_from_segment_data_path_and_file_unique() -> None:
    data: OneBotSegmentData = {"path": "/tmp/4.png", "file_unique": "u4"}
    image = _image_from_segment_data(data)
    assert image is not None
    assert isinstance(image, CompImage)
    assert image.file is None
    assert image.path == "/tmp/4.png"
    assert image.file_unique == "u4"


def test_image_from_segment_data_file_plus_url() -> None:
    data: OneBotSegmentData = {"file": "cached://5.png", "url": "http://example.com/5.png"}
    image = _image_from_segment_data(data)
    assert image is not None
    assert isinstance(image, CompImage)
    assert image.file == "cached://5.png"
    assert image.url == "http://example.com/5.png"


def test_image_from_segment_data_all_missing_returns_none() -> None:
    assert _image_from_segment_data({}) is None
    empty_strings: OneBotSegmentData = {"file": "", "url": "", "path": "", "file_unique": ""}
    assert _image_from_segment_data(empty_strings) is None


# --- _parse_forward_node --------------------------------------------------------


def test_parse_forward_node_with_int_user_id() -> None:
    node = {
        "type": "node",
        "data": {
            "user_id": 12345,
            "nickname": "Alice",
            "content": [
                {"type": "image", "data": {"url": "http://example.com/a.png"}},
                {"type": "text", "data": {"text": "ignored"}},
            ],
        },
    }
    content = _parse_forward_node(node)
    assert content is not None
    assert content.uin == "12345"  # int user_id 转 str
    assert content.name == "Alice"
    assert len(content.components) == 1
    first = content.components[0]
    assert isinstance(first, CompImage)
    assert first.url == "http://example.com/a.png"


def test_parse_forward_node_with_str_user_id() -> None:
    node = {
        "type": "node",
        "data": {
            "user_id": "10001",
            "nickname": "Bob",
            "content": [{"type": "image", "data": {"path": "/tmp/b.png"}}],
        },
    }
    content = _parse_forward_node(node)
    assert content is not None
    assert content.uin == "10001"
    assert content.name == "Bob"
    assert len(content.components) == 1


def test_parse_forward_node_missing_user_id_defaults_empty() -> None:
    node = {
        "data": {
            "content": [{"type": "image", "data": {"url": "http://example.com/c.png"}}],
        },
    }
    content = _parse_forward_node(node)
    assert content is not None
    assert content.uin == ""
    assert content.name == ""


def test_parse_forward_node_no_image_returns_none() -> None:
    node = {
        "type": "node",
        "data": {
            "user_id": "1",
            "nickname": "NoImg",
            "content": [{"type": "text", "data": {"text": "hi"}}],
        },
    }
    assert _parse_forward_node(node) is None


def test_parse_forward_node_non_dict_returns_none() -> None:
    assert _parse_forward_node("not a dict") is None
    assert _parse_forward_node(123) is None
    assert _parse_forward_node(["a"]) is None


def test_parse_forward_node_data_not_dict_returns_none() -> None:
    assert _parse_forward_node({"type": "node", "data": "oops"}) is None


def test_parse_forward_node_content_not_list_returns_none() -> None:
    assert _parse_forward_node({"type": "node", "data": {"content": "oops"}}) is None


def test_parse_forward_node_empty_content_returns_none() -> None:
    assert _parse_forward_node({"type": "node", "data": {"content": []}}) is None
