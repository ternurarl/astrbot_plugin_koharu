# pyright: reportPrivateUsage=false
"""koharu_client.py 模块级函数单元测试。

覆盖:_normalize_operation_list / _find_operation / _parse_event_data /
_suffix_from_content_type / save_exported_images / _save_images_from_zip。
全部为纯逻辑,不发起任何网络请求。
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import cast

import pytest
from PIL import Image as PILImage

from koharu_client import (
    JsonValue,
    KoharuApiError,
    OperationInfo,
    _find_operation,
    _normalize_operation_list,
    _parse_event_data,
    _save_images_from_zip,
    _suffix_from_content_type,
    save_exported_images,
)


# --- _normalize_operation_list --------------------------------------------------


def test_normalize_operation_list_list_input_filters_non_dict() -> None:
    data: JsonValue = [
        {"id": "op-1"},
        {"id": "op-2", "status": "running"},
        "junk",
        42,
        None,
        True,
    ]
    result = _normalize_operation_list(data)
    assert result == [{"id": "op-1"}, {"id": "op-2", "status": "running"}]


def test_normalize_operation_list_dict_with_operations_key() -> None:
    data: JsonValue = {"operations": [{"id": "a"}, {"id": "b"}, "x", 7]}
    result = _normalize_operation_list(data)
    assert len(result) == 2
    assert result[0].get("id") == "a"
    assert result[1].get("id") == "b"


def test_normalize_operation_list_dict_with_jobs_and_items_keys() -> None:
    jobs: JsonValue = {"jobs": [{"jobId": "j1"}]}
    assert _normalize_operation_list(jobs) == [{"jobId": "j1"}]
    items: JsonValue = {"items": [{"operationId": "o1"}]}
    assert _normalize_operation_list(items) == [{"operationId": "o1"}]


def test_normalize_operation_list_dict_other_shape_takes_dict_values() -> None:
    data: JsonValue = {"alpha": {"id": "x"}, "beta": {"jobId": "y"}, "gamma": 3, "delta": "no"}
    assert _normalize_operation_list(data) == [{"id": "x"}, {"jobId": "y"}]


def test_normalize_operation_list_non_list_dict_returns_empty() -> None:
    assert _normalize_operation_list("hello") == []
    assert _normalize_operation_list(42) == []
    assert _normalize_operation_list(1.5) == []
    assert _normalize_operation_list(None) == []


# --- _find_operation ------------------------------------------------------------


def test_find_operation_by_id() -> None:
    operations: list[OperationInfo] = [{"id": "op-1", "status": "running"}]
    found = _find_operation(operations, "op-1")
    assert found is not None
    assert found is operations[0]


def test_find_operation_by_operation_id_and_job_id() -> None:
    operations: list[OperationInfo] = [
        {"operationId": "op-2", "status": "done"},
        {"jobId": "job-9"},
    ]
    assert _find_operation(operations, "op-2") is operations[0]
    assert _find_operation(operations, "job-9") is operations[1]


def test_find_operation_matches_int_id_via_str_conversion() -> None:
    # 运行时 id 可能是 int,实现通过 str() 统一比较。
    operations = cast(list[OperationInfo], [{"id": 12345}])
    found = _find_operation(operations, "12345")
    assert found is not None
    assert found is operations[0]


def test_find_operation_not_found_and_empty() -> None:
    operations: list[OperationInfo] = [{"id": "op-1"}]
    assert _find_operation(operations, "missing") is None
    assert _find_operation([], "op-1") is None


# --- _parse_event_data ----------------------------------------------------------


def test_parse_event_data_valid_json() -> None:
    result = _parse_event_data('{"status": "loaded", "n": 3}')
    assert isinstance(result, dict)
    assert result["status"] == "loaded"
    assert result["n"] == 3


def test_parse_event_data_invalid_json_returns_raw() -> None:
    raw = "{not json at all"
    assert _parse_event_data(raw) == raw


def test_parse_event_data_empty_string() -> None:
    assert _parse_event_data("") == ""


# --- _suffix_from_content_type --------------------------------------------------


def test_suffix_from_content_type_image_types() -> None:
    assert _suffix_from_content_type("image/jpeg") == ".jpg"
    assert _suffix_from_content_type("image/png") == ".png"
    assert _suffix_from_content_type("image/webp") == ".webp"


def test_suffix_from_content_type_with_charset_and_case() -> None:
    assert _suffix_from_content_type("image/jpeg; charset=utf-8") == ".jpg"
    assert _suffix_from_content_type("IMAGE/PNG; charset=binary") == ".png"


def test_suffix_from_content_type_non_image_returns_none() -> None:
    assert _suffix_from_content_type("text/plain") is None
    assert _suffix_from_content_type("application/octet-stream") is None
    assert _suffix_from_content_type("") is None


# --- save_exported_images / _save_images_from_zip -------------------------------


def _png_bytes() -> bytes:
    buffer = io.BytesIO()
    PILImage.new("RGB", (2, 2), (10, 20, 30)).save(buffer, "PNG")
    return buffer.getvalue()


def _zip_bytes(entries: list[tuple[str, bytes]]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in entries:
            archive.writestr(name, data)
    return buffer.getvalue()


def test_save_exported_images_single_png(tmp_path: Path) -> None:
    content = _png_bytes()
    saved = save_exported_images(content, "image/png", tmp_path)
    assert len(saved) == 1
    target = Path(saved[0])
    assert target == tmp_path / "translated.png"
    assert target.read_bytes() == content


def test_save_exported_images_single_jpeg_extension(tmp_path: Path) -> None:
    saved = save_exported_images(_png_bytes(), "image/jpeg", tmp_path)
    assert Path(saved[0]).name == "translated.jpg"


def test_save_exported_images_unknown_content_type_defaults_png(tmp_path: Path) -> None:
    saved = save_exported_images(_png_bytes(), "application/octet-stream", tmp_path)
    assert Path(saved[0]).name == "translated.png"


def test_save_exported_images_zip_extracts_images_ordered(tmp_path: Path) -> None:
    png = _png_bytes()
    jpg = _png_bytes()
    content = _zip_bytes(
        [
            ("page-2.png", png),
            ("notes.txt", b"hello"),
            ("page-1.jpg", jpg),
            ("empty-dir/", b""),
        ]
    )
    saved = save_exported_images(content, "application/zip", tmp_path)
    assert len(saved) == 2
    # 编号按 zip 内成员位置(001 起),非图片与目录被过滤。
    names = [Path(path).name for path in saved]
    assert names == ["001-page-2.png", "003-page-1.jpg"]
    assert Path(saved[0]).read_bytes() == png
    assert Path(saved[1]).read_bytes() == jpg
    # 实现会保留原始 zip 归档文件。
    assert (tmp_path / "koharu-rendered.zip").exists()


def test_save_exported_images_zip_without_images_raises(tmp_path: Path) -> None:
    content = _zip_bytes([("readme.txt", b"no images here")])
    with pytest.raises(KoharuApiError):
        save_exported_images(content, "application/zip", tmp_path)


def test_save_exported_images_empty_zip_raises(tmp_path: Path) -> None:
    with pytest.raises(KoharuApiError):
        save_exported_images(_zip_bytes([]), "application/zip", tmp_path)


def test_save_exported_images_zip_magic_detected_without_zip_content_type(
    tmp_path: Path,
) -> None:
    # 即使 content_type 不含 "zip",PK\x03\x04 魔数也应走 zip 分支。
    content = _zip_bytes([("page-1.png", _png_bytes())])
    saved = save_exported_images(content, "application/octet-stream", tmp_path)
    assert len(saved) == 1
    assert Path(saved[0]).name == "001-page-1.png"


def test_save_images_from_zip_filters_non_images_and_dirs(tmp_path: Path) -> None:
    png = _png_bytes()
    content = _zip_bytes(
        [
            ("000-cover.png", png),
            ("a.txt", b"text"),
            ("folder/", b""),
            ("nested/x.bmp", png),
            ("noext", b"whatever"),
        ]
    )
    saved = _save_images_from_zip(content, tmp_path)
    assert len(saved) == 2
    names = [Path(path).name for path in saved]
    assert names == ["001-000-cover.png", "004-x.bmp"]
