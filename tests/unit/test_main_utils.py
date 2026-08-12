# pyright: reportPrivateUsage=false
"""main.py 模块级纯函数单元测试。

覆盖:_dedupe_mapped / _image_from_path / _safe_path / _is_relative_to /
_contains_quote / _image_has_alpha / _prepare_jpeg_image / _compress_image。
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

from astrbot.api.message_components import (
    BaseMessageComponent,
    Forward,
    Image,
    Plain,
    Reply,
)
from PIL import Image as PILImage

import main


# --- _dedupe_mapped -------------------------------------------------------------


def test_dedupe_mapped_no_duplicates() -> None:
    unique, index_map = main._dedupe_mapped(["a.png", "b.png", "c.png"])
    assert unique == ["a.png", "b.png", "c.png"]
    assert index_map == {0: 0, 1: 1, 2: 2}


def test_dedupe_mapped_duplicates_at_start_middle_end() -> None:
    unique, index_map = main._dedupe_mapped(["a", "a", "b", "c", "a", "b", "b"])
    # 保序去重,重复位置映射到首次下标。
    assert unique == ["a", "b", "c"]
    assert index_map == {0: 0, 1: 0, 2: 1, 3: 2, 4: 0, 5: 1, 6: 1}
    assert len(index_map) == 7  # 每个原位置都有映射


def test_dedupe_mapped_all_duplicates() -> None:
    unique, index_map = main._dedupe_mapped(["x", "x", "x"])
    assert unique == ["x"]
    assert index_map == {0: 0, 1: 0, 2: 0}


def test_dedupe_mapped_empty() -> None:
    unique, index_map = main._dedupe_mapped([])
    assert unique == []
    assert index_map == {}


# --- _image_from_path -----------------------------------------------------------


def test_image_from_path_returns_file_url_image() -> None:
    image = main._image_from_path("/tmp/manga/page-1.png")
    assert isinstance(image, Image)
    assert image.file is not None
    assert image.file.startswith("file:///")


# --- _safe_path -----------------------------------------------------------------


def test_safe_path_normal_relative_and_absolute() -> None:
    assert main._safe_path("a/b/c.png") == str(Path("a/b/c.png"))
    assert main._safe_path("/abs/path.png") == "/abs/path.png"


class _FspathFailureObject:
    """Path() 转换会抛错、但 str() 可用的对象,用于触发 _safe_path 的异常分支。"""

    def __fspath__(self) -> str:
        raise TypeError("not a filesystem path")

    def __str__(self) -> str:
        return "fallback-string"


def test_safe_path_fspath_failure_falls_back_to_str() -> None:
    assert main._safe_path(cast(str, _FspathFailureObject())) == "fallback-string"


# --- _is_relative_to ------------------------------------------------------------


def test_is_relative_to() -> None:
    base = Path("/data/outputs")
    assert main._is_relative_to(Path("/data/outputs/2024/x.png"), base)
    assert main._is_relative_to(base, base)  # 相同路径
    assert not main._is_relative_to(Path("/other/x.png"), base)  # 不相对
    assert not main._is_relative_to(Path("/data/outputs-extra/x.png"), base)  # 仅前缀
    assert not main._is_relative_to(Path("relative/x.png"), base)  # 相对路径输入


# --- _contains_quote ------------------------------------------------------------


def test_contains_quote_with_reply() -> None:
    messages: list[BaseMessageComponent] = [Reply(id=1)]
    assert main._contains_quote(messages)


def test_contains_quote_with_forward() -> None:
    messages: list[BaseMessageComponent] = [Forward(id="fwd-1")]
    assert main._contains_quote(messages)


def test_contains_quote_with_both() -> None:
    messages: list[BaseMessageComponent] = [Reply(id=1), Forward(id="fwd-1")]
    assert main._contains_quote(messages)


def test_contains_quote_plain_image_only() -> None:
    messages: list[BaseMessageComponent] = [Image(file="a.png"), Plain(text="hi")]
    assert not main._contains_quote(messages)


def test_contains_quote_empty() -> None:
    assert not main._contains_quote([])


# --- _image_has_alpha -----------------------------------------------------------


def test_image_has_alpha_rgba() -> None:
    image = PILImage.new("RGBA", (4, 4), (0, 0, 0, 0))
    assert main._image_has_alpha(image)


def test_image_has_alpha_la() -> None:
    image = PILImage.new("LA", (4, 4))
    assert main._image_has_alpha(image)


def test_image_has_alpha_palette_with_transparency() -> None:
    image = PILImage.new("P", (4, 4))
    image.info["transparency"] = 0
    assert main._image_has_alpha(image)


def test_image_has_alpha_palette_without_transparency() -> None:
    image = PILImage.new("P", (4, 4))
    assert not main._image_has_alpha(image)


def test_image_has_alpha_rgb() -> None:
    image = PILImage.new("RGB", (4, 4), (255, 0, 0))
    assert not main._image_has_alpha(image)


# --- _prepare_jpeg_image --------------------------------------------------------


def test_prepare_jpeg_image_no_alpha_converts_to_rgb() -> None:
    image = PILImage.new("RGB", (6, 6), (10, 20, 30))
    result = main._prepare_jpeg_image(image)
    assert result.mode == "RGB"
    assert result.size == (6, 6)


def test_prepare_jpeg_image_rgba_composites_white_background() -> None:
    image = PILImage.new("RGBA", (4, 4), (0, 0, 0, 0))
    image.putpixel((1, 1), (255, 0, 0, 255))
    result = main._prepare_jpeg_image(image)
    assert result.mode == "RGB"
    assert result.size == (4, 4)
    assert result.getpixel((0, 0)) == (255, 255, 255)  # 透明区域合成白色
    assert result.getpixel((1, 1)) == (255, 0, 0)  # 不透明像素保留


# --- _compress_image ------------------------------------------------------------


def test_compress_image_jpeg(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    target = tmp_path / "out.jpg"
    image = PILImage.new("RGB", (16, 16), (120, 60, 30))
    image.save(source, "PNG")
    main._compress_image(source, target, "JPEG", 85)
    assert target.exists()
    assert source.exists()  # 实现不删除源文件
    with PILImage.open(target) as reopened:
        assert reopened.format == "JPEG"
        assert reopened.size == (16, 16)


def test_compress_image_webp_keeps_rgba(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    target = tmp_path / "out.webp"
    image = PILImage.new("RGBA", (12, 12), (10, 200, 30, 128))
    image.save(source, "PNG")
    main._compress_image(source, target, "WEBP", 80)
    assert target.exists()
    with PILImage.open(target) as reopened:
        assert reopened.format == "WEBP"
        assert reopened.size == (12, 12)
        assert reopened.mode == "RGBA"


def test_compress_image_webp_grayscale_converted_to_rgb(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    target = tmp_path / "out.webp"
    image = PILImage.new("L", (8, 8), 128)
    image.save(source, "PNG")
    main._compress_image(source, target, "WEBP", 75)
    assert target.exists()
    with PILImage.open(target) as reopened:
        assert reopened.format == "WEBP"
        assert reopened.mode == "RGB"
