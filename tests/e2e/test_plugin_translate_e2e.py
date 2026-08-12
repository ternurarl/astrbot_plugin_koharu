# pyright: reportPrivateUsage=false
# 说明:E2E 测试按任务要求通过 __new__ 注入配置并直接驱动插件私有实现
# (_data_dir/_translate_lock/_translate_images),无公开入口,故放宽 reportPrivateUsage。
"""插件级真实链路 E2E 测试。

覆盖:
- ``test_plugin_translate_images``:通过 __new__ 构造插件实例(config 用
  DEFAULT_CONFIG 拷贝 + _data_dir 指向 tmp_path),真实调用
  ``_translate_images`` 走完 Koharu pipeline,断言输出路径非空、文件存在且可被
  Pillow 打开。auto_load_llm 保持默认 False,复用 Koharu 当前已加载的 LLM。
- ``test_plugin_max_images_limit``:max_images_per_request=1 时传 2 张图,
  断言抛 ValueError(插件自身限制,不触网)。
"""

from __future__ import annotations

import asyncio
import copy
from pathlib import Path

import pytest
from PIL import Image

from koharu_client import KoharuClient
from main import DEFAULT_CONFIG, KoharuMangaTranslatorPlugin, PluginConfig


def _make_plugin_instance(tmp_path: Path, koharu_api_base: str) -> KoharuMangaTranslatorPlugin:
    """用 __new__ 构造插件实例并注入 config(默认配置拷贝)与 _data_dir。

    绕过 __init__(需要 AstrBot Context),手动补齐 _translate_images 依赖的
    _translate_lock。
    """
    config: PluginConfig = copy.deepcopy(DEFAULT_CONFIG)
    config["koharu_api_base_url"] = koharu_api_base
    plugin = KoharuMangaTranslatorPlugin.__new__(KoharuMangaTranslatorPlugin)
    plugin.config = config
    plugin._data_dir = tmp_path / "plugin_data"
    plugin._translate_lock = asyncio.Lock()
    return plugin


@pytest.mark.asyncio
async def test_plugin_translate_images(
    koharu_client: KoharuClient,
    koharu_api_base: str,
    test_images: list[Path],
    tmp_path: Path,
) -> None:
    """插件级真实翻译链路:2 张测试图经 Koharu pipeline 得到可打开的输出图片。"""
    plugin = _make_plugin_instance(tmp_path, koharu_api_base)
    image_paths = [str(path) for path in test_images]
    output_paths = await plugin._translate_images(image_paths, "Simplified Chinese")
    assert output_paths, "_translate_images 未返回任何输出路径"
    for path_str in output_paths:
        path = Path(path_str)
        assert path.is_file() and path.stat().st_size > 0, f"输出文件异常: {path}"
        with Image.open(path) as image:
            image.verify()


@pytest.mark.asyncio
async def test_plugin_max_images_limit(
    test_images: list[Path],
    tmp_path: Path,
    koharu_api_base: str,
) -> None:
    """max_images_per_request=1 且传入 2 张图时,插件自身抛 ValueError(不触网)。"""
    plugin = _make_plugin_instance(tmp_path, koharu_api_base)
    plugin.config["max_images_per_request"] = 1
    image_paths = [str(path) for path in test_images]
    with pytest.raises(ValueError, match="单次最多支持 1 张图片"):
        await plugin._translate_images(image_paths, "Simplified Chinese")
