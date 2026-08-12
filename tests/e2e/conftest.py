"""E2E 测试共享设施:Koharu 连通性守卫、客户端 fixture、测试图片生成。

Koharu HTTP API 地址默认 ``http://127.0.0.1:4000/api/v1``,可用环境变量
``KOHARU_API_BASE`` 覆盖。所有依赖真实 Koharu 实例的用例通过
``koharu_client`` fixture 获取客户端;实例不可达时该 fixture 直接 skip,
保证本目录在无 Koharu 环境下也能收集并通过(除 skip 外无失败)。
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
import pytest_asyncio
from PIL import Image, ImageDraw, ImageFont

from koharu_client import KoharuClient

DEFAULT_API_BASE = "http://127.0.0.1:4000/api/v1"

TEST_IMAGE_WIDTH = 480
TEST_IMAGE_HEIGHT = 680


def _normalize_api_base(raw: str) -> str:
    """与 KoharuClient 相同的 base_url 规范化(补全 /api/v1 后缀)。"""
    normalized = raw.rstrip("/")
    if not normalized.endswith("/api/v1"):
        normalized = f"{normalized}/api/v1"
    return normalized


@pytest.fixture(scope="session")
def koharu_api_base() -> str:
    """Koharu API base URL(环境变量 KOHARU_API_BASE 优先,否则默认值)。"""
    return _normalize_api_base(os.environ.get("KOHARU_API_BASE", DEFAULT_API_BASE))


@pytest.fixture(scope="session")
def koharu_available(koharu_api_base: str) -> bool:
    """探测 GET /meta(短超时 5s);返回 False 表示 Koharu 实例不可用。"""
    try:
        with httpx.Client(timeout=5.0) as probe:
            response = probe.get(f"{koharu_api_base}/meta")
        return response.status_code == 200
    except httpx.HTTPError:
        return False


@pytest_asyncio.fixture
async def koharu_client(
    koharu_api_base: str,
    koharu_available: bool,
) -> AsyncIterator[KoharuClient]:
    """真实 Koharu 客户端;实例未运行/未就绪时 skip 整个用例。"""
    if not koharu_available:
        pytest.skip(
            f"Koharu 实例未运行 (KOHARU_API_BASE={koharu_api_base}),跳过 E2E"
        )
    client = KoharuClient(koharu_api_base, timeout=300.0, connect_timeout=5.0)
    try:
        await client.wait_until_ready(timeout_seconds=10.0, interval_seconds=0.5)
    except Exception:
        await client.aclose()
        pytest.skip(
            f"Koharu 实例未就绪 (KOHARU_API_BASE={koharu_api_base}),跳过 E2E"
        )
    try:
        yield client
    finally:
        await client.aclose()


@pytest.fixture
def test_images(tmp_path: Path) -> list[Path]:
    """生成 2 张真实 PNG 测试图(浅色底 + 深色矩形 + 英文文字)。"""
    return make_test_images(tmp_path / "pages", 2)


def make_test_images(output_dir: Path, count: int = 2) -> list[Path]:
    """在 output_dir 下生成 count 张有效 PNG 测试图并返回路径列表。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index in range(count):
        path = output_dir / f"test_page_{index + 1}.png"
        _draw_test_page(path, index)
        paths.append(path)
    return paths


def _draw_test_page(path: Path, index: int) -> None:
    """绘制一张测试页:浅色底,深色矩形内白字,另有一行深色文字。"""
    image = Image.new(
        "RGB",
        (TEST_IMAGE_WIDTH, TEST_IMAGE_HEIGHT),
        (246, 240, 230),
    )
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=36)
    top = 100 + index * 50
    draw.rectangle((60, top, 420, top + 90), fill=(30, 30, 30))
    draw.text((80, top + 10), f"Hello World {index + 1}", fill=(255, 255, 255), font=font)
    draw.text((80, 430), "This is a test page.", fill=(40, 40, 40), font=font)
    image.save(path, "PNG")
