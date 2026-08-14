"""tests/integration 公共 fixtures。"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from _fakes import FakeCtx, make_plugin as _make_plugin
from koharu_client import KoharuClient
from main import KoharuMangaTranslatorPlugin

MakePlugin = Callable[..., KoharuMangaTranslatorPlugin]
"""make_plugin fixture 返回的工厂类型(测试函数参数注解用)。"""

KoharuClientFactory = Callable[[Callable[[httpx.Request], httpx.Response]], KoharuClient]
"""koharu_client_factory fixture 返回的工厂类型(测试函数参数注解用)。"""


@pytest.fixture
def make_plugin(tmp_path: Path) -> Callable[..., KoharuMangaTranslatorPlugin]:
    """插件工厂 fixture:每次调用返回一个全新、独立的插件实例。

    数据目录统一落在该测试的 tmp_path 下;context 默认是"无平台"的
    FakeCtx(触发 OneBot 读取会得到平台不支持错误)。
    """

    def factory(
        *,
        overrides: dict[str, object] | None = None,
        context: FakeCtx | None = None,
        data_dir: Path | None = None,
    ) -> KoharuMangaTranslatorPlugin:
        return _make_plugin(
            overrides=overrides,
            context=context,
            data_dir=data_dir if data_dir is not None else tmp_path / "koharu-data",
        )

    return factory


@pytest.fixture
def koharu_client_factory() -> Callable[[Callable[[httpx.Request], httpx.Response]], KoharuClient]:
    """KoharuClient 工厂:用 httpx.MockTransport 替换内部 AsyncClient,不访问网络。"""

    def factory(handler: Callable[[httpx.Request], httpx.Response]) -> KoharuClient:
        client = KoharuClient("http://koharu-test")
        transport = httpx.MockTransport(handler)
        # 替换内部 AsyncClient(私有属性注入经 setattr,规避 reportPrivateUsage)。
        setattr(client, "_client", httpx.AsyncClient(transport=transport, base_url=client.base_url))
        return client

    return factory
