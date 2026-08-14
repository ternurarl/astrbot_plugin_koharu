# pyright: reportPrivateUsage=false
"""插件配置读取逻辑单元测试。

通过 __new__ 构造插件实例(不调用 __init__,避免框架依赖),
手动注入 config 后验证 _raw_config_value/_str_conf/_int_conf/_float_conf/_bool_conf,
并校验 DEFAULT_CONFIG 与 _conf_schema.json、PluginConfig 声明的一致性。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast, get_origin, get_type_hints

import pytest

from main import DEFAULT_CONFIG, KoharuMangaTranslatorPlugin, PluginConfig

_SCHEMA_PATH = Path(__file__).resolve().parents[2] / "_conf_schema.json"


def _make_plugin(**overrides: str | int | float | bool) -> KoharuMangaTranslatorPlugin:
    plugin = KoharuMangaTranslatorPlugin.__new__(KoharuMangaTranslatorPlugin)
    # 与产品代码相同的边界单点 cast:外部 dict → PluginConfig。
    plugin.config = cast(PluginConfig, dict(overrides))
    return plugin


# --- 默认值回退 ----------------------------------------------------------------


def test_raw_config_value_falls_back_to_default() -> None:
    plugin = _make_plugin()
    assert plugin._raw_config_value("queue_depth") == 3
    assert plugin._raw_config_value("target_language") == "zh-CN"
    assert plugin._raw_config_value("llm_temperature") == -1.0
    assert plugin._raw_config_value("compress_return_images") is False


def test_str_conf_default_and_explicit() -> None:
    plugin = _make_plugin(target_language="日本語")
    assert plugin._str_conf("target_language") == "日本語"
    assert plugin._str_conf("koharu_api_base_url") == "http://koharu-headless:4000/api/v1"
    assert plugin._str_conf("llm_max_tokens") == "0"  # int 默认值转 str


# --- 显式覆盖与类型转换 ---------------------------------------------------------


def test_int_conf_accepts_string_number_and_int() -> None:
    plugin = _make_plugin(queue_depth="5", max_images_per_request=10)
    assert plugin._int_conf("queue_depth") == 5
    assert plugin._int_conf("max_images_per_request") == 10


def test_float_conf_accepts_string_number() -> None:
    plugin = _make_plugin(llm_temperature="0.5")
    assert plugin._float_conf("llm_temperature") == 0.5
    assert plugin._float_conf("llm_max_tokens") == 0.0  # int 默认值转 float


def test_bool_conf_true_variants() -> None:
    plugin = _make_plugin(
        compress_return_images="1",
        close_project_after_export="true",
        delete_project_after_export="是",
        auto_load_llm="TRUE",
    )
    assert plugin._bool_conf("compress_return_images") is True
    assert plugin._bool_conf("close_project_after_export") is True
    assert plugin._bool_conf("delete_project_after_export") is True
    assert plugin._bool_conf("auto_load_llm") is True


def test_bool_conf_false_variants_and_real_bool() -> None:
    plugin = _make_plugin(
        compress_return_images=False,
        auto_load_llm="0",
        close_project_after_export="no",
        delete_project_after_export="false",
    )
    assert plugin._bool_conf("compress_return_images") is False
    assert plugin._bool_conf("auto_load_llm") is False
    assert plugin._bool_conf("close_project_after_export") is False
    assert plugin._bool_conf("delete_project_after_export") is False


def test_bool_conf_real_bool_true() -> None:
    plugin = _make_plugin(compress_return_images=True)
    assert plugin._bool_conf("compress_return_images") is True


# --- 非法值回退默认 -------------------------------------------------------------


def test_int_conf_invalid_value_falls_back_to_default() -> None:
    plugin = _make_plugin(queue_depth="abc", max_send_images="12x")
    assert plugin._int_conf("queue_depth") == 3  # DEFAULT_CONFIG["queue_depth"]
    assert plugin._int_conf("max_send_images") == 0


def test_float_conf_invalid_value_falls_back_to_default() -> None:
    plugin = _make_plugin(llm_temperature="abc")
    assert plugin._float_conf("llm_temperature") == -1.0


# --- 未知键 --------------------------------------------------------------------


def test_raw_config_value_unknown_key_raises_key_error() -> None:
    plugin = _make_plugin()
    with pytest.raises(KeyError):
        plugin._raw_config_value("no_such_key")


def test_int_conf_unknown_key_raises_key_error() -> None:
    plugin = _make_plugin()
    with pytest.raises(KeyError):
        plugin._int_conf("no_such_key")


# --- DEFAULT_CONFIG 与 _conf_schema.json / PluginConfig 一致性 ------------------


def test_default_config_keys_match_conf_schema() -> None:
    with _SCHEMA_PATH.open(encoding="utf-8") as fh:
        schema_data: object = json.load(fh)
    assert isinstance(schema_data, dict)
    schema_keys: set[str] = set(cast(dict[str, object], schema_data).keys())
    assert set(DEFAULT_CONFIG.keys()) == schema_keys


def test_default_config_values_match_plugin_config_declared_types() -> None:
    hints = get_type_hints(PluginConfig)
    assert set(hints.keys()) == set(DEFAULT_CONFIG.keys())
    for key, value in DEFAULT_CONFIG.items():
        expected = hints[key]
        if expected is bool:
            assert isinstance(value, bool), f"{key}: 期望 bool,实际 {type(value)}"
        elif expected is int:
            assert isinstance(value, int) and not isinstance(value, bool), (
                f"{key}: 期望 int,实际 {type(value)}"
            )
        elif expected is str:
            assert isinstance(value, str), f"{key}: 期望 str,实际 {type(value)}"
        elif expected is float:
            assert isinstance(value, float), f"{key}: 期望 float,实际 {type(value)}"
        else:
            # 泛型注解（如 dict[str, str]）用 get_origin 取原类做 isinstance。
            assert isinstance(value, get_origin(expected) or expected), (
                f"{key}: 期望 {expected},实际 {type(value)}"
            )
