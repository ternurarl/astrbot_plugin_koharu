# pyright: reportPrivateUsage=false
"""Koharu 0.66 持久化配置纯函数单元测试。

覆盖:normalize_language / display_language / _float_or_none / _int_or_none /
build_expected_config / config_differs。全部为纯逻辑,不发起任何网络请求。
"""

from __future__ import annotations

import copy
from typing import cast

from main import (
    DEFAULT_CONFIG,
    PluginConfig,
    _float_or_none,
    _int_or_none,
    build_expected_config,
    config_differs,
    display_language,
    normalize_language,
)
from koharu_client import AppConfig


def _default_cfg(**overrides: object) -> PluginConfig:
    """DEFAULT_CONFIG 深拷贝 + 覆盖项（新键自动补全）。"""
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    for key, value in overrides.items():
        cfg[key] = value  # type: ignore[literal]
    return cfg


def _server_config() -> AppConfig:
    """模拟 GET /config 的 0.66 全量响应。"""
    return {
        "pipeline": {
            "detection": {"model": "koharu-layout-rfdetr-seg-2xl"},
            "ocr": {"model": "baberu-ocr"},
            "inpainting": {"model": "lama"},
            "translation": {
                "model": {
                    "provider": "deepseek",
                    "model": "deepseek-v4-flash",
                    "quantization": None,
                    "vision": False,
                },
                "generation": {
                    "temperature": None,
                    "max_tokens": None,
                },
                "target_language": "en-US",
                "instructions": None,
            },
            "processor": {
                "koharu-layout-rfdetr-seg-2xl": {
                    "text_threshold": 0.5,
                },
                "flux2-klein": None,
                "rorem-mixed": None,
            },
        },
        "providers": {
            "openai-compatible": {"base_url": None, "vision": False},
            "lm-studio": {"base_url": "http://localhost:1234"},
        },
        "typesetting": {"font_families": ["CCWildWords", "Adobe 黑体 Std"]},
    }


# --- normalize_language / display_language -------------------------------------


def test_normalize_language_canonical_tags() -> None:
    assert normalize_language("zh-CN") == "zh-CN"
    assert normalize_language("en-US") == "en-US"
    assert normalize_language("ja-JP") == "ja-JP"


def test_normalize_language_aliases_case_insensitive() -> None:
    assert normalize_language("zh") == "zh-CN"
    assert normalize_language("zh-hans") == "zh-CN"
    assert normalize_language("ZH-TW") == "zh-TW"
    assert normalize_language("tl") == "fil-PH"


def test_normalize_language_display_names() -> None:
    assert normalize_language("Simplified Chinese") == "zh-CN"
    assert normalize_language("Traditional Chinese") == "zh-TW"
    assert normalize_language("English") == "en-US"
    assert normalize_language("Japanese") == "ja-JP"


def test_normalize_language_unknown_returns_none() -> None:
    assert normalize_language("Klingon") is None
    assert normalize_language("") is None
    assert normalize_language("   ") is None


def test_display_language_zh_and_unknown() -> None:
    assert display_language("zh-CN") == "简体中文"
    assert display_language("Simplified Chinese") == "简体中文"
    assert display_language("zh-TW") == "繁体中文"
    assert display_language("en-US") == "English"
    assert display_language("日本語") == "日本語"


# --- _float_or_none / _int_or_none --------------------------------------------


def test_float_or_none_semantics() -> None:
    assert _float_or_none(-1.0) is None  # -1 = 不覆盖
    assert _float_or_none(0.0) is None  # 0（WebUI 清空 float 被转成 0.0）= 不覆盖
    assert _float_or_none(0.5) == 0.5
    assert _float_or_none("0.5") is None  # 字符串由 _cfg_float 解析，这里不算数
    assert _float_or_none(True) is None


def test_int_or_none_semantics() -> None:
    assert _int_or_none(0) is None  # 0 = 不覆盖
    assert _int_or_none(512) == 512
    assert _int_or_none(-3) is None
    assert _int_or_none(False) is None


# --- build_expected_config -----------------------------------------------------


def test_build_expected_defaults_do_not_override() -> None:
    """除默认 target_language=zh-CN 覆盖外,其余字段留空/默认时保持服务端现有值。"""
    current = _server_config()
    expected = build_expected_config(current, _default_cfg())
    # 默认 target_language=zh-CN 是设计行为:把服务端语言强制为简体中文。
    assert expected["pipeline"]["translation"]["target_language"] == "zh-CN"  # type: ignore[index]
    assert expected["pipeline"]["translation"]["model"]["provider"] == "deepseek"  # type: ignore[index]
    assert expected["pipeline"]["ocr"]["model"] == "baberu-ocr"  # type: ignore[index]
    assert expected["pipeline"]["processor"] == current["pipeline"]["processor"]
    assert expected["providers"] == current["providers"]
    assert expected["typesetting"] == current["typesetting"]
    assert config_differs(current, expected) == {"pipeline"}


def test_build_expected_ocr_and_inpainting_models() -> None:
    cfg = _default_cfg(pipeline_ocr_model="manga-ocr", pipeline_inpainting_model="flux2-klein")
    expected = build_expected_config(_server_config(), cfg)
    assert expected["pipeline"]["ocr"]["model"] == "manga-ocr"  # type: ignore[index]
    assert expected["pipeline"]["inpainting"]["model"] == "flux2-klein"  # type: ignore[index]
    # 未配置的字段保持服务端现有值。
    assert expected["pipeline"]["translation"]["model"]["provider"] == "deepseek"  # type: ignore[index]
    assert config_differs(_server_config(), expected) == {"pipeline"}


def test_build_expected_processor_preserves_existing_keys() -> None:
    """processor 覆盖时保留现有键（整段替换语义下必须全量）。"""
    cfg = _default_cfg(
        pipeline_inpainting_model="rorem-mixed",
        pipeline_inpainting_prompt="clean background",
        pipeline_inpainting_negative_prompt="no text",
        pipeline_detection_bubble_threshold=0.8,
    )
    expected = build_expected_config(_server_config(), cfg)
    processor = expected["pipeline"]["processor"]
    assert processor["koharu-layout-rfdetr-seg-2xl"]["text_threshold"] == 0.5  # type: ignore[index]
    assert processor["koharu-layout-rfdetr-seg-2xl"]["bubble_threshold"] == 0.8  # type: ignore[index]
    assert processor["rorem-mixed"] == {  # type: ignore[index]
        "prompt": "clean background",
        "negative_prompt": "no text",
    }
    # prompt 同时写入 flux2-klein（用户可切换修补模型），保留现有 None 之外的键。
    assert processor["flux2-klein"] == {"prompt": "clean background"}  # type: ignore[index]


def test_build_expected_translation_model_and_language() -> None:
    cfg = _default_cfg(
        translation_provider="openai",
        translation_model="gpt-4o",
        translation_quantization="",
        translation_vision=True,
        target_language="Simplified Chinese",
        system_prompt="translate carefully",
        llm_temperature=0.7,
        llm_max_tokens=1024,
    )
    expected = build_expected_config(_server_config(), cfg)
    translation = expected["pipeline"]["translation"]
    assert translation["model"] == {
        "provider": "openai",
        "model": "gpt-4o",
        "quantization": None,
        "vision": True,
    }
    # 旧文案映射为 BCP47。
    assert translation["target_language"] == "zh-CN"
    assert translation["instructions"] == "translate carefully"
    assert translation["generation"]["temperature"] == 0.7  # type: ignore[index]
    assert translation["generation"]["max_tokens"] == 1024  # type: ignore[index]


def test_build_expected_generation_defaults_keep_server_values() -> None:
    """llm_temperature=-1 / llm_max_tokens=0 时不覆盖 generation 字段。"""
    current = _server_config()
    cfg = _default_cfg(translation_provider="deepseek", translation_model="deepseek-v4-flash")
    expected = build_expected_config(current, cfg)
    generation = expected["pipeline"]["translation"]["generation"]
    assert generation["temperature"] is None  # type: ignore[index]
    assert generation["max_tokens"] is None  # type: ignore[index]


def test_build_expected_unrecognized_language_skips_override() -> None:
    """无法识别的语言值不覆盖 target_language（避免 PATCH 422）。"""
    current = _server_config()
    cfg = _default_cfg(target_language="日本語")
    expected = build_expected_config(current, cfg)
    assert expected["pipeline"]["translation"]["target_language"] == "en-US"  # type: ignore[index]
    assert config_differs(current, expected) == set()


def test_build_expected_providers_settings() -> None:
    cfg = _default_cfg(
        target_language="en-US",  # 抵消默认 zh-CN 覆盖，让 pipeline 无差异
        openai_compatible_base_url="http://my-llm:8000/v1",
        openai_compatible_vision=True,
        lm_studio_base_url="http://studio:1234",
    )
    expected = build_expected_config(_server_config(), cfg)
    providers = expected["providers"]
    assert providers["openai-compatible"] == {  # type: ignore[index]
        "base_url": "http://my-llm:8000/v1",
        "vision": True,
    }
    assert providers["lm-studio"]["base_url"] == "http://studio:1234"  # type: ignore[index]
    # 填 base_url 会同时把翻译 provider 切到 openai-compatible（三件套设计行为）→ pipeline 也 diff。
    assert config_differs(_server_config(), expected) == {"pipeline", "providers"}


def test_build_expected_typesetting_fonts() -> None:
    cfg = _default_cfg(font_families="Noto Sans SC,CCWildWords")
    expected = build_expected_config(_server_config(), cfg)
    assert expected["typesetting"]["font_families"] == ["Noto Sans SC", "CCWildWords"]


def test_build_expected_empty_fonts_keeps_server() -> None:
    cfg = _default_cfg(font_families="")
    expected = build_expected_config(_server_config(), cfg)
    assert expected["typesetting"]["font_families"] == ["CCWildWords", "Adobe 黑体 Std"]


# --- config_differs ------------------------------------------------------------


def test_config_differs_sections() -> None:
    current = _server_config()
    same = copy.deepcopy(current)
    assert config_differs(current, same) == set()

    pipeline_only = copy.deepcopy(current)
    pipeline_only["pipeline"]["ocr"]["model"] = "manga-ocr"  # type: ignore[index]
    assert config_differs(current, pipeline_only) == {"pipeline"}

    typesetting_only = copy.deepcopy(current)
    typesetting_only["typesetting"] = {"font_families": ["X"]}
    assert config_differs(current, typesetting_only) == {"typesetting"}


# --- 对抗性审核补充用例（None 防御 / 非法输入 / 钳制） ------------------------


def test_build_expected_none_values_do_not_crash() -> None:
    """配置值全部为 None（旧配置缺键/值为 null）时不崩溃且不覆盖。"""
    cfg = cast(PluginConfig, {key: None for key in DEFAULT_CONFIG})
    current = _server_config()
    expected = build_expected_config(current, cfg)
    # 除默认声明值（target_language 等被 None 回落为默认 zh-CN）外保持服务端值。
    assert expected["pipeline"]["ocr"]["model"] == "baberu-ocr"  # type: ignore[index]
    assert expected["pipeline"]["translation"]["model"]["provider"] == "deepseek"  # type: ignore[index]


def test_build_expected_string_numbers_parsed() -> None:
    """数字字符串配置（手改配置文件场景）被解析，而不是静默忽略。"""
    cfg = _default_cfg(
        llm_temperature="0.7",
        llm_max_tokens="512",
        pipeline_detection_text_threshold="0.6",
        target_language="en-US",
    )
    expected = build_expected_config(_server_config(), cfg)
    translation = expected["pipeline"]["translation"]
    assert translation["generation"]["temperature"] == 0.7  # type: ignore[index]
    assert translation["generation"]["max_tokens"] == 512  # type: ignore[index]
    processor = expected["pipeline"]["processor"]
    assert processor["koharu-layout-rfdetr-seg-2xl"]["text_threshold"] == 0.6  # type: ignore[index]


def test_build_expected_bool_strings_parsed_correctly() -> None:
    """字符串 "false" 不能被 bool() 强转成 True。"""
    cfg = _default_cfg(
        target_language="en-US",
        translation_vision="false",
        openai_compatible_vision="0",
    )
    expected = build_expected_config(_server_config(), cfg)
    assert expected["pipeline"]["translation"]["model"]["vision"] is False  # type: ignore[index]
    assert expected["providers"]["openai-compatible"]["vision"] is False  # type: ignore[index]


def test_build_expected_provider_empty_or_unknown_skips_model() -> None:
    """provider 为空/不在白名单时不重建 translation.model（避免 provider:"" 422）。"""
    current = _server_config()
    empty = _default_cfg(translation_provider="", translation_model="gpt-4o", target_language="en-US")
    expected = build_expected_config(current, empty)
    # provider 空 → 保留服务端现有 model。
    assert expected["pipeline"]["translation"]["model"]["provider"] == "deepseek"  # type: ignore[index]
    unknown = _default_cfg(translation_provider="deepseek ", translation_model="gpt-4o", target_language="en-US")
    expected2 = build_expected_config(current, unknown)
    assert expected2["pipeline"]["translation"]["model"]["provider"] == "deepseek"  # type: ignore[index]


def test_build_expected_threshold_clamped_to_unit_range() -> None:
    """阈值钳制到 0..=1（服务端校验在运行时，超范围会延迟到翻译时报错）。"""
    cfg = _default_cfg(
        target_language="en-US",
        pipeline_detection_text_threshold=1.5,
        pipeline_detection_bubble_threshold=0.3,
    )
    expected = build_expected_config(_server_config(), cfg)
    processor = expected["pipeline"]["processor"]
    assert processor["koharu-layout-rfdetr-seg-2xl"]["text_threshold"] == 1.0  # type: ignore[index]
    assert processor["koharu-layout-rfdetr-seg-2xl"]["bubble_threshold"] == 0.3  # type: ignore[index]


def test_build_expected_generation_clamped() -> None:
    """temperature 钳制 0..=2，max_tokens 钳制 u32 上限（超限 422）。"""
    cfg = _default_cfg(
        target_language="en-US",
        llm_temperature=99.0,
        llm_max_tokens=99999999999,
    )
    expected = build_expected_config(_server_config(), cfg)
    generation = expected["pipeline"]["translation"]["generation"]
    assert generation["temperature"] == 2.0  # type: ignore[index]
    assert generation["max_tokens"] == 4294967295  # type: ignore[index]


def test_build_expected_invalid_base_url_skipped() -> None:
    """非法 base_url（无 scheme）跳过覆盖，不把整 section 拖进 422。"""
    cfg = _default_cfg(
        target_language="en-US",
        openai_compatible_base_url="not-a-url",
        lm_studio_base_url="http://studio:1234",
    )
    expected = build_expected_config(_server_config(), cfg)
    providers = expected["providers"]
    assert providers["openai-compatible"].get("base_url") is None  # type: ignore[index]
    assert providers["lm-studio"]["base_url"] == "http://studio:1234"  # type: ignore[index]


def test_build_expected_typesetting_merges_existing_keys() -> None:
    """typesetting 覆盖 font_families 时保留服务端现有其他键。"""
    current = _server_config()
    current["typesetting"] = {"font_families": ["X"], "vertical_text": True}
    cfg = _default_cfg(target_language="en-US", font_families="Noto Sans SC")
    expected = build_expected_config(current, cfg)
    assert expected["typesetting"] == {
        "font_families": ["Noto Sans SC"],
        "vertical_text": True,
    }


def test_config_differs_missing_section_treated_as_empty() -> None:
    """服务端缺失 section（None）与期望的空 dict 视为相等，避免空 section PATCH。"""
    current: AppConfig = {}
    expected: AppConfig = {"typesetting": {}, "providers": {}}
    assert config_differs(current, expected) == set()


# --- 自定义端点三件套（base_url + key + model）---------------------------------


def test_build_expected_compatible_trio_switches_provider() -> None:
    """填了 openai-compatible 的 base_url 即自动把翻译提供商切到 openai-compatible。"""
    cfg = _default_cfg(
        target_language="en-US",
        openai_compatible_base_url="https://my-llm.example.com/v1",
        translation_model="my-model",
    )
    expected = build_expected_config(_server_config(), cfg)
    model = expected["pipeline"]["translation"]["model"]
    assert model["provider"] == "openai-compatible"  # type: ignore[index]
    assert model["model"] == "my-model"  # type: ignore[index]
    assert expected["providers"]["openai-compatible"]["base_url"] == (  # type: ignore[index]
        "https://my-llm.example.com/v1"
    )


def test_build_expected_compatible_api_key_alone_switches_provider() -> None:
    """只填 api_key 也强制切换 provider（key 需要 PUT 到 openai-compatible keyring）。"""
    cfg = _default_cfg(
        target_language="en-US",
        openai_compatible_api_key="sk-test",
        translation_model="my-model",
    )
    expected = build_expected_config(_server_config(), cfg)
    model = expected["pipeline"]["translation"]["model"]
    assert model["provider"] == "openai-compatible"  # type: ignore[index]


def test_build_expected_explicit_provider_wins_without_trio() -> None:
    """未填端点/key 时显式 translation_provider 生效（不被三件套覆盖）。"""
    cfg = _default_cfg(
        target_language="en-US",
        translation_provider="deepseek",
        translation_model="deepseek-v4-pro",
    )
    expected = build_expected_config(_server_config(), cfg)
    model = expected["pipeline"]["translation"]["model"]
    assert model["provider"] == "deepseek"  # type: ignore[index]
    assert model["model"] == "deepseek-v4-pro"  # type: ignore[index]
