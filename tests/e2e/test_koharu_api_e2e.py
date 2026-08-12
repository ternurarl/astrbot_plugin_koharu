"""Koharu HTTP API 全生命周期 E2E 测试(真实实例)。

覆盖:
- ``test_api_full_lifecycle``:ready → create project → create pages → 解析 pipeline
  steps → start pipeline(真实 OCR+LLM+inpainting)→ wait_operation → export → 保存
  输出图片并验证可被 Pillow 打开。pipeline 失败时如实抛出 KoharuApiError(含错误详情)。
- ``test_config_pipeline_steps``:GET /config 的 pipeline 段与契约 PipelineConfig
  对应,各引擎键齐全且非空。

所有用例在 finally 中清理 close_project() + delete_project()。
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
from PIL import Image

from koharu_client import KoharuClient, save_exported_images

OPERATION_FINISHED_STATUSES = frozenset(
    {"finished", "completed", "complete", "succeeded", "success", "done"}
)

DEFAULT_TARGET_LANGUAGE = "Simplified Chinese"


@pytest.mark.asyncio
async def test_api_full_lifecycle(
    koharu_client: KoharuClient,
    test_images: list[Path],
    tmp_path: Path,
) -> None:
    """完整走一遍 Koharu API 生命周期并验证渲染输出。"""
    client = koharu_client
    project_id: str | None = None
    try:
        meta = await client.get_meta()
        assert meta.get("version"), f"GET /meta 未返回 version: {meta!r}"

        # 先关闭可能残留的 current project,再创建新项目(与插件逻辑一致)
        await client.close_project_if_any()

        project_name = f"e2e-{int(time.time())}"
        project = await client.create_project(project_name)
        project_id = str(
            project.get("id")
            or project.get("projectId")
            or project.get("project_id")
            or ""
        )
        assert project_id, f"create_project 未返回 project id: {project!r}"

        created = await client.create_pages(test_images, replace=True)
        assert len(created.get("pages", [])) == len(test_images), (
            f"create_pages 返回页数不符: {created!r}"
        )

        steps = await client.get_pipeline_steps_from_config()
        assert steps, "Koharu /config pipeline 为空,无法启动 pipeline"

        target_language = os.environ.get(
            "KOHARU_TARGET_LANGUAGE", DEFAULT_TARGET_LANGUAGE
        )
        operation_id = await client.start_pipeline(
            steps,
            target_language=target_language,
        )
        operation = await client.wait_operation(operation_id, timeout_seconds=600.0)
        assert str(operation.get("status", "")).lower() in OPERATION_FINISHED_STATUSES, (
            f"pipeline 未完成: {operation!r}"
        )

        content, content_type = await client.export_project("rendered")
        assert content, "export_project 返回空内容"
        output_dir = tmp_path / "rendered_output"
        saved = save_exported_images(
            content,
            content_type,
            output_dir,
            base_name="translated",
        )
        assert saved, "save_exported_images 未保存任何图片"
        for path_str in saved:
            path = Path(path_str)
            assert path.is_file() and path.stat().st_size > 0, f"输出文件异常: {path}"
            with Image.open(path) as image:
                image.verify()
    finally:
        if project_id:
            try:
                await client.close_project()
            except Exception as exc:
                print(f"[e2e] close_project 清理失败(忽略): {exc!r}")
            try:
                await client.delete_project(project_id)
            except Exception as exc:
                print(f"[e2e] delete_project 清理失败(忽略): {exc!r}")


@pytest.mark.asyncio
async def test_config_pipeline_steps(koharu_client: KoharuClient) -> None:
    """GET /config 的 pipeline 段各引擎键齐全且非空。"""
    config = await koharu_client.get_config()
    pipeline = config.get("pipeline")
    assert pipeline is not None, f"GET /config 缺少 pipeline 段: {config!r}"
    for key in ("detector", "segmenter", "ocr", "translator", "inpainter", "renderer"):
        value = pipeline.get(key)
        assert value, f"pipeline.{key} 缺失或为空"
    font = pipeline.get("fontDetector") or pipeline.get("font_detector")
    assert font, "pipeline.fontDetector/font_detector 缺失或为空"
    bubble = pipeline.get("bubbleSegmenter") or pipeline.get("bubble_segmenter")
    assert bubble, "pipeline.bubbleSegmenter/bubble_segmenter 缺失或为空"
    steps = await koharu_client.get_pipeline_steps_from_config()
    assert len(steps) >= 8, f"pipeline steps 不足 8 个: {steps}"
