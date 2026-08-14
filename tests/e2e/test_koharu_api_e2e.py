"""Koharu HTTP API 全生命周期 E2E 测试(真实实例)。

覆盖:
- ``test_api_full_lifecycle``:ready → create project → create pages → 解析 pipeline
  steps → start pipeline(真实 OCR+LLM+inpainting)→ wait_operation → export → 保存
  输出图片并验证可被 Pillow 打开。pipeline 失败时如实抛出 KoharuApiError(含错误详情)。
- ``test_config_pipeline_steps``:GET /config 的 pipeline 段与 0.66 契约
  PipelineConfig 对应(detection/ocr/translation/inpainting 键齐全),
  get_pipeline_steps_from_config 返回 ["full"]。

所有用例在 finally 中清理 close_project() + delete_project()(0.66 按名称删除)。
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from PIL import Image

from koharu_client import KoharuClient, extract_project_id, save_exported_images

OPERATION_FINISHED_STATUSES = frozenset(
    {"finished", "completed", "complete", "succeeded", "success", "done"}
)


@pytest.mark.asyncio
async def test_api_full_lifecycle(
    koharu_client: KoharuClient,
    test_images: list[Path],
    tmp_path: Path,
) -> None:
    """完整走一遍 Koharu API 生命周期并验证渲染输出。"""
    client = koharu_client
    project_name: str | None = None
    try:
        meta = await client.get_meta()
        assert meta.get("version"), f"GET /meta 未返回 version: {meta!r}"

        # 先关闭可能残留的 current project,再创建新项目(与插件逻辑一致)
        await client.close_project_if_any()

        created_name = f"e2e-{int(time.time())}"
        project = await client.create_project(created_name)
        project_name = extract_project_id(project)
        assert project_name, f"create_project 未返回项目标识: {project!r}"

        created = await client.create_pages(test_images)
        assert len(created.get("pages", [])) == len(test_images), (
            f"create_pages 返回页数不符: {created!r}"
        )

        steps = await client.get_pipeline_steps_from_config()
        assert steps, "Koharu /config pipeline 为空,无法启动 pipeline"

        operation_id = await client.start_pipeline(steps)
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
        if project_name:
            try:
                await client.close_project()
            except Exception as exc:
                print(f"[e2e] close_project 清理失败(忽略): {exc!r}")
            try:
                await client.delete_project(project_name)
            except Exception as exc:
                print(f"[e2e] delete_project 清理失败(忽略): {exc!r}")


@pytest.mark.asyncio
async def test_config_pipeline_steps(koharu_client: KoharuClient) -> None:
    """GET /config 的 pipeline 段为 0.66 键名,get_pipeline_steps_from_config 返回 full。"""
    config = await koharu_client.get_config()
    pipeline = config.get("pipeline")
    assert pipeline is not None, f"GET /config 缺少 pipeline 段: {config!r}"
    for key in ("detection", "ocr", "translation", "inpainting"):
        assert pipeline.get(key), f"pipeline.{key} 缺失或为空"
    steps = await koharu_client.get_pipeline_steps_from_config()
    assert steps == ["full"], f"0.66 应为固定阶段流水线(full),实际: {steps}"
