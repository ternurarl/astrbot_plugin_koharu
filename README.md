# astrbot_plugin_koharu

Koharu 漫画翻译插件。通过 Koharu HTTP API 在 AstrBot 聊天中翻译漫画图片。

## 功能

- 使用指令 `漫画翻译` 触发漫画图片翻译。
- 支持单图和多图。
- 支持两种使用方式：
  - 发送 `/漫画翻译 Simplified Chinese` 并附带图片。
  - 先发送 `/漫画翻译`，再按提示发送图片。
- 支持 AstrBot WebUI 配置管理。
- 支持中文和英文 WebUI 文案，资源位于 `.astrbot-plugin/i18n/`。
- 翻译结果保存到 `data/plugin_data/astrbot_plugin_koharu/outputs/`。

## 前置条件

- 请先完成 Koharu 本体部署：

- 官方仓库：[Koharu](https://github.com/mayocream/koharu)
- 参考指南：[Koharu 安装指南](https://koharu.rs/zh-CN/how-to/install-koharu/)

插件可接受 `http://host:port` 或 `http://host:port/api/v1`。

## Koharu 调用流程

每次翻译请求会执行：

1. 调用 `GET /meta` 等待 Koharu 就绪。
2. 调用 `POST /projects` 创建 Koharu 项目。
3. 调用 `POST /pages` 上传图片。
4. 可选调用 `PUT /llm/current` 加载 LLM。
5. 调用 `POST /pipelines` 启动翻译 pipeline。
6. 调用 `GET /operations` 轮询任务状态。
7. 调用 `POST /projects/current/export` 导出 rendered 图片。
8. 将翻译后的图片发送回聊天。

## 配置项

- `koharu_api_base_url`：Koharu HTTP API 地址。
- `target_language`：指令未指定语言时的默认目标语言。
- `pipeline_steps`：可选，逗号分隔的 engine id。留空则从 Koharu `/config` 读取当前选择的引擎。
- `auto_load_llm`：当 Koharu 没有预先加载翻译模型时启用。
- `llm_kind`、`llm_provider_id`、`llm_model_id`：LLM 加载目标。
- `pipeline_timeout_seconds`：等待 Koharu 翻译完成的最长时间。
- `max_images_per_request`：单次输入图片数限制。
- `max_send_images`：最多返回图片数，`0` 表示全部返回。

## 指令

```text
/漫画翻译
/漫画翻译 Simplified Chinese
/manga_translate English
```

如果指令消息没有附带图片，插件会等待同一会话中的下一条图片消息。

---

# English

Koharu manga translation plugin. It translates manga images in AstrBot chats through the Koharu HTTP API.

## Features

- Command `漫画翻译` triggers manga image translation.
- Supports one image or multiple images.
- Supports both command-with-image and command-then-image workflows.
- Supports AstrBot WebUI configuration.
- Supports Chinese and English WebUI text through `.astrbot-plugin/i18n/`.

## Usage

```text
/漫画翻译
/漫画翻译 Simplified Chinese
/manga_translate English
```
