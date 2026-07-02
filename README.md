# whisper-sub

基于 [faster-whisper](https://github.com/SYSTRAN/faster-whisper) 的视频字幕生成与翻译工具集。支持通过命令行脚本或 MCP（Model Context Protocol）服务两种方式使用：

- 用 faster-whisper 对视频/音频转写，生成 SRT 字幕（可选带时间戳的纯文本）；
- 调用本地 OpenAI 兼容接口（如 LM Studio 加载 Hy-MT2 等翻译模型）将 SRT 字幕翻译成中文或其他语言，严格保留序号、时间轴与结构；
- 通过 MCP Server 把上述能力暴露为工具，供 Claude Code / Claude Desktop 或其他外部 agent 调用。

## 环境依赖

- Python ≥ 3.11
- 推荐使用 CUDA GPU（默认 `device=cuda`、`compute_type=float16`），也可回退 CPU
- 本地翻译服务（可选）：如 [LM Studio](https://lmstudio.ai/) 加载翻译模型并开启 OpenAI 兼容接口

安装依赖（使用 [uv](https://github.com/astral-sh/uv)）：

```bash
uv sync
```

> 推荐使用 CUDA GPU 运行 faster-whisper（默认 `device=cuda`、`compute_type=float16`），也可回退 CPU。

## 脚本说明

### `subtitle.py` —— 字幕生成（命令行）

使用 faster-whisper 对视频文件转写，输出 SRT 字幕。模型默认从 Hugging Face Hub 下载到 `./models` 目录，也可指向本地已转换的模型目录。

```bash
python subtitle.py <视频路径> [选项]
```

常用参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `-m, --model` | `large-v3-turbo` | 模型大小或本地路径（`tiny`/`base`/`small`/`medium`/`large-v3`/`large-v3-turbo` 等） |
| `--models-dir` | `./models` | 模型下载/缓存目录（`download_root`） |
| `-o, --output` | 与视频同名 `.srt` | 输出文件路径 |
| `-l, --language` | `zh` | 音频源语言代码 |
| `--device` | `cuda` | 推理设备 `auto`/`cpu`/`cuda` |
| `--compute_type` | `float16` | 计算精度 |
| `--beam_size` | `5` | Beam search 大小 |
| `--local-files-only` | 关闭 | 仅使用本地缓存模型，不联网下载 |
| `--txt` | 关闭 | 同时生成带时间戳的纯文本文件 |

示例：

```bash
python subtitle.py video.mp4 -l zh -m large-v3-turbo --txt
# 生成 video.srt 与 video.txt
```

### `translate_srt.py` —— 字幕翻译（命令行）

将 SRT 字幕翻译成中文（或其他语言）。调用本地 OpenAI 兼容接口，采用 Hy-MT2 推荐的「结构化数据翻译」策略：将 SRT 视为结构化数据，仅翻译字幕文本，严格保留序号、时间轴与结构。

内置容错机制：批次翻译条目数校验、未翻译/原样回显检测、单条回退重译，保证输出条目与原文一一对应。

```bash
python translate_srt.py <输入.srt> [选项]
```

常用参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `-o, --output` | 原文件名加 `.zh` | 输出文件路径 |
| `--base-url` | `http://127.0.0.1:1234/v1` | OpenAI 兼容接口地址 |
| `--api-key` | `lm-studio` | API Key（本地服务通常随意） |
| `-m, --model` | `hy-mt2-1.8b` | 翻译模型名称（需与本地服务加载的模型 ID 一致） |
| `--target-lang` | `中文` | 目标语言（支持 `zh`/`en`/`ja`/`ko` 等代码或全称） |
| `--batch-size` | `10` | 每批最大条目数 |
| `--max-chars` | `4000` | 每批最大字符数 |
| `--temperature` | `0.1` | 采样温度 |
| `--max-tokens` | `4096` | 每批响应最大 token 数 |

示例：

```bash
python translate_srt.py video.srt --target-lang 中文
# 生成 video.zh.srt
```

### `mcp_server.py` —— MCP Server

把字幕生成与翻译能力封装为 MCP 工具，供 Claude Code / Claude Desktop（stdio）或外部 agent（HTTP/SSE）调用。复用 `subtitle.py` 与 `translate_srt.py` 中的函数，并在进程内缓存 Whisper 模型以避免重复加载。

提供的三个工具：

| 工具 | 作用 |
| --- | --- |
| `health_check` | 检查服务健康状态、依赖可导入性、模型目录是否存在 |
| `generate_subtitle` | 指定视频路径，生成 SRT 字幕（基于 faster-whisper） |
| `translate_subtitle` | 指定 SRT 文件，翻译成中文或指定语言（基于本地 OpenAI 兼容接口） |

启动方式：

```bash
# 本地 stdio（供 Claude Code / Claude Desktop 调用，默认）
python mcp_server.py

# streamable-http 模式（推荐，供外部 agent 经网络访问）
python mcp_server.py --transport streamable-http --host 0.0.0.0 --port 8000
# 端点：http://<host>:8000/mcp

# SSE 模式（兼容旧客户端）
python mcp_server.py --transport sse --host 0.0.0.0 --port 8000
# 端点：http://<host>:8000/sse
```

常用参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `-t, --transport` | `stdio` | 传输模式 `stdio`/`streamable-http`/`sse` |
| `--host` | `0.0.0.0` | HTTP/SSE 监听地址 |
| `-p, --port` | `8000` | HTTP/SSE 监听端口 |
| `--path` | `/mcp` | streamable-http 挂载路径 |
| `--allowed-hosts` | 不限 | 允许的 Host 头白名单（逗号分隔），用于 DNS rebinding 防护 |
| `--allow-all-hosts` | 关闭 | 关闭 Host 头校验（仅用于受信任局域网） |
| `--preload` / `--no-preload` | 启用 | 启动时即加载 Whisper 模型，避免首次调用等待 |
| `--preload-model` | `large-v3` | 预加载的模型 |
| `--preload-device` | `cuda` | 预加载的推理设备 |
| `--preload-compute-type` | `float16` | 预加载的计算精度 |

在 Claude Code 中接入 stdio 服务，可在 `.mcp.json` / `settings.json` 中配置；外部 agent 通过 HTTP 访问时，默认放行任意 Host 头（适合受信任局域网），公网部署请改用 `--allowed-hosts` 或反向代理。

## 典型工作流

```bash
# 1. 生成字幕
python subtitle.py video.mp4 -l zh

# 2. 翻译字幕
python translate_srt.py video.srt --target-lang 中文

# 或通过 MCP 服务，让 Claude 等 agent 直接调用 generate_subtitle / translate_subtitle
python mcp_server.py
```

## 配置默认值

`mcp_server.py` 顶部集中维护了默认配置（模型、设备、接口地址、翻译模型与批次参数等）。命令行脚本与 MCP 服务保持一致；如需调整默认行为，修改对应常量即可。
