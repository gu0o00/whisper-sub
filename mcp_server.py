"""
whisper-sub MCP Server

提供三个工具：
1. health_check        —— 判断 MCP 服务是否健康（并检查依赖与模型目录）
2. generate_subtitle   —— 指定视频文件路径，生成 SRT 字幕文件（基于 faster-whisper）
3. translate_subtitle  —— 指定 SRT 字幕文件，生成中文字幕文件（基于本地 OpenAI 兼容接口）

支持多种传输模式，可通过命令行参数选择：
- stdio            （默认）本地标准输入输出，供 Claude Code / Claude Desktop 调用
- streamable-http  通过 HTTP 暴露，供外部 agent 经网络访问（推荐，MCP 规范的新协议）
- sse              Server-Sent Events 传输（兼容旧客户端）

示例：
  # 本地 stdio
  python mcp_server.py
  # 网络访问（默认 0.0.0.0:8000，路径 /mcp）
  python mcp_server.py --transport streamable-http --host 0.0.0.0 --port 8000
  # SSE 模式
  python mcp_server.py --transport sse --host 0.0.0.0 --port 8000
"""

import argparse
import logging
import os
import sys
import threading
from pathlib import Path

# 配置日志：输出到 stderr（stdio 模式下不干扰 MCP 协议的 stdout 通信）
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stderr,
)
logger = logging.getLogger("whisper-sub")

# 屏蔽 openai/httpx 每次请求的 INFO 级输出（如 "HTTP Request: POST .../chat/completions ... 200 OK"），
# 仅保留 WARNING 及以上，避免刷屏 stderr 日志。
for _noisy in ("httpx", "openai", "openai._base_client"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

# 将 nvidia-cublas-cu12 的 DLL 目录加入 PATH，解决 cublas64_12.dll 找不到的问题
_cublas_dir = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia" / "cublas" / "bin"
if _cublas_dir.is_dir():
    os.add_dll_directory(str(_cublas_dir))

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

# 复用现有脚本中的可复用函数
from subtitle import (
    collect_segments,
    fix_segment_durations,
    generate_srt,
)
from translate_srt import (
    entries_to_srt,
    is_lang_code,
    lang_to_code,
    make_batches,
    normalize_target_lang,
    parse_srt,
    translate_batches,
)
from openai import OpenAI

# 默认配置（与现有脚本保持一致）
DEFAULT_MODEL = "large-v3-turbo"
# 默认配置
DEFAULT_MODEL = "large-v3"
DEFAULT_MODELS_DIR = str(Path(__file__).parent / "models")
DEFAULT_DEVICE = "cuda"
DEFAULT_COMPUTE_TYPE = "float16"
DEFAULT_BEAM_SIZE = 5
DEFAULT_LANGUAGE = "zh"

DEFAULT_BASE_URL = "http://127.0.0.1:1234/v1"
DEFAULT_API_KEY = "sk-lm-o6sPQC0k:fa7Zl81TeXS2rB1qTDUD"
DEFAULT_TRANSLATE_MODEL = "hy-mt2-1.8b"
DEFAULT_TARGET_LANG = "中文"
DEFAULT_BATCH_SIZE = 10
DEFAULT_MAX_CHARS = 4000
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 4096

mcp = FastMCP("whisper-sub")


# ---------------------------------------------------------------------------
# Whisper 模型缓存
# ---------------------------------------------------------------------------
# 每次调用 generate_subtitle 都重新构造 WhisperModel 会带来巨大的开销
# （大模型加载、GPU 显存初始化可能耗时数秒到数十秒）。这里在进程内缓存
# 已加载的模型实例，按 (model, device, compute_type, models_dir,
# local_files_only) 作为 key 复用，避免重复加载。同时提供启动时预加载
# 的能力（见 main 中的 --preload）。
_whisper_lock = threading.Lock()
_whisper_cache: dict[tuple, object] = {}


def get_whisper_model(
    model: str,
    device: str,
    compute_type: str,
    models_dir: str,
    local_files_only: bool,
):
    """返回（按需加载并）缓存的 WhisperModel 实例。

    仅当加载参数与缓存 key 不同时才重新构造模型；相同参数直接复用，
    从而避免每次调用 generate_subtitle 都重复加载模型。
    """
    dev = "cuda" if device == "cuda" else "cpu" if device == "cpu" else "auto"
    ctype = compute_type if compute_type != "auto" else "default"
    key = (model, dev, ctype, str(models_dir), bool(local_files_only))

    # 双检：命中则直接返回，避免每次都拿锁
    cached = _whisper_cache.get(key)
    if cached is not None:
        return cached

    with _whisper_lock:
        cached = _whisper_cache.get(key)
        if cached is not None:
            return cached

        # 延迟导入，避免 health_check 时也强制加载模型库
        from faster_whisper import WhisperModel

        logger.info(
            "加载 WhisperModel | model=%r, device=%r, compute_type=%r, "
            "models_dir=%r, local_files_only=%r",
            model, dev, ctype, models_dir, local_files_only,
        )
        instance = WhisperModel(
            model,
            device=dev,
            compute_type=ctype,
            download_root=models_dir,
            local_files_only=local_files_only,
        )
        _whisper_cache[key] = instance
        return instance


@mcp.tool()
def health_check() -> dict:
    """判断 MCP 服务是否健康。

    检查项：
    - MCP server 本身可达（能调用本工具即说明 server 已启动）；
    - faster-whisper / openai 依赖是否可导入；
    - 本地模型目录是否存在、是否包含模型文件。

    返回一个包含各检查项状态与详细信息的字典。
    """
    logger.info("调用 health_check | 参数: (无)")
    result = {
        "status": "ok",
        "mcp_server": "running",
        "checks": {},
    }

    # 依赖检查
    try:
        import faster_whisper  # noqa: F401
        result["checks"]["faster_whisper"] = "ok"
    except Exception as e:  # noqa: BLE001
        result["checks"]["faster_whisper"] = f"error: {e}"
        result["status"] = "degraded"

    try:
        import openai  # noqa: F401
        result["checks"]["openai"] = "ok"
    except Exception as e:  # noqa: BLE001
        result["checks"]["openai"] = f"error: {e}"
        result["status"] = "degraded"

    # 模型目录检查
    models_dir = Path(DEFAULT_MODELS_DIR)
    if models_dir.is_dir():
        # 粗略判断目录下是否存在模型文件（.pt / .bin / 子目录等）
        has_model = any(models_dir.iterdir())
        result["checks"]["models_dir"] = {
            "path": str(models_dir),
            "exists": True,
            "has_content": has_model,
        }
    else:
        result["checks"]["models_dir"] = {
            "path": str(models_dir),
            "exists": False,
            "has_content": False,
        }
        # 目录不存在不算致命，首次运行会自动创建并下载

    return result


@mcp.tool()
def generate_subtitle(
    video_path: str,
    language: str = DEFAULT_LANGUAGE,
    model: str = DEFAULT_MODEL,
    models_dir: str = DEFAULT_MODELS_DIR,
    device: str = DEFAULT_DEVICE,
    compute_type: str = DEFAULT_COMPUTE_TYPE,
    beam_size: int = DEFAULT_BEAM_SIZE,
    output: str | None = None,
    local_files_only: bool = False,
) -> dict:
    """为指定视频文件生成 SRT 字幕文件。

    参数：
    - video_path: 视频文件路径（必填）
    - language: 音频源语言代码（默认 zh）
    - model: 模型大小或本地路径（默认 large-v3-turbo）
    - models_dir: 模型缓存目录（默认 ./models）
    - device: 推理设备 auto/cpu/cuda（默认 cuda）
    - compute_type: 计算精度（默认 float16）
    - beam_size: beam search 大小（默认 5）
    - output: 输出 SRT 路径（默认 {视频名}.{检测语言}.srt）
    - local_files_only: 仅使用本地缓存模型，不联网下载（默认 False）

    返回包含输出路径、片段数、检测到的语言与时长的字典。
    """
    logger.info(
        "调用 generate_subtitle | 参数: "
        "video_path=%r, language=%r, model=%r, models_dir=%r, "
        "device=%r, compute_type=%r, beam_size=%r, output=%r, local_files_only=%r",
        video_path, language, model, models_dir,
        device, compute_type, beam_size, output, local_files_only,
    )
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"视频文件不存在：{video_path}")

    os.makedirs(models_dir, exist_ok=True)

    whisper_model = get_whisper_model(
        model, device, compute_type, models_dir, local_files_only,
    )

    segments_iter, info = whisper_model.transcribe(
        video_path,
        language=language,
        beam_size=beam_size,
        vad_filter=True,
    )
    # faster-whisper 的 segments 是惰性生成器，迭代才真正触发转写；
    # 边收集边把进度记到 stderr 日志（stdout 用于 MCP 协议，不能写）。
    segments = collect_segments(
        segments_iter, info.duration,
        log=lambda m: logger.info("generate_subtitle | %s", m),
    )
    logger.info("generate_subtitle | 转写完成，共 %d 段", len(segments))

    # 修正异常超长的字幕持续时间（faster-whisper 偶发出现几十分钟的片段）
    fixed = fix_segment_durations(segments)
    if fixed:
        logger.info("generate_subtitle | 修正 %d 条超长字幕", fixed)

    # 确定输出路径
    # 默认文件名格式：{文件名}.{语言类型}.srt，语言类型使用实际识别到的语言代码
    detected_lang = info.language
    if output:
        base = os.path.splitext(output)[0]
    else:
        video_base = os.path.splitext(video_path)[0]
        base = f"{video_base}.{detected_lang}"
    srt_path = base + ".srt"

    generate_srt(segments, srt_path)

    return {
        "output": srt_path,
        "segments": len(segments),
        "detected_language": info.language,
        "language_probability": float(info.language_probability),
        "duration_seconds": float(info.duration),
    }


@mcp.tool()
def translate_subtitle(
    srt_path: str,
    output: str | None = None,
    base_url: str = DEFAULT_BASE_URL,
    api_key: str = DEFAULT_API_KEY,
    model: str = DEFAULT_TRANSLATE_MODEL,
    target_lang: str = DEFAULT_TARGET_LANG,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_chars: int = DEFAULT_MAX_CHARS,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> dict:
    """将指定 SRT 字幕文件翻译成中文（或指定目标语言）。

    调用本地 OpenAI 兼容接口（默认 LM Studio：http://127.0.0.1:1234/v1），
    采用结构化数据翻译策略，严格保留序号、时间轴与结构。

    参数：
    - srt_path: 输入 SRT 文件路径（必填）
    - output: 输出文件路径（默认 {文件名}.{目标语言代码}.srt，替换原语言段）
    - base_url: OpenAI 兼容接口地址
    - api_key: API Key（本地服务通常随意）
    - model: 翻译模型名称（需与本地服务加载的模型 ID 一致，默认 hy-mt2-1.8b）
    - target_lang: 目标语言（默认 中文）
    - batch_size: 每批最大条目数（默认 10）
    - max_chars: 每批最大字符数（默认 4000）
    - temperature: 采样温度（默认 0.1）
    - max_tokens: 每批响应最大 token 数（默认 4096）

    返回包含输出路径与已翻译条目数的字典。
    """
    logger.info(
        "调用 translate_subtitle | 参数: "
        "srt_path=%r, output=%r, base_url=%r, api_key=%r, model=%r, "
        "target_lang=%r, batch_size=%r, max_chars=%r, "
        "temperature=%r, max_tokens=%r",
        srt_path, output, base_url, "***" if api_key else api_key, model,
        target_lang, batch_size, max_chars,
        temperature, max_tokens,
    )
    if not os.path.isfile(srt_path):
        raise FileNotFoundError(f"字幕文件不存在：{srt_path}")

    # 归一化目标语言：允许传 zh / en / ja 等代码，内部映射为语言全称，
    # 避免小模型把 "翻译成 zh" 误解而漂移到英语。
    target_lang = normalize_target_lang(target_lang)

    # 输出路径
    # 若未显式指定，按 {文件名}.{目标语言代码}.srt 生成：
    #   BBI-174.ja.srt -> BBI-174.zh.srt（替换已存在的语言段）
    #   BBI-174.srt    -> BBI-174.zh.srt（无语言段则追加）
    if output:
        out_path = output
    else:
        root, ext = os.path.splitext(srt_path)
        ext = ext or ".srt"
        target_code = lang_to_code(target_lang)
        parts = root.split(".")
        if len(parts) > 1 and is_lang_code(parts[-1]):
            parts[-1] = target_code
            new_root = ".".join(parts)
        else:
            new_root = f"{root}.{target_code}"
        out_path = new_root + ext

    with open(srt_path, "r", encoding="utf-8") as f:
        content = f.read()
    entries = parse_srt(content)
    if not entries:
        raise ValueError(f"未能解析出任何字幕条目：{srt_path}")

    client = OpenAI(base_url=base_url, api_key=api_key)
    batches = make_batches(entries, batch_size, max_chars)
    logger.info("translate_subtitle | 分为 %d 个批次", len(batches))

    # 逐批翻译，进度记到 stderr 日志（stdout 用于 MCP 协议，不能写）
    translated = translate_batches(
        client, model, batches, target_lang, temperature, max_tokens,
        log=lambda m: logger.info("translate_subtitle | %s", m),
    )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(entries_to_srt(translated))

    return {
        "output": out_path,
        "translated": len(translated),
        "batches": len(batches),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="whisper-sub MCP Server")
    parser.add_argument(
        "--transport", "-t",
        choices=["stdio", "streamable-http", "sse"],
        default="stdio",
        help="传输模式（默认 stdio；网络访问用 streamable-http 或 sse）",
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0",
        help="HTTP/SSE 监听地址（默认 0.0.0.0，对外可访问）",
    )
    parser.add_argument(
        "--port", "-p", type=int, default=8000,
        help="HTTP/SSE 监听端口（默认 8000）",
    )
    parser.add_argument(
        "--path", type=str, default="/mcp",
        help="streamable-http 的挂载路径（默认 /mcp）",
    )
    parser.add_argument(
        "--allowed-hosts", type=str, default=None,
        help="允许的 Host 头列表（逗号分隔），用于 DNS rebinding 防护白名单。"
        "示例：10.1.1.123:8000,10.1.1.50:*。不指定时按 --allow-all-hosts 处理。",
    )
    parser.add_argument(
        "--allow-all-hosts", action="store_true",
        help="关闭 DNS rebinding Host 头校验，允许任意 Host/IP 访问"
        "（仅用于受信任的局域网；公网部署请改用 --allowed-hosts 或反向代理）。",
    )
    parser.add_argument(
        "--preload", action="store_true", default=True,
        help="启动时即加载 Whisper 模型（使用默认配置），后续调用 generate_subtitle "
        "可直接复用，避免首次调用时的加载等待。默认启用。",
    )
    parser.add_argument(
        "--no-preload", dest="preload", action="store_false",
        help="关闭启动时预加载模型，改为首次调用 generate_subtitle 时按需加载。",
    )
    parser.add_argument(
        "--preload-model", type=str, default=DEFAULT_MODEL,
        help=f"预加载的模型（默认 {DEFAULT_MODEL}，仅当 --preload 生效）",
    )
    parser.add_argument(
        "--preload-device", type=str, default=DEFAULT_DEVICE,
        help=f"预加载的推理设备（默认 {DEFAULT_DEVICE}，仅当 --preload 生效）",
    )
    parser.add_argument(
        "--preload-compute-type", type=str, default=DEFAULT_COMPUTE_TYPE,
        help=f"预加载的计算精度（默认 {DEFAULT_COMPUTE_TYPE}，仅当 --preload 生效）",
    )
    args = parser.parse_args()

    if args.preload:
        os.makedirs(DEFAULT_MODELS_DIR, exist_ok=True)
        get_whisper_model(
            args.preload_model,
            args.preload_device,
            args.preload_compute_type,
            DEFAULT_MODELS_DIR,
            False,
        )

    # FastMCP 构造时若 host 在 localhost 范围会自动启用 DNS rebinding 防护，
    # 仅放行 127.0.0.1/localhost/::1 的 Host 头，导致外部 agent 用 LAN IP
    # 访问时返回 421 Misdirected Request。这里按参数重置安全策略。
    def apply_network_settings() -> None:
        mcp.settings.host = args.host
        mcp.settings.port = args.port
        if args.allow_all_hosts or args.allowed_hosts is None:
            # 默认对外提供服务的场景下，关闭 Host 头校验
            mcp.settings.transport_security = TransportSecuritySettings(
                enable_dns_rebinding_protection=False
            )
        else:
            hosts = [h.strip() for h in args.allowed_hosts.split(",") if h.strip()]
            origins = [
                f"http://{h.rstrip(':*')}:*" for h in hosts
            ]
            mcp.settings.transport_security = TransportSecuritySettings(
                enable_dns_rebinding_protection=True,
                allowed_hosts=hosts,
                allowed_origins=origins,
            )

    if args.transport == "stdio":
        mcp.run()
    elif args.transport == "streamable-http":
        # streamable-http 模式：外部 agent 通过 HTTP POST 访问
        # 端点：http://<host>:<port><path>，如 http://10.1.1.50:8000/mcp
        apply_network_settings()
        mcp.settings.streamable_http_path = args.path
        print(f"🌐 streamable-http 模式启动：http://{args.host}:{args.port}{args.path}", file=sys.stderr)
        mcp.run(transport="streamable-http")
    elif args.transport == "sse":
        # SSE 模式：兼容旧客户端，端点 http://<host>:<port>/sse
        apply_network_settings()
        print(f"🌐 SSE 模式启动：http://{args.host}:{args.port}/sse", file=sys.stderr)
        mcp.run(transport="sse")
