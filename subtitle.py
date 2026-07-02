"""
使用 faster-whisper 对视频文件生成字幕
模型：本地 models/large-v3-turbo.pt
输出：SRT 格式字幕文件
"""

import argparse
import os
import sys
from pathlib import Path

from faster_whisper import WhisperModel


def format_timestamp(seconds: float) -> str:
    """将秒数转换为 SRT 时间戳格式 HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def generate_srt(segments: list, output_path: str) -> None:
    """将识别结果写入 SRT 字幕文件"""
    with open(output_path, "w", encoding="utf-8") as f:
        for i, segment in enumerate(segments, start=1):
            start = format_timestamp(segment.start)
            end = format_timestamp(segment.end)
            text = segment.text.strip()
            f.write(f"{i}\n{start} --> {end}\n{text}\n\n")

    print(f"✅ 字幕已生成：{output_path}")


def generate_txt(segments: list, output_path: str) -> None:
    """将识别结果写入纯文本文件（带时间戳）"""
    with open(output_path, "w", encoding="utf-8") as f:
        for segment in segments:
            start = format_timestamp(segment.start)
            end = format_timestamp(segment.end)
            text = segment.text.strip()
            f.write(f"[{start} --> {end}] {text}\n")

    print(f"✅ 文本已生成：{output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="使用 faster-whisper 为视频生成字幕"
    )
    parser.add_argument(
        "video", type=str, help="视频文件路径"
    )
    parser.add_argument(
        "-m", "--model", type=str, default="large-v3-turbo",
        help=(
            "模型大小或路径（默认：large-v3-turbo）。"
            "可选：tiny, base, small, medium, large-v1, large-v2, large-v3, "
            "large-v3-turbo, turbo；也可直接指定本地已转换的模型目录路径。"
        )
    )
    parser.add_argument(
        "--models-dir", type=str,
        default=str(Path(__file__).parent / "models"),
        help="模型下载/缓存目录，即 download_root（默认：./models）",
    )
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="输出文件路径（默认与视频同名，.srt 扩展名）"
    )
    parser.add_argument(
        "-l", "--language", type=str, default="zh",
        help="音频源语言代码，即视频中说的语言（默认：zh）"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        choices=["auto", "cpu", "cuda"],
        help="推理设备（默认：cuda）"
    )
    parser.add_argument(
        "--compute_type", type=str, default="float16",
        choices=["auto", "int8", "int8_float16", "int8_bfloat16",
                 "int16", "float16", "bfloat16", "float32"],
        help="计算精度（默认：float16）"
    )
    parser.add_argument(
        "--beam_size", type=int, default=5,
        help="Beam search 大小（默认：5）"
    )
    parser.add_argument(
        "--local-files-only", action="store_true",
        help="仅使用本地已缓存的模型，不联网下载（默认：允许下载到 ./models）",
    )
    parser.add_argument(
        "--txt", action="store_true",
        help="同时生成带时间戳的纯文本文件"
    )

    args = parser.parse_args()

    # 检查视频文件
    if not os.path.isfile(args.video):
        print(f"❌ 视频文件不存在：{args.video}")
        sys.exit(1)

    # 确保模型缓存目录存在
    os.makedirs(args.models_dir, exist_ok=True)

    # 解析设备
    device = "cuda" if args.device == "cuda" else "cpu" if args.device == "cpu" else "auto"
    compute_type = args.compute_type if args.compute_type != "auto" else "default"

    print(f"📦 模型：{args.model}")
    print(f"📁 模型目录：{args.models_dir}")
    print(f"🎬 视频：{args.video}")
    print(f"🌐 语言：{args.language}")
    print(f"⚙️  设备：{device}，精度：{compute_type}")

    # 加载模型（model 为大小名时自动从 HF Hub 下载到 models_dir）
    print("⏳ 正在加载模型...")
    model = WhisperModel(
        args.model,
        device=device,
        compute_type=compute_type,
        download_root=args.models_dir,
        local_files_only=args.local_files_only,
    )

    # 转写
    print("⏳ 正在转写...")
    segments, info = model.transcribe(
        args.video,
        language=args.language,
        beam_size=args.beam_size,
        vad_filter=True,  # 过滤静音部分
    )

    print(f"ℹ️  检测语言：{info.language}（概率 {info.language_probability:.2%}）")
    print(f"ℹ️  时长：{info.duration:.0f} 秒")

    # 收集所有片段
    segments = list(segments)

    # 确定输出路径
    if args.output:
        base = os.path.splitext(args.output)[0]
    else:
        base = os.path.splitext(args.video)[0]

    srt_path = base + ".srt"
    txt_path = base + ".txt"

    # 输出 SRT
    generate_srt(segments, srt_path)

    # 可选 TXT
    if args.txt:
        generate_txt(segments, txt_path)

    print(f"✅ 完成！共 {len(segments)} 条字幕。")


if __name__ == "__main__":
    main()
