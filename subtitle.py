"""
使用 faster-whisper 对视频文件生成字幕
模型：本地 models/large-v3-turbo.pt
输出：SRT 格式字幕文件
"""

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

from faster_whisper import WhisperModel


MAX_SEGMENT_DURATION = 5.0  # 单条字幕最大持续时长（秒）


def format_timestamp(seconds: float) -> str:
    """将秒数转换为 SRT 时间戳格式 HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def fix_segment_durations(segments: list, max_duration: float = MAX_SEGMENT_DURATION) -> int:
    """修正异常超长的字幕持续时间。

    faster-whisper 偶发出现某条字幕 end 远超实际（如持续几十分钟），
    这里将超过 max_duration 的片段结束时间截断为 start + max_duration，
    并保证与下一条片段不重叠。

    直接原地改写 list 元素（faster-whisper 的 Segment 是不可变 NamedTuple，
    所以替换为同等字段的可变对象）。返回被修正的片段数量。
    """
    fixed = 0
    n = len(segments)
    for i in range(n):
        segment = segments[i]
        start = segment.start
        end = segment.end
        # 防御 NaN/inf：非有限值直接跳过，避免误判
        if not (start < end and end - start > max_duration):
            continue

        # 截断为本片段起点 + 最大时长
        new_end = start + max_duration
        # 若与下一条片段重叠，则压到下一条起点前留 1ms 间隔
        if i + 1 < n:
            next_start = segments[i + 1].start
            if new_end > next_start:
                new_end = max(start + 0.001, next_start - 0.001)

        segments[i] = SimpleNamespace(start=start, end=new_end, text=segment.text)
        fixed += 1
    return fixed


def collect_segments(
    segments_iter,
    total_duration: float,
    progress_step: int = 50,
    log=print,
    progress=None,
) -> list:
    """流式收集 faster-whisper 的惰性片段生成器。

    faster-whisper 的 transcribe() 返回惰性生成器，迭代才真正触发转写；
    长视频耗时很久，这里边收集边按周期输出进度，避免看起来像卡死。

    log 为人类可读的进度文本回调，默认 print（CLI 场景）；MCP 等需要写入
    stderr 的场景可传入 logger 相关回调。

    progress 为结构化进度回调，签名为 progress(value, total, message=None)：
    以「已处理的音频秒数 / 总时长」作为进度比例（value=当前秒数，total=总秒数），
    供 MCP 客户端渲染进度条。默认 None（不上报）。迭代结束后会上报一次 100%。
    """
    segments: list = []
    for seg in segments_iter:
        segments.append(seg)
        if progress_step and len(segments) % progress_step == 0:
            # 以当前片段结束时间作为「已处理到的音频位置」，并夹紧到 [0, total]
            cur = float(getattr(seg, "end", getattr(seg, "start", 0.0)) or 0.0)
            cur = max(0.0, min(cur, total_duration))
            if log:
                log(f"  ... 已转写 {len(segments)} 段，当前 {cur:.0f}s / {total_duration:.0f}s")
            if progress:
                progress(
                    cur, total_duration,
                    f"已转写 {len(segments)} 段，{cur:.0f}s/{total_duration:.0f}s",
                )
    # 全部收集完成：上报 100%
    if progress and total_duration:
        progress(total_duration, total_duration, f"转写完成，共 {len(segments)} 段")
    return segments


def generate_srt(segments: list, output_path: str, log=print) -> None:
    """将识别结果写入 SRT 字幕文件"""
    with open(output_path, "w", encoding="utf-8") as f:
        for i, segment in enumerate(segments, start=1):
            start = format_timestamp(segment.start)
            end = format_timestamp(segment.end)
            text = segment.text.strip()
            f.write(f"{i}\n{start} --> {end}\n{text}\n\n")

    if log:
        log(f"✅ 字幕已生成：{output_path}")


def generate_txt(segments: list, output_path: str, log=print) -> None:
    """将识别结果写入纯文本文件（带时间戳）"""
    with open(output_path, "w", encoding="utf-8") as f:
        for segment in segments:
            start = format_timestamp(segment.start)
            end = format_timestamp(segment.end)
            text = segment.text.strip()
            f.write(f"[{start} --> {end}] {text}\n")

    if log:
        log(f"✅ 文本已生成：{output_path}")


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
    segments_iter, info = model.transcribe(
        args.video,
        language=args.language,
        beam_size=args.beam_size,
        vad_filter=True,  # 过滤静音部分
    )

    print(f"ℹ️  检测语言：{info.language}（概率 {info.language_probability:.2%}）")
    print(f"ℹ️  时长：{info.duration:.0f} 秒")

    # 收集所有片段
    # faster-whisper 的 segments 是惰性生成器，list() 才真正触发转写；
    # 长视频在这里耗时很久，所以边迭代边打印进度，避免看起来像卡死。
    segments = collect_segments(segments_iter, info.duration)
    print(f"ℹ️  转写完成，共 {len(segments)} 段。")

    # 修正异常超长的字幕持续时间（faster-whisper 偶发出现几十分钟的片段）
    fixed = fix_segment_durations(segments)
    if fixed:
        print(f"ℹ️  共修正 {fixed} 条超长字幕的结束时间。")

    # 确定输出路径
    # 默认文件名格式：{文件名}.{语言类型}.srt，语言类型使用实际识别到的语言代码
    detected_lang = info.language
    if args.output:
        base = os.path.splitext(args.output)[0]
    else:
        video_base = os.path.splitext(args.video)[0]
        base = f"{video_base}.{detected_lang}"

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
