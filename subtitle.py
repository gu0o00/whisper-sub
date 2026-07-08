"""
使用 faster-whisper 对视频文件生成字幕
模型：本地 models/large-v3-turbo.pt
输出：SRT 格式字幕文件
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from faster_whisper import WhisperModel


MAX_SEGMENT_DURATION = 5.0  # 单条字幕最大持续时长（秒）

# 人声增强滤波链（面向 ASR/Whisper 转写，重点避免漏掉安静/被压低的对话）：
#   highpass=f=80       去低频隆隆声，保留男声基频
#   lowpass=f=7500      只切 7.5kHz 以上，保留齿擦音/清辅音频谱
#   afftdn=nr=6         降噪：压呼吸/气声等宽带噪声（频谱减法，对语音间隙的喘息
#                       有效，对语音内的气声有限）；默认 nr=12 偏激进会抹轻声，
#                       控制在 6 平衡"压呼吸"与"保轻声"——这是两者权衡的主旋钮
#   compand=...         轻度压扩（保守）：仅对极轻声(-55dB)温和抬 ~8dB、响声限幅，
#                       不大幅压缩动态范围——过强压缩会破坏 Whisper 依赖的语音自然
#                       动态特征反而降低识别率。轻声抬升主要交给 dynaudnorm
#                       "compand=attacks=0:decays=0.5:points=-80/-80|-55/-47|-30/-27|0/-4:gain=0, "
#   dynaudnorm=...      响度归一：自适应抬安静段；g=15 适度增益（更大易过响失真）
FFMPEG_AUDIO_FILTER = (
    "highpass=f=80, lowpass=f=7500, afftdn=nr=6, "
    "dynaudnorm=f=150:g=15:p=0.9"
)

# Whisper 转写参数（为保留轻声/被压低对话调优，CLI 与 MCP 共用以避免参数漂移）：
#   vad_parameters.threshold=0.15   默认 0.5 偏激进，轻声易被判非语音切掉，降到 0.15
#   no_speech_threshold=0.3         默认 0.6：段被判"非语音"概率超此值就跳过，轻声男声
#                                   极易被误判，降到 0.3 保留（漏台词的关键修复之一）
#   log_prob_threshold=-1.5         默认 -1.0：平均对数概率低于此值丢弃，轻声置信度天然
#                                   偏低，降到 -1.5 避免被当噪声扔掉（漏台词的关键修复之二）
#   compression_ratio_threshold=4.0 默认 2.4：压缩比超此值丢弃（防幻觉），轻声短句易
#                                   触发，放宽到 4.0
#   min_silence_duration_ms=1000     默认 2000：缩短以避免吞掉短停顿后的语音
WHISPER_TRANSCRIBE_KWARGS = dict(
    vad_filter=True,
    vad_parameters=dict(threshold=0.15, min_silence_duration_ms=1000),
    no_speech_threshold=0.3,
    log_prob_threshold=-1.5,
    compression_ratio_threshold=4.0,
)


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


def enhance_audio(video_path: str, wav_path: str, log=print) -> str:
    """用 ffmpeg 对视频音频做人声增强并导出 16kHz 单声道 wav。

    滤波链见 FFMPEG_AUDIO_FILTER：高通/低通框定人声频段、afftdn 降噪、
    compand 压扩动态范围。输出 wav 供 faster-whisper 转写使用，调用方需在
    转写结束后自行删除该临时文件。

    返回生成的 wav 文件路径。ffmpeg 缺失或执行失败时抛出异常，并清理残缺输出。
    """
    if shutil.which("ffmpeg") is None:
        raise FileNotFoundError(
            "未找到 ffmpeg，请先安装并将其加入 PATH（https://ffmpeg.org/）"
        )

    # -y 覆盖已存在的输出，避免 ffmpeg 在非交互环境下卡在覆盖确认提示
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",  # 丢弃视频流，仅处理音频
        "-af", FFMPEG_AUDIO_FILTER,
        "-ar", "16000",
        "-ac", "1",
        "-threads", "0",
        wav_path,
    ]
    if log:
        log(f"🔊 正在增强音频并导出：{wav_path}")

    # ffmpeg 进度信息走 stderr，捕获后仅在失败时输出，避免刷屏
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        if os.path.isfile(wav_path):
            try:
                os.remove(wav_path)
            except OSError:
                pass
        raise RuntimeError(
            f"ffmpeg 音频增强失败（返回码 {result.returncode}）：\n{result.stderr}"
        )
    if log:
        log(f"✅ 音频增强完成：{wav_path}")
    return wav_path


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
            "模型大小或路径（默认：large-v3-turbo，兼顾速度与识别效果）。"
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
    parser.add_argument(
        "--enhance-audio", action="store_true", default=True,
        help="先用 ffmpeg 对视频音频做人声增强（高通 80Hz/低通 7.5kHz/降噪/compand 压扩/"
             "响度归一），导出 16kHz 单声道 wav 再转写，转写完成后自动删除该 wav。默认启用。",
    )
    parser.add_argument(
        "--no-enhance-audio", dest="enhance_audio", action="store_false",
        help="关闭音频增强，直接用原始视频音轨转写。",
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
    if args.enhance_audio:
        print(f"🔊 音频增强：已启用（将导出 16kHz 单声道 wav 再转写）")
    else:
        print(f"🔊 音频增强：已关闭（直接使用原始视频音轨）")

    # 加载模型（model 为大小名时自动从 HF Hub 下载到 models_dir）
    print("⏳ 正在加载模型...")
    model = WhisperModel(
        args.model,
        device=device,
        compute_type=compute_type,
        download_root=args.models_dir,
        local_files_only=args.local_files_only,
    )

    # 音频输入：默认直接用视频文件；启用「音频增强」时先用 ffmpeg 导出增强后的 wav
    audio_input = args.video
    enhanced_wav = None
    if args.enhance_audio:
        # 临时 wav 保存在当前目录，与视频同名但扩展名为 .wav
        video_stem = Path(args.video).stem
        enhanced_wav = os.path.join(os.getcwd(), f"{video_stem}.wav")
        enhance_audio(args.video, enhanced_wav)
        audio_input = enhanced_wav

    # 转写
    # 注意：faster-whisper 的 segments 是惰性生成器，真正读取音频发生在迭代期间，
    # 所以增强 wav 必须保留到 collect_segments 结束才能删除。
    try:
        print("⏳ 正在转写...")
        segments_iter, info = model.transcribe(
            audio_input,
            language=args.language,
            beam_size=args.beam_size,
            # 转写参数（VAD + 三个丢弃阈值）为保留轻声调优，见 WHISPER_TRANSCRIBE_KWARGS
            **WHISPER_TRANSCRIBE_KWARGS,
        )

        print(f"ℹ️  检测语言：{info.language}（概率 {info.language_probability:.2%}）")
        print(f"ℹ️  时长：{info.duration:.0f} 秒")

        # 收集所有片段
        # faster-whisper 的 segments 是惰性生成器，list() 才真正触发转写；
        # 长视频在这里耗时很久，所以边迭代边打印进度，避免看起来像卡死。
        segments = collect_segments(segments_iter, info.duration)
    finally:
        # 转写完成（含异常退出）后清理临时增强 wav
        if enhanced_wav and os.path.isfile(enhanced_wav):
            try:
                os.remove(enhanced_wav)
                print(f"🧹 已清理临时音频：{enhanced_wav}")
            except OSError as e:
                print(f"⚠️ 清理临时音频失败：{enhanced_wav}（{e}）")

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
