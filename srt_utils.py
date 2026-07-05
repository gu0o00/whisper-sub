"""
SRT 字幕文件公共基础模块。

提供 SRT 解析、序列化、语言代码映射等所有翻译脚本共用的工具函数。
"""
import re
from dataclasses import dataclass

# ---- SRT 时间轴正则 ---------------------------------------------------------
# 形如 00:01:23,456 --> 00:01:25,789
TIMESTAMP_RE = re.compile(
    r"^\d{2}:\d{2}:\d{2}[.,]\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}[.,]\d{3}\s*$"
)


@dataclass
class SrtEntry:
    """SRT 字幕条目"""
    index: int          # 原始序号
    timestamp: str      # 时间轴行，原样保留
    text: str           # 字幕文本（可能含换行）


# 目标语言归一化：允许调用方传语言代码（zh / en / ja ...）或别名，
# 内部统一映射成小模型更易理解的语言全称，避免 "翻译成 zh" 这类
# 小模型识别不稳、容易漂移到英语的情况。
_LANG_NAME_MAP = {
    "zh": "中文", "chinese": "中文", "中文": "中文", "chs": "中文", "简体中文": "中文",
    "en": "English", "english": "English",
    "ja": "日语", "jp": "日语", "japanese": "日语", "日文": "日语",
    "ko": "韩语", "kor": "韩语", "korean": "韩语", "韩文": "韩语",
}

# 语言别名 → 短代码，用于生成输出文件名中的语言段（如 .zh）
_LANG_CODE_MAP = {
    "zh": "zh", "chinese": "zh", "中文": "zh", "chs": "zh", "简体中文": "zh", "简体": "zh",
    "en": "en", "english": "en", "英文": "en", "英语": "en",
    "ja": "ja", "jp": "ja", "japanese": "ja", "日语": "ja", "日文": "ja",
    "ko": "ko", "kor": "ko", "korean": "ko", "韩语": "ko", "韩文": "ko",
}


def normalize_target_lang(lang: str) -> str:
    """把语言代码/别名归一化为语言全称。未知值原样返回。"""
    if not lang:
        return lang
    key = lang.strip().lower()
    return _LANG_NAME_MAP.get(key, lang.strip())


def lang_to_code(lang: str) -> str:
    """把语言代码/别名归一化为短代码（用于文件名）。未知值原样小写返回。

    支持 BCP-47 格式（如 zh-Hans、en-US），提取主语言部分作为短代码。
    """
    if not lang:
        return lang
    key = lang.strip().lower()
    if key in _LANG_CODE_MAP:
        return _LANG_CODE_MAP[key]
    # Bing API 使用 BCP-47 格式（zh-Hans, en-US 等），尝试提取连字符前部分
    if "-" in key:
        primary = key.split("-")[0]
        if primary in _LANG_CODE_MAP:
            return _LANG_CODE_MAP[primary]
    return key


def is_lang_code(s: str) -> bool:
    """判断字符串是否是已知的语言代码/别名（用于识别文件名里的语言段）。

    支持 BCP-47 格式的主语言部分（如 zh-Hans → zh）。
    """
    if not s:
        return False
    key = s.strip().lower()
    if key in _LANG_CODE_MAP:
        return True
    if "-" in key:
        return key.split("-")[0] in _LANG_CODE_MAP
    return False


def parse_srt(content: str) -> list[SrtEntry]:
    """解析 SRT 文本为条目列表。容忍 BOM、CRLF 与多余空行。"""
    content = content.lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n")
    blocks = re.split(r"\n\s*\n", content.strip())
    entries: list[SrtEntry] = []
    for block in blocks:
        lines = [ln for ln in block.split("\n") if ln.strip() != ""]
        if len(lines) < 2:
            continue
        # 第一行是序号
        try:
            index = int(lines[0].strip())
        except ValueError:
            # 容错：缺少序号时用当前计数
            index = len(entries) + 1
        # 找到时间轴行
        ts_idx = next(
            (i for i, ln in enumerate(lines) if TIMESTAMP_RE.match(ln.strip())),
            None,
        )
        if ts_idx is None:
            continue
        timestamp = lines[ts_idx].strip()
        text = "\n".join(lines[ts_idx + 1:]).strip()
        entries.append(SrtEntry(index=index, timestamp=timestamp, text=text))
    return entries


def entries_to_srt(entries: list[SrtEntry]) -> str:
    """把条目列表序列化回 SRT 文本。"""
    parts = []
    for i, e in enumerate(entries, start=1):
        parts.append(f"{i}\n{e.timestamp}\n{e.text}\n")
    return "\n".join(parts)
