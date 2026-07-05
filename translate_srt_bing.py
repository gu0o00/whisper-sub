"""
使用 Microsoft Translator (Bing) API 批量翻译 SRT 字幕文件。

命令行直接调用或作为 MCP 工具被 import 调用均可。
"""
import argparse
import os
import sys
import uuid
import time
import requests

from srt_utils import (
    SrtEntry,
    entries_to_srt,
    is_lang_code,
    lang_to_code,
    parse_srt,
)

# ---- 配置 ---------------------------------------------------------
KEY = "BING_TRANSLATOR_API_KEY" 
ENDPOINT = "https://api.cognitive.microsofttranslator.com"
LOCATION = "eastasia"

# API 限制：每请求最多 100 条，总字符数不超过 50000
MAX_ITEMS_PER_REQUEST = 100
MAX_CHARS_PER_REQUEST = 50000


def batch_entries(entries: list[SrtEntry]) -> list[list[SrtEntry]]:
    """按 API 限制分批（100条/批，50000字符/批）。"""
    batches: list[list[SrtEntry]] = []
    cur: list[SrtEntry] = []
    cur_chars = 0
    for e in entries:
        e_chars = len(e.text)
        if cur and (len(cur) >= MAX_ITEMS_PER_REQUEST or cur_chars + e_chars > MAX_CHARS_PER_REQUEST):
            batches.append(cur)
            cur, cur_chars = [], 0
        cur.append(e)
        cur_chars += e_chars
    if cur:
        batches.append(cur)
    return batches


def translate_batch(batch: list[SrtEntry], source_lang: str, target_lang: str) -> list[str]:
    """向 Bing API 提交一批文本，返回译文列表。"""
    params = {
        "api-version": "3.0",
        "from": source_lang,
        "to": [target_lang],
    }
    headers = {
        "Ocp-Apim-Subscription-Key": KEY,
        "Ocp-Apim-Subscription-Region": LOCATION,
        "Content-type": "application/json",
        "X-ClientTraceId": str(uuid.uuid4()),
    }
    body = [{"text": e.text} for e in batch]

    url = ENDPOINT + "/translate"
    response = requests.post(url, params=params, headers=headers, json=body)

    if response.status_code != 200:
        print(f"   [ERROR] API status {response.status_code}: {response.text[:500]}")
        return [e.text for e in batch]  # 回退为原文

    result = response.json()
    translations = []
    for item in result:
        if "translations" in item and len(item["translations"]) > 0:
            translations.append(item["translations"][0]["text"])
        else:
            translations.append(item.get("text", ""))  # 回退原文
    return translations


def make_output_path(input_file: str, output: str | None, target_lang: str) -> str:
    """根据输入文件路径和目标语言生成输出文件路径。

    规则（与 translate_srt_ai 一致）：
      foo.ja.srt  → foo.zh.srt（替换已存在的语言段）
      foo.srt     → foo.zh.srt（无语言段则追加）
    """
    if output:
        return output
    root, ext = os.path.splitext(input_file)
    ext = ext or ".srt"
    target_code = lang_to_code(target_lang)
    parts = root.split(".")
    if len(parts) > 1 and is_lang_code(parts[-1]):
        parts[-1] = target_code
        new_root = ".".join(parts)
    else:
        new_root = f"{root}.{target_code}"
    return new_root + ext


def translate_srt_via_bing(
    srt_path: str,
    output: str | None = None,
    source_lang: str = "ja",
    target_lang: str = "zh-Hans",
    progress=None,
) -> tuple[str, int]:
    """核心翻译函数，供 CLI 和 MCP 共用。

    Args:
        srt_path: 输入 SRT 文件路径
        output: 输出文件路径（可选，默认自动生成）
        source_lang: 源语言代码（Bing API 格式，如 ja、en）
        target_lang: 目标语言代码（Bing API 格式，如 zh-Hans）
        progress: 进度回调 progress(value, total, message=None)

    Returns:
        (output_path, translated_count)
    """
    if not os.path.isfile(srt_path):
        raise FileNotFoundError(f"字幕文件不存在：{srt_path}")

    out_path = make_output_path(srt_path, output, target_lang)

    with open(srt_path, "r", encoding="utf-8") as f:
        content = f.read()
    entries = parse_srt(content)
    if not entries:
        raise ValueError(f"未能解析出任何字幕条目：{srt_path}")

    batches = batch_entries(entries)
    total_batches = len(batches)

    translated_entries: list[SrtEntry] = []
    for bi, batch in enumerate(batches, start=1):
        if progress:
            progress(bi - 1, total_batches, f"开始翻译批次 {bi}/{total_batches}（{len(batch)} 条）")
        print(f"--- Batch {bi}/{total_batches} ({len(batch)} items)...", end=" ", flush=True)
        start = time.time()
        texts = translate_batch(batch, source_lang, target_lang)
        elapsed = time.time() - start
        for entry, text in zip(batch, texts):
            translated_entries.append(
                SrtEntry(index=entry.index, timestamp=entry.timestamp, text=text)
            )
        print(f"DONE ({elapsed:.1f}s)")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(entries_to_srt(translated_entries))

    if progress and total_batches:
        progress(total_batches, total_batches, f"翻译完成，共 {len(translated_entries)} 条")

    return out_path, len(translated_entries)


def main():
    parser = argparse.ArgumentParser(description="使用 Microsoft Translator (Bing) API 将 SRT 字幕翻译成中文")
    parser.add_argument("input", type=str, help="输入 SRT 文件路径")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="输出文件路径（默认自动生成，如 ofes-043.zh.srt）")
    parser.add_argument("--from-lang", type=str, default="ja",
                        help="源语言代码，Bing API 格式（默认 ja）")
    parser.add_argument("--to-lang", type=str, default="zh-Hans",
                        help="目标语言代码，Bing API 格式（默认 zh-Hans）")
    args = parser.parse_args()

    print(f"=== Input:  {args.input}")
    print(f"=== Source: {args.from_lang} -> Target: {args.to_lang}")

    try:
        out_path, count = translate_srt_via_bing(
            srt_path=args.input,
            output=args.output,
            source_lang=args.from_lang,
            target_lang=args.to_lang,
        )
        print(f"\n=== DONE! Translated {count} entries -> {out_path}")
    except (FileNotFoundError, ValueError) as e:
        print(f"[ERROR] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
