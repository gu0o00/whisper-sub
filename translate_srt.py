"""
将 SRT 字幕文件翻译成中文。

调用本地 OpenAI 兼容接口（默认 http://127.0.0.1:1234/v1，如 LM Studio），
采用 Hy-MT2 模型推荐的「结构化数据翻译（Structured Data Translation）」策略：
将 SRT 视为结构化数据，仅翻译用户可见的字幕文本，严格保留序号、时间轴与结构。

参考：https://huggingface.co/tencent/Hy-MT2-1.8B
"""

import argparse
import os
import re
import sys
import time
from dataclasses import dataclass

from openai import OpenAI

# ---- SRT 时间轴正则 ---------------------------------------------------------
# 形如 00:01:23,456 --> 00:01:25,789
TIMESTAMP_RE = re.compile(
    r"^\d{2}:\d{2}:\d{2}[.,]\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}[.,]\d{3}\s*$"
)


@dataclass
class SrtEntry:
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


def normalize_target_lang(lang: str) -> str:
    """把语言代码/别名归一化为语言全称。未知值原样返回。"""
    if not lang:
        return lang
    key = lang.strip().lower()
    return _LANG_NAME_MAP.get(key, lang.strip())


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


def build_prompt(batch: list[SrtEntry], target_lang: str) -> str:
    """构造 Hy-MT2 风格的结构化数据翻译 prompt。"""
    source_data = entries_to_srt(batch)
    return (
        f"### Task\n"
        f"将下面 SRT 字幕数据中的「字幕文本」翻译成 {target_lang}。"
        f"SRT 是结构化数据，必须严格保留其结构与所有非文本字段。\n\n"
        f"### Strict Rules\n"
        f"- 必须完整保留 SRT 结构：序号行、时间轴行、空行分隔，与原文完全一致。\n"
        f"- 序号与时间轴（如 00:01:23,456 --> 00:01:25,789）一律不得修改、不得翻译、不得重新编号。\n"
        f"- 仅翻译字幕文本内容，不翻译序号与时间轴。\n"
        f"- 即使原文非常短（如单个语气词），也必须翻译成 {target_lang}，"
        f"严禁原样保留原文或回显源文。\n"
        f"- 不得添加任何解释、注释、前后缀或额外说明，只输出翻译后的 SRT 数据。\n"
        f"- 输入共 {len(batch)} 条字幕，输出必须同样是 {len(batch)} 条，顺序一一对应。\n"
        f"- 保留原文字幕文本中的换行：原文文本有几行，译文文本就对应几行。\n"
        f"- 不要合并或拆分任何条目。\n\n"
        f"### Source Data\n"
        f"{source_data}"
    )


def call_model(
    client: OpenAI, model: str, prompt: str, temperature: float, max_tokens: int
) -> str:
    """调用 OpenAI 兼容接口，返回纯文本响应。"""
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是一个专业的字幕翻译引擎。你只能输出翻译后的 SRT 数据，"
                    "绝不能输出任何解释、Markdown 代码块标记或额外文字。"
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()


def strip_code_fence(text: str) -> str:
    """去掉模型可能误加的 ``` 代码块围栏。"""
    text = text.strip()
    if text.startswith("```"):
        # 去掉首行 ```xxx
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


def is_likely_untranslated(src: str, dst: str) -> bool:
    """粗略判断译文是否其实没翻译（原样回显）。

    判定规则：
    - 归一化（去空白、标点）后完全相同 → 未翻译；
    - 译文里仍含平假名/片假名，且几乎不含中文汉字 → 视为未翻译
      （目标语言是中文时，正常译文应包含中文汉字）。
    """
    norm = lambda s: re.sub(r"[\s\W_]+", "", s)
    if norm(src) == norm(dst):
        return True
    has_kana = bool(re.search(r"[぀-ゟ゠-ヿ]", dst))
    # 目标是中文：译文应含 CJK 汉字（中日韩统一表意文字）
    has_cjk = bool(re.search(r"[一-鿿]", dst))
    if has_kana and not has_cjk:
        return True
    return False


def translate_batch(
    client: OpenAI,
    model: str,
    batch: list[SrtEntry],
    target_lang: str,
    temperature: float,
    max_tokens: int,
    retries: int = 2,
) -> list[str]:
    """翻译一个批次，返回与 batch 等长的译文列表。

    流程：
    1. 整批翻译，校验条目数；不匹配则重试；
    2. 数量匹配后逐条检测「回显/未翻译」，对疑似未翻译的条目单独重译；
    3. 整批仍失败则全部回退为逐条翻译。
    """
    prompt = build_prompt(batch, target_lang)
    texts: list[str] | None = None
    for attempt in range(retries + 1):
        try:
            raw = call_model(client, model, prompt, temperature, max_tokens)
            raw = strip_code_fence(raw)
            parsed = parse_srt(raw)
            if len(parsed) == len(batch):
                texts = [e.text for e in parsed]
                break
            print(
                f"   ⚠️  条目数不匹配（期望 {len(batch)}，得到 {len(parsed)}），"
                f"重试 {attempt + 1}/{retries + 1}"
            )
        except Exception as e:  # noqa: BLE001
            print(f"   ⚠️  调用失败：{e}，重试 {attempt + 1}/{retries + 1}")
            time.sleep(1)

    if texts is None:
        # 整批失败：全部回退为逐条翻译
        print("   ↩️  批次翻译失败，回退为逐条翻译")
        return [translate_single(client, model, e.text, target_lang, temperature)
                for e in batch]

    # 回显检测：对疑似未翻译的条目单独重译
    fixed = 0
    for i, (src, dst) in enumerate(zip(batch, texts)):
        if is_likely_untranslated(src.text, dst):
            retry = translate_single(client, model, src.text, target_lang, temperature)
            if not is_likely_untranslated(src.text, retry):
                texts[i] = retry
                fixed += 1
            else:
                # 单条重译仍回显，保留单条结果（至少不混入整批失败）
                texts[i] = retry
    if fixed:
        print(f"   🔁 检测到 {fixed} 条疑似未翻译，已单独重译")
    return texts


def translate_single(
    client: OpenAI,
    model: str,
    text: str,
    target_lang: str,
    temperature: float,
) -> str:
    """单条文本翻译回退路径。"""
    prompt = (
        f"Translate the following text into {target_lang}. "
        f"Note that you should only output the translated result "
        f"without any additional explanation:\n\n{text}"
    )
    try:
        raw = call_model(client, model, prompt, temperature, max_tokens=512)
        return strip_code_fence(raw)
    except Exception as e:  # noqa: BLE001
        print(f"   ❌ 单条翻译失败，保留原文：{e}")
        return text


def make_batches(
    entries: list[SrtEntry], batch_size: int, max_chars: int
) -> list[list[SrtEntry]]:
    """按条数与字符数双约束分批。"""
    batches: list[list[SrtEntry]] = []
    cur: list[SrtEntry] = []
    cur_chars = 0
    for e in entries:
        e_chars = len(e.text) + len(e.timestamp) + 8
        if cur and (len(cur) >= batch_size or cur_chars + e_chars > max_chars):
            batches.append(cur)
            cur, cur_chars = [], 0
        cur.append(e)
        cur_chars += e_chars
    if cur:
        batches.append(cur)
    return batches


def main():
    parser = argparse.ArgumentParser(description="将 SRT 字幕翻译成中文（本地 OpenAI 兼容接口）")
    parser.add_argument("input", type=str, help="输入 SRT 文件路径")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="输出文件路径（默认在原文件名后加 .zh）")
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:1234/v1",
                        help="OpenAI 兼容接口地址（默认 http://127.0.0.1:1234/v1）")
    parser.add_argument("--api-key", type=str, default="lm-studio",
                        help="API Key（本地服务通常随意，默认 lm-studio）")
    parser.add_argument("-m", "--model", type=str, default="hy-mt2-1.8b",
                        help="模型名称（需与本地服务加载的模型 ID 一致）")
    parser.add_argument("--target-lang", type=str, default="中文",
                        help="目标语言（默认：中文）")
    parser.add_argument("--batch-size", type=int, default=10,
                        help="每批最大条目数（默认：10，小模型不宜过大）")
    parser.add_argument("--max-chars", type=int, default=4000,
                        help="每批最大字符数（默认：4000）")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="采样温度（默认：0.1，偏低以保持稳定）")
    parser.add_argument("--max-tokens", type=int, default=4096,
                        help="每批响应最大 token 数（默认：4096）")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"❌ 输入文件不存在：{args.input}")
        sys.exit(1)

    # 输出路径
    if args.output:
        output = args.output
    else:
        root, ext = os.path.splitext(args.input)
        output = f"{root}.zh{ext or '.srt'}"

    # 读取并解析
    with open(args.input, "r", encoding="utf-8") as f:
        content = f.read()
    entries = parse_srt(content)
    if not entries:
        print(f"❌ 未能解析出任何字幕条目：{args.input}")
        sys.exit(1)

    print(f"📄 输入：{args.input}")
    print(f"📝 输出：{output}")
    print(f"🌐 目标语言：{args.target_lang}")
    print(f"🤖 接口：{args.base_url}  模型：{args.model}")
    print(f"📊 共 {len(entries)} 条字幕")

    # 归一化目标语言：允许传 zh / en / ja 等代码，内部映射为语言全称
    target_lang = normalize_target_lang(args.target_lang)
    if target_lang != args.target_lang:
        print(f"↔️  目标语言归一化：{args.target_lang} → {target_lang}")

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    batches = make_batches(entries, args.batch_size, args.max_chars)
    print(f"📦 分为 {len(batches)} 个批次（每批 ≤ {args.batch_size} 条 / {args.max_chars} 字符）")

    translated: list[SrtEntry] = []
    for bi, batch in enumerate(batches, start=1):
        print(f"⏳ 翻译批次 {bi}/{len(batches)}（{len(batch)} 条）...")
        texts = translate_batch(
            client, args.model, batch, target_lang,
            args.temperature, args.max_tokens,
        )
        for entry, text in zip(batch, texts):
            translated.append(
                SrtEntry(index=entry.index, timestamp=entry.timestamp, text=text)
            )

    # 写出
    with open(output, "w", encoding="utf-8") as f:
        f.write(entries_to_srt(translated))

    print(f"✅ 完成！共翻译 {len(translated)} 条字幕 → {output}")


if __name__ == "__main__":
    main()
