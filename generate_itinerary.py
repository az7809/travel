"""
generate_itinerary.py — DeepSeek 行程解析引擎

读取 raw_wechat.txt → DeepSeek API (openai SDK) → output_itinerary.json

完全无人交互。自动读取、自动调用、自动校验、自动落盘。
"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from system_prompt import SYSTEM_PROMPT

# ---------------------------------------------------------------------------
# 路径常量
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
RAW_CHAT = HERE / "raw_wechat.txt"
OUTPUT_JSON = HERE / "output_itinerary.json"

# 加载 .env（如果存在），但环境变量优先（Claude Code 注入）
load_dotenv(HERE / ".env", override=True)

# DeepSeek 配置
DEEPSEEK_BASE = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("ANTHROPIC_API_KEY", "")


def main() -> int:
    # 0. Windows 控制台 UTF-8 支持
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    # 1. 校验 API Key
    if not DEEPSEEK_API_KEY:
        print("错误：未设置 DEEPSEEK_API_KEY 或 ANTHROPIC_API_KEY 环境变量。", file=sys.stderr)
        return 1

    # 1. 读取原始聊天记录
    if not RAW_CHAT.exists():
        print(f"错误：找不到 {RAW_CHAT}", file=sys.stderr)
        print("请在当前目录下创建 raw_wechat.txt 并粘贴微信聊天记录。", file=sys.stderr)
        return 1

    raw_text = RAW_CHAT.read_text(encoding="utf-8").strip()
    if not raw_text:
        print("错误：raw_wechat.txt 是空的。", file=sys.stderr)
        return 1

    print(f"读取 raw_wechat.txt：{len(raw_text)} 字符")

    # 2. 初始化 DeepSeek 客户端（openai SDK 格式）
    client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE,
    )

    # 3. 调用 API
    print(f"调用 DeepSeek API ({DEEPSEEK_MODEL}) ...")
    try:
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            temperature=0.3,           # 低温度保证输出稳定
            max_tokens=8192,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": raw_text},
            ],
        )
    except Exception as exc:
        print(f"API 调用失败：{exc}", file=sys.stderr)
        return 1

    # 4. 提取 + 校验 JSON
    content = response.choices[0].message.content
    if not content:
        print("错误：API 返回空内容。", file=sys.stderr)
        return 1

    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        print(f"JSON 解析失败：{exc}", file=sys.stderr)
        print(f"原始返回前 500 字符：\n{content[:500]}", file=sys.stderr)
        return 1

    # 5. 持久化
    OUTPUT_JSON.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"输出已保存：{OUTPUT_JSON}")

    # 6. 简要回显关键信息
    summary = data.get("summary", {})
    warning = data.get("warning")
    security = data.get("security_risk")
    days = summary.get("total_days", "?")

    print(f"行程天数：{days} 天")
    print(f"涉及城市：{', '.join(summary.get('cities', []))}")
    if warning:
        print(f"⚠️  警告：{warning[:120]}...")
    if security:
        print(f"🛡️  安保提醒已生成")

    return 0


if __name__ == "__main__":
    sys.exit(main())
