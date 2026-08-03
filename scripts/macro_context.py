"""宏观/政策/国际形势分析 — 拉取近期新闻 → LLM 结构化宏观背景 → 缓存供分析注入

用法:
  python scripts/macro_context.py                     # 尝试 AKShare 新闻 + reports/macro_news.md
  python scripts/macro_context.py --news "昨日新闻..." # 手动/联网提供新闻文本
  python scripts/macro_context.py --force-llm          # 强制重新跑 LLM

输出: knowledge/macro_context_latest.json (含 composite_macro_score / recommendation)
被 daily_thinking.py 读取注入 thinking 分析。
"""

import argparse
import asyncio
import json
import os
import re
import sys
from datetime import date
from pathlib import Path

import requests
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

OUT_PATH = Path("knowledge/macro_context_latest.json")


def _macro_system_prompt() -> str:
    txt = Path("knowledge/prompts/system/macro_analyst.txt").read_text(encoding="utf-8")
    return re.sub(r"^---.*?---", "", txt, flags=re.S).strip()


def fetch_news(days: int = 7) -> str:
    """拉取近期 A 股相关新闻 (AKShare 尝试, 失败返回空; reports/macro_news.md 兜底)"""
    parts = []
    # 1. AKShare 央视新闻 (需代理, 失败跳过)
    try:
        import akshare as ak
        df = ak.news_cctv(date=(date.today()).strftime("%Y%m%d"))
        if df is not None and not df.empty and "title" in df.columns:
            for _, r in df.head(10).iterrows():
                parts.append(f"- {r.get('title', '')}")
    except Exception as e:
        logger.debug(f"AKShare news 失败: {e}")
    # 2. reports/macro_news.md 兜底 (Claude 联网或手动更新)
    md = Path("reports/macro_news.md")
    if md.exists():
        parts.append(md.read_text(encoding="utf-8").strip())
    return "\n".join(parts) if parts else ""


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--news", type=str, default="", help="直接提供新闻文本")
    ap.add_argument("--force-llm", action="store_true", help="强制重新跑 LLM (跳过缓存)")
    args = ap.parse_args()

    news = args.news.strip() or fetch_news()
    if not news:
        logger.warning("无新闻输入 (AKShare 失败且无 reports/macro_news.md)")
        news = "最近一周无显著政策/国际事件。"

    # 缓存: 若今天已生成且未强制, 直接读
    if OUT_PATH.exists() and not args.force_llm:
        cached = json.loads(OUT_PATH.read_text(encoding="utf-8"))
        if cached.get("date") == date.today().isoformat():
            print(json.dumps(cached, ensure_ascii=False, indent=2))
            return

    from openai import AsyncOpenAI
    client = AsyncOpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        timeout=120.0,
    )

    sys_prompt = _macro_system_prompt()
    user_content = (
        f"基于以下最近一周宏观/政策/国际新闻, 输出结构化宏观背景分析"
        f"(宏观环境/政策事件/行业轮动/对A股影响方向/风险):\n{news}"
    )
    resp = await client.chat.completions.create(
        model=os.getenv("DEEPSEEK_FLASH_MODEL", "deepseek-v4-flash"),
        messages=[{"role": "system", "content": sys_prompt},
                  {"role": "user", "content": user_content}],
        temperature=0.3, max_tokens=4000,
        extra_body={"thinking": {"type": "disabled"}},
    )
    content = (resp.choices[0].message.content or "").strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1].rsplit("```", 1)[0]

    # 解析 LLM 输出 (可能是 JSON 或文本)
    ctx = {}
    try:
        ctx = json.loads(content)
    except json.JSONDecodeError:
        ctx = {"raw_analysis": content,
               "composite_macro_score": 50,
               "recommendation": content[:800]}

    ctx["date"] = date.today().isoformat()
    ctx["news_sources"] = "akshare+cache" if not args.news else "manual"
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(ctx, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(ctx, ensure_ascii=False, indent=2))
    logger.info(f"宏观背景已缓存: {OUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
