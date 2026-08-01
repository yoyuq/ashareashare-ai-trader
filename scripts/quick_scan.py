"""
快速全市场扫描 — 避开 Eastmoney, 用 Baostock + Tencent + DeepSeek

用法: python scripts/quick_scan.py
"""

import asyncio, json, os, sys, time
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv; load_dotenv()


async def scan():
    from data.router import get_data_router

    router = get_data_router()
    today = date.today()

    # ── Step 1: 股票列表 ──
    logger.info("Step 1: 获取股票列表...")
    df_all = await router.get_stock_list()
    # Filter: only active, exclude BSE (4xx/8xx/9xx)
    df_all = df_all[df_all["status"] == "1"]
    df_all = df_all[~df_all["symbol"].str.match(r"^(bj\.)?[489]\d{5}")]
    logger.info(f"有效股票: {len(df_all)} 只")

    # ── Step 2: 批量获取实时行情 (Tencent) ──
    logger.info("Step 2: 获取实时行情...")
    symbols = df_all["symbol"].tolist()[:800]  # Top 800 by listing
    quotes = []
    batch_size = 50

    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i+batch_size]
        tc = ",".join(s.replace(".", "") for s in batch)
        try:
            r = requests.get(
                f"https://qt.gtimg.cn/q={tc}",
                timeout=10,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            r.encoding = "gbk"
            for line in r.text.strip().split("\n"):
                if "=" not in line or "~" not in line:
                    continue
                try:
                    fields = line.split("=", 1)[1].strip('"').split("~")
                    if len(fields) < 33:
                        continue
                    name = fields[1]
                    code = fields[2]
                    price = float(fields[3]) if fields[3] else 0
                    prev_close = float(fields[4]) if fields[4] else 0
                    pct = float(fields[32]) if fields[32] else 0
                    vol = float(fields[6]) if len(fields) > 6 and fields[6] else 0
                    amount_wan = float(fields[37]) if len(fields) > 37 and fields[37] else 0
                    amount = amount_wan * 10000  # 万元转元
                    turnover = float(fields[38]) if len(fields) > 38 and fields[38] else 0
                    pe = float(fields[39]) if len(fields) > 39 and fields[39] else 0
                    pb = float(fields[46]) if len(fields) > 46 and fields[46] else 0
                    mv = float(fields[45]) * 1e8 if len(fields) > 45 and fields[45] else 0  # 亿转元

                    if price <= 0 or amount <= 0:
                        continue

                    prefix = "sh" if code.startswith("6") else "sz"
                    quotes.append({
                        "symbol": f"{prefix}.{code}", "code": code, "name": name,
                        "price": price, "prev_close": prev_close, "pct_change": pct,
                        "volume": vol, "amount": amount, "turnover": turnover,
                        "pe_ttm": pe, "pb": pb, "total_mv": mv,
                    })
                except (ValueError, IndexError):
                    continue
        except Exception as e:
            logger.debug(f"Batch {i}: {e}")
            continue

        if (i // batch_size + 1) % 20 == 0:
            logger.info(f"  行情: {len(quotes)}/{i+batch_size}")

    df = pd.DataFrame(quotes)
    logger.info(f"获取行情: {len(df)} 只")

    # ── Step 3: PreScreener ──
    logger.info("Step 3: PreScreener 筛选...")
    from analysis.pre_screener import PreScreener

    # Detect regime
    pct_mean = df["pct_change"].mean()
    up_ratio = (df["pct_change"] > 0).mean()
    if pct_mean < -1.5: regime = "strong_bear"
    elif pct_mean < -0.3: regime = "weak_bear"
    elif pct_mean < 0.3: regime = "range_bound"
    elif pct_mean < 1.5: regime = "weak_bull"
    else: regime = "strong_bull"
    logger.info(f"市场体制: {regime} (均涨{pct_mean:+.2f}% 上涨比{up_ratio:.0%})")

    # Add derived columns PreScreener needs
    df["vol_ratio"] = 1.0
    df["pct_60d"] = df["pct_change"]
    df["amplitude"] = (df["price"] / df["prev_close"]).abs() * 5
    # PB 为 0 时用 PE 和价格估算 (PB ≈ PE * ROE, 假设 ROE=10%)
    df["pb"] = df.apply(lambda r: r["pb"] if r["pb"] > 0 else (r["pe_ttm"] * 0.1 if r["pe_ttm"] > 0 else 2.0), axis=1)

    # 用宽松参数：Tencent数据PE/PB不完整，放宽质量过滤
    screener = PreScreener()
    screener.MAX_PE_SPIKE = 500    # 放宽PE上限
    screener.MIN_PRICE = 1.0       # 放宽最低价
    screener.MIN_MARKET_CAP = 5e8  # 放宽最低市值到5亿
    result = screener.screen(df, regime=regime, top_n=200)
    df_top = result.df
    logger.info(f"PreScreener: {len(df)} -> {len(df_top)} "
                f"score range [{result.score_distribution['min']:.0f}-{result.score_distribution['max']:.0f}]")

    # ── Step 4: DeepSeek Flash 精筛 ──
    logger.info("Step 4: DeepSeek Flash 精筛...")
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        timeout=60.0,
    )

    all_scores = {}
    for i in range(0, min(200, len(df_top)), 25):
        batch = df_top.iloc[i:i+25]
        lines = []
        for _, r in batch.iterrows():
            mv = r.get('total_mv', 0) / 1e8
            lines.append(
                f"{r['code']} {r['name']} price={r['price']:.2f} chg={r['pct_change']:+.1f}% "
                f"PE={r.get('pe_ttm',0):.0f} PB={r.get('pb',0):.1f} MV={mv:.0f}亿 "
                f"换手={r.get('turnover',0):.1f}%"
            )
        prompt = (
            f"评估以下{len(lines)}只A股短期潜力(0-100分)。"
            f"考虑动量、估值、成交量、市值。当前市场: {regime}。"
            f"只返回JSON数组[{{\"code\":\"..\",\"score\":0}}]按score降序:\n" + "\n".join(lines)
        )
        try:
            resp = await client.chat.completions.create(
                model="deepseek-v4-flash",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0, max_tokens=2000,
            )
            content = resp.choices[0].message.content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("```", 1)[0]
            items = json.loads(content)
            if isinstance(items, list):
                for item in items:
                    all_scores[item.get("code", "")] = item.get("score", 50)
            logger.debug(f"Flash batch {i}: {len(items)} scored")
        except Exception as e:
            logger.warning(f"Flash batch {i}: {e}")

    if all_scores:
        df_top = df_top.copy()
        df_top["llm_score"] = df_top["code"].map(lambda c: all_scores.get(str(c), 50))
        df_top["combined"] = df_top["composite_score"] * 0.3 + df_top["llm_score"] * 0.7
        df_top = df_top.nlargest(80, "combined")

    logger.info(f"Flash 精筛后: {len(df_top)} 只")

    # ── Step 5: DeepSeek Pro 深度分析 ──
    logger.info("Step 5: DeepSeek Pro 深度分析...")

    results = []
    for batch_i in range(0, min(80, len(df_top)), 15):
        batch = df_top.iloc[batch_i:batch_i+15]
        lines = []
        for _, r in batch.iterrows():
            mv = r.get('total_mv', 0) / 1e8
            board = "主板"
            code_str = str(r['code'])
            if code_str.startswith('3'): board = "创业板"
            elif code_str.startswith('688'): board = "科创板"
            lines.append(
                f"{code_str} {r['name']} [{board}] price={r['price']:.2f} "
                f"chg={r['pct_change']:+.1f}% PE={r.get('pe_ttm',0):.0f} "
                f"PB={r.get('pb',0):.1f} MV={mv:.0f}亿 换手={r.get('turnover',0):.1f}% "
                f"因子分={r.get('composite_score',50):.0f} Flash分={r.get('llm_score',50):.0f}"
            )

        prompt = (
            f"分析以下{len(lines)}只A股，给出最终评分(0-100)和操作(BUY/HOLD/SELL)。"
            f"当前市场: {regime}。"
            f"评分标准: 85+强烈看多, 70-84偏多, 55-69中性, <55偏空。"
            f"每只包含: final_score, action, conviction, technical, fundamental, risk。"
            f"只返回JSON数组:\n" + "\n".join(lines)
        )

        try:
            resp = await client.chat.completions.create(
                model=os.getenv("DEEPSEEK_FLASH_MODEL", "deepseek-v4-flash"),
                messages=[
                    {"role": "system", "content": "你是资深A股分析师。只返回JSON。"
                     "评分考虑: 技术趋势、估值合理性、市场体制适配度、流动性。"
                     f"当前{regime}市场，震荡市偏好低估值+业绩确定的标的。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3, max_tokens=4000,
            )
            content = resp.choices[0].message.content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("```", 1)[0]
            items = json.loads(content)
            if isinstance(items, list):
                for idx, item in enumerate(items):
                    if not item.get("code") and idx < len(batch):
                        item["code"] = str(batch.iloc[idx]["code"])
                        item["name"] = str(batch.iloc[idx]["name"])
                results.extend(items)
                logger.info(f"Pro batch {batch_i}: {len(items)} analyzed")
        except Exception as e:
            logger.warning(f"Pro batch {batch_i}: {e}")

    results.sort(key=lambda x: x.get("final_score", 0), reverse=True)

    # ── Step 6: PostFilter ──
    logger.info("Step 6: PostFilter 精炼...")
    from analysis.post_filter import PostFilter
    pf = PostFilter()
    passed = pf.filter(results, regime=regime, min_score=65)

    # ── Save ──
    out = {
        "date": today.isoformat(),
        "generated_at": datetime.now().isoformat(),
        "regime": regime,
        "total_scanned": len(df),
        "prescreened": len(result.df),
        "flash_screened": len(df_top) if all_scores else len(result.df),
        "deep_analyzed": len(results),
        "refined": len(passed),
        "results": results,
        "refined_results": passed,
        "filter_log": pf.filter_log,
    }
    outdir = Path(__file__).parent.parent / "reports"
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "deep_analysis_top100.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # ── Print Results ──
    print(f"\n{'='*70}")
    print(f"A股AI扫描结果: {today} | 市场: {regime}")
    print(f"{'='*70}")
    print(f"全市场: {len(df)}只 -> PreScreener: {len(result.df)} -> Flash: {len(df_top)} -> Pro: {len(results)} -> 精炼: {len(passed)}")
    print()

    buys = [r for r in results if r.get("action") == "BUY"]
    print(f"BUY 推荐: {len(buys)} 只 (Pro 原始)")
    print(f"精炼推荐: {len(passed)} 只 (PostFilter 后)")
    print()
    print(f"{'代码':<8} {'名称':<10} {'Pro评分':>6} {'确信度':>5} {'操作':>6} {'逻辑'}")
    print("-" * 70)
    for r in passed[:15]:
        code = r.get("code", "")
        name = r.get("name", "")
        score = r.get("final_score", 0)
        conv = r.get("conviction", 0)
        action = r.get("action", "")
        tech = r.get("technical", "")[:40]
        print(f"{code:<8} {name:<10} {score:>6} {conv:>4.0%} {action:>6}  {tech}")

    print(f"\n过滤 ({len(pf.filter_log)}条):")
    for log in pf.filter_log:
        clean = log.replace("❌", "X").replace("⚡", "!").replace("✅", "OK")
        print(f"  {clean}")

    print(f"\n结果已保存: reports/deep_analysis_top100.json")
    return out


if __name__ == "__main__":
    asyncio.run(scan())
