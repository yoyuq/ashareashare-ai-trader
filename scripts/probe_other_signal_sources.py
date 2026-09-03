"""候选信号源真实可用性探测 (零模拟纪律, 与龙虎榜接入同法).

对每个候选源: 在 2-3 个跨 2015→2026 的日期/标的上真实拉取, 验证
  (a) 是否返回不同数据 (真历史 point-in-time, 还是像新浪龙虎榜那样复用快照=假历史)
  (b) 历史深度能否覆盖需要的回测段
  (c) 字段是否有用.

候选源按与头号策略(冷落低波精选)+目标(熊/震荡跑赢+牛市不崩)对齐度排序:
  S1 北向资金 (外资净买/持仓)     — 牛熊 regime/风险
  S2 两融余额个股 (融资杠杆)      — 2015式杠杆崩 precursor, 牛市不崩
  S3 股东户数 (筹码集中度)        — 冷落/关注度 直接相关选股信号
  S4 业绩预告/快报 (盈利惊喜)     — 基本面地板
  S5 限售解禁 (解禁压力)         — 风险因子
  S6 高管增持/减持 机构持仓       — 机构行为
  S7 大宗交易 (大宗/产业资本)     — 资金行为
  S8 行业/概念板块历史            — 行业配置
  S9 国债收益率/宏观             — 无风险利率 regime
"""
from __future__ import annotations

import hashlib
import os
import sys
import time

for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ.pop(k, None)


def force_utf8():
    for n in ("stdout", "stderr"):
        s = getattr(sys, n, None)
        if hasattr(s, "reconfigure"):
            try:
                s.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def h(df):
    if df is None or len(df) == 0:
        return "EMPTY"
    return hashlib.md5(df.to_csv(index=False).encode()).hexdigest()[:10]


def run(tag, fn, dates=()):
    """fn(d) -> df. 对 dates 各拉一次, hash 对比判真历史/假历史."""
    print(f"\n### {tag}")
    seen = {}
    for d in dates:
        try:
            df = fn(d)
            n = len(df) if df is not None else 0
            hh = h(df)
            flag = "新快照" if hh not in seen else f"同hash={seen[hh]}"
            seen[hh] = str(d)
            cols = list(df.columns)[:10] if n else []
            print(f"    {d}: rows={n} hash={hh} [{flag}] cols={cols}")
            if n and d == dates[0]:
                print(f"        sample: {df.iloc[0].to_dict()}")
        except Exception as e:
            print(f"    {d}: ERR {type(e).__name__} {str(e)[:75]}")
        time.sleep(0.6)


def probe():
    import akshare as ak
    D = ["2024-11-15", "2020-07-15", "2016-04-15"]  # 跨段验证

    # S1 北向资金全序列 (无日期参数, 一拉全史) — point-in-time 天然
    print("\n### S1 北向资金历史 (stock_hsgt_hist_em / stock_hsgt_fund_flow_summary_em)")
    for fname in ("stock_hsgt_hist_em", "stock_hsgt_north_net_flow_in_em"):
        if not hasattr(ak, fname):
            print(f"    无 {fname}"); continue
        try:
            df = getattr(ak, fname)(symbol="北向资金") if fname == "stock_hsgt_hist_em" else getattr(ak, fname)()
            print(f"    {fname}: rows={len(df)} cols={list(df.columns)[:8]} 最早={df.iloc[0].to_dict()}?")
        except Exception as e:
            print(f"    {fname}: ERR {type(e).__name__} {str(e)[:75]}")

    # S2 两融余额个股
    print("\n### S2 融资融券个股余额 (stock_margin_detail_szse/sse, per-date)")
    for fn in ("stock_margin_detail_szse", "stock_margin_detail_sse"):
        if not hasattr(ak, fn):
            print(f"    无 {fn}"); continue
        run(fn, lambda d, f=fn: getattr(ak, f)(date=d.replace("-", "")), D)

    # S3 股东户数 (per-symbol 时序, 无日期参数)
    print("\n### S3 股东户数 (stock_zh_a_gdhs, per-symbol 时序)")
    try:
        df = ak.stock_zh_a_gdhs(symbol="600519")
        print(f"    gdhs(600519): rows={len(df)} cols={list(df.columns)}")
        print(f"        sample: {df.iloc[0].to_dict()}")
    except Exception as e:
        print(f"    gdhs(600519): ERR {type(e).__name__} {str(e)[:75]}")

    # S4 业绩预告 per-date
    print("\n### S4 业绩预告 (stock_yjyg_em, per-date)")
    run(lambda d: ak.stock_yjyg_em(date=d.replace("-", "")), "stock_yjyg_em", D)

    # S5 限售解禁 (按区间)
    print("\n### S5 限售解禁 (stock_restricted_release_detail_em)")
    for fname in ("stock_restricted_release_detail_em", "stock_restricted_release_detail_sina"):
        if not hasattr(ak, fname):
            print(f"    无 {fname}"); continue
        try:
            df = getattr(ak, fname)(start_date="2024-11-11", end_date="2024-11-15")
            print(f"    {fname}: rows={len(df)} cols={list(df.columns)[:8]}")
        except Exception as e:
            print(f"    {fname}: ERR {type(e).__name__} {str(e)[:75]}")

    # S7 大宗交易 (按区间, 东财)
    print("\n### S7 大宗交易 (stock_dzjy_sctj / stock_dzjy_mrmx)")
    for fname, call in [
        ("stock_dzjy_sctj", lambda: ak.stock_dzjy_sctj(start_date="20241111", end_date="20241115")),
        ("stock_dzjy_mrmx", lambda: ak.stock_dzjy_mrmx(start_date="20241111", end_date="20241115")),
    ]:
        if not hasattr(ak, fname):
            print(f"    无 {fname}"); continue
        try:
            df = call()
            print(f"    {fname}: rows={len(df)} cols={list(df.columns)[:8]}")
        except Exception as e:
            print(f"    {fname}: ERR {type(e).__name__} {str(e)[:75]}")

    # S8 行业板块历史
    print("\n### S8 行业板块历史 (stock_board_industry_hist_em)")
    if hasattr(ak, "stock_board_industry_name_em"):
        try:
            names = ak.stock_board_industry_name_em()
            print(f"    行业板块数={len(names)} cols={list(names.columns)[:5]}")
            if len(names):
                board = names.iloc[0]["板块名称"]
        except Exception as e:
            print(f"    board_name_em ERR {type(e).__name__} {str(e)[:60]}")
    try:
        df = ak.stock_board_industry_hist_em(symbol="银行", start_date="20200101", end_date="20201231", period="日k")
        print(f"    行业history(银行): rows={len(df)} cols={list(df.columns)[:8]}")
    except Exception as e:
        print(f"    board_hist ERR {type(e).__name__} {str(e)[:75]}")


def main() -> int:
    force_utf8()
    probe()
    print("\n探测完成 — 据真实返回判可接入性")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())