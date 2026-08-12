# -*- coding: utf-8 -*-
"""飞书历史极端压力测试 — 第三张王牌: 真实历史K线回测议会信号在极端行情下的存活.

核心: 用真实历史K线(腾讯接口, 平台实测可访问) → 可解释的"议会规则信号"生成动态仓位
      → 回算净值, 对比【买入持有】, 展示在 股灾/熊市/疫情/流动性危机 中的回撤管控.

为什么是"议会信号": 不能离线一批批调LLM, 用一套可解释的规则近似议会裁决:
  趋势(MA20 vs MA60)判断方向 → 波动率判断温和/剧烈 → 映射风险等级1-5 → 对应仓位
  + 恐慌减仓(单日大跌). 默认参数不tune(防过拟合, 呼应系统"诚实"卖点).

用法: 替换飞书一个代码节点,
  入参 arg1 = scenario(2015/2018/2020/2024)
        arg2 = 平台「股票历史行情」工具输出(可选, 优先使用; 缺省则腾讯联网兜底)
  出参 report_html.

数据源优先级: 平台历史行情工具 > 腾讯联网(qfq). 两源都失败则报错(绝不模拟).
"""
import json
import math


def main(arg1: str, arg2: str = "") -> dict:
    import urllib.request

    # ===== 场景 → 真实历史窗口 =====
    SCENARIOS = {
        "2015": {"code": "sh000001", "name": "2015 股灾", "start": "2015-01-01", "end": "2016-06-30",
                 "desc": "2015年6月杠杆牛崩塌, 上证从5178点三个月跌至2850, 千股跌停频现"},
        "2018": {"code": "sh000001", "name": "2018 熊市", "start": "2018-01-01", "end": "2018-12-31",
                 "desc": "全年单边阴跌, 贸易摩擦+去杠杆, 上证跌约25%"},
        "2020": {"code": "sh000001", "name": "2020 疫情冲击", "start": "2019-12-01", "end": "2020-06-30",
                 "desc": "2020年2月新冠冲击, 春节后首日暴跌, 多国熔断"},
        "2024": {"code": "sh000001", "name": "2024 微盘流动性危机", "start": "2024-01-01", "end": "2024-03-01",
                 "desc": "2024年1-2月微盘股流动性踩踏, 上证也受拖累调整"},
    }
    sc = SCENARIOS.get(arg1, SCENARIOS["2015"])

    # ===== 解析平台「股票历史行情」工具输出 → (closes, dates) =====
    # 容错: 支持 行式[{"date","open","close","high","low","volume"}...] /
    #        列式({"date":[...], "close":[...]}) / 嵌套在 data/kline/daily/bars 键下.
    def _parse_platform_kline(s):
        s = (s or "").strip()
        if not s:
            return None
        if s.startswith("```"):
            s = s.strip("`")
            if s.startswith("json"):
                s = s[4:].strip()
        try:
            obj = json.loads(s)
        except Exception:
            return None

        def _num(x):
            if isinstance(x, bool):
                return None
            if isinstance(x, (int, float)):
                return float(x)
            if isinstance(x, str):
                try:
                    return float(x.replace("%", "").replace(",", "").strip())
                except ValueError:
                    return None
            return None

        date_keys = ("date", "time", "day", "datetime", "trade_date", "日期", "交易日期")
        close_keys = ("close", "c", "收盘", "收盘价", "close_price")
        open_keys = ("open", "o", "开盘", "开盘价")
        high_keys = ("high", "h", "最高", "最高价")
        low_keys = ("low", "l", "最低", "最低价")

        # 收集所有含 日期+收盘 的行对象
        rows = {}

        def _collect(v):
            if isinstance(v, dict):
                # 单条K线: 同时含日期和收盘字段
                d = next((v[k] for k in v if any(kk in k.lower() for kk in date_keys)), None)
                c = next((v[k] for k in v if any(kk in k.lower() for kk in close_keys)), None)
                if d is not None and c is not None:
                    dd, cc = str(d), _num(c)
                    if cc is not None:
                        rows[dd] = cc
                # 列式: 每个字段是数组
                close_arr = next((v[k] for k in v if any(kk in k.lower() for kk in close_keys)
                                  and isinstance(v[k], list)), None)
                date_arr = next((v[k] for k in v if any(kk in k.lower() for kk in date_keys)
                                 and isinstance(v[k], list)), None)
                if close_arr and date_arr and len(close_arr) == len(date_arr):
                    for i, x in enumerate(close_arr):
                        cn = _num(x)
                        if cn is not None:
                            rows[str(date_arr[i])] = cn
                # 递归进子键
                for k, val in v.items():
                    if isinstance(val, (dict, list)):
                        _collect(val)
            elif isinstance(v, list):
                for x in v:
                    _collect(x)

        _collect(obj)
        if len(rows) < 30:
            return None
        # 按日期字符串排序(平台通常 ISO 格式, 字典序即时间序)
        sdates = sorted(rows.keys())
        closes = [rows[k] for k in sdates]
        return closes, sdates

    # ===== 数据源优先级: 平台工具 > 腾讯(qfq). 两源失败则报错(绝不模拟) =====
    closes, dates = [], []
    real = False
    data_source = ""

    # 1) 平台「股票历史行情」工具输出
    platform = _parse_platform_kline(arg2)
    if platform:
        closes, dates = platform
        real = True
        data_source = "平台股票历史行情工具"

    # ===== 腾讯联网兜底 =====
    def fetch_close(code, start, end):
        url = ("https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?"
               f"param={code},day,{start},{end},500,qfq")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        days = data["data"][code]["day"] or data["data"][code].get("qfqday", [])
        closes, dates = [], []
        for row in days:
            try:
                closes.append(float(row[2]))   # row: [date, open, close, high, low, ...]
                dates.append(row[0])
            except (TypeError, ValueError, IndexError):
                continue
        return closes, dates

    # 2) 腾讯联网兜底 (仅当未拿到平台数据)
    if not real:
        try:
            closes, dates = fetch_close(sc["code"], sc["start"], sc["end"])
            if len(closes) >= 30:
                real = True
                data_source = "真实上证日K(腾讯)"
        except Exception:
            closes, dates = [], []

    # 3) 真实数据都拿不到 → 诚实报错，绝不降级模拟
    if not real:
        raise RuntimeError(
            f"压力测试[{sc['name']}]拿不到真实历史K线（平台历史行情工具与腾讯均失败/数据不足30条）。"
            "按项目原则'一点模拟都不用有'，拒绝用模拟数据，请检查数据源。")

    # ===== 可解释的议会规则信号 =====
    # 趋势: MA20 vs MA60 → 方向; 波动: 近20日std → 温和/剧烈
    def ma(seq, w):
        if len(seq) < w:
            return seq[-1] if seq else 0
        return sum(seq[-w:]) / w

    def signal(closes, i):
        if i < 20:
            return 3, 0.8
        ma20 = ma(closes[:i + 1], 20)
        ma60 = ma(closes[:i + 1], 60)
        rets = [(closes[k] - closes[k - 1]) / closes[k - 1] for k in range(max(1, i - 19), i + 1)]
        vol = (sum(r * r for r in rets) / max(len(rets), 1)) ** 0.5 if rets else 0
        _trend = 1 if ma20 > ma60 else -1
        _vol_high = vol > 0.02
        # 风险等级 1-5 (可解释映射)
        if _trend > 0 and not _vol_high:
            risk = 1
        elif _trend > 0:
            risk = 2
        elif _vol_high and closes[i] < ma60:
            risk = 5
        elif _vol_high:
            risk = 4
        else:
            risk = 3
        # 仓位映射
        pos = {1: 1.3, 2: 1.1, 3: 0.8, 4: 0.5, 5: 0.2}[risk]
        # 恐慌日: 单日跌 > 4% → 次日减半
        if i > 0 and (closes[i] - closes[i - 1]) / closes[i - 1] < -0.04:
            pos *= 0.5
        return risk, pos

    # ===== 回算净值 =====
    ai_nav, bh_nav = [1.0], [1.0]
    risks = []
    for i in range(1, len(closes)):
        ret = (closes[i] - closes[i - 1]) / closes[i - 1]
        risk, pos = signal(closes, i)
        risks.append(risk)
        ai_nav.append(ai_nav[-1] * (1 + ret * pos))
        bh_nav.append(bh_nav[-1] * (1 + ret))

    def drawdown(nav):
        peak = nav[0]
        max_dd = 0
        for v in nav:
            peak = max(peak, v)
            dd = (v - peak) / peak
            max_dd = min(max_dd, dd)
        return max_dd

    ai_ret = (ai_nav[-1] - 1) * 100
    bh_ret = (bh_nav[-1] - 1) * 100
    ai_dd = drawdown(ai_nav) * 100
    bh_dd = drawdown(bh_nav) * 100

    # ===== SVG 净值曲线 =====
    W, H = 760, 220
    n = len(ai_nav)
    def pol(nav, color):
        pts = []
        for i, v in enumerate(nav):
            x = i * W / max(n - 1, 1)
            y = H - (v - min(min(ai_nav), min(bh_nav))) / (
                max(max(ai_nav), max(bh_nav)) - min(min(ai_nav), min(bh_nav)) or 1) * (H - 30) - 15
            pts.append(f"{x:.0f},{y:.0f}")
        return f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="2"/>'
    lo = min(min(ai_nav), min(bh_nav))
    hi = max(max(ai_nav), max(bh_nav))
    span = (hi - lo) or 1
    def pol2(nav, color):
        pts = []
        for i, v in enumerate(nav):
            x = i * W / max(n - 1, 1)
            y = H - (v - lo) / span * (H - 30) - 15
            pts.append(f"{x:.0f},{y:.0f}")
        return f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="2"/>'

    # ===== 风险等级色带 =====
    RC = {1: "#3fb950", 2: "#56d364", 3: "#d29922", 4: "#f0883e", 5: "#f85149"}
    risk_band = "".join(
        f'<div style="flex:1;height:10px;background:{RC.get(r,"#30363d")};"></div>' for r in risks)

    # ===== 结论文字 (用绝对值比较回撤) =====
    if abs(ai_dd) < abs(bh_dd):
        verdict = (f"议会信号回撤 {abs(ai_dd):.1f}% vs 买入持有 {abs(bh_dd):.1f}% — "
                   f"回撤少了 {abs(bh_dd)-abs(ai_dd):.1f}pp, 在极端行情中靠动态降仓保护了本金")
    else:
        verdict = f"本段议会信号未优于买入持有 (回撤 {abs(ai_dd):.1f}% vs {abs(bh_dd):.1f}%) — 诚实披露, 不美化"

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="UTF-8">
<title>压力测试 · {sc["name"]}</title><style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;background:#0d1117;color:#c9d1d9;padding:20px;line-height:1.6}}
.container{{max-width:1000px;margin:0 auto}}
.header{{background:linear-gradient(135deg,#161b22,#f8514922);border:1px solid #30363d;border-radius:12px;padding:26px;margin-bottom:18px}}
.header h1{{font-size:24px;color:#f0f6fc}}
.header .desc{{color:#8b949e;font-size:13px;margin-top:8px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin-bottom:18px}}
.cell{{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:14px;text-align:center}}
.cell .v{{font-size:22px;font-weight:700}}
.cell .l{{font-size:11px;color:#8b949e;margin-top:4px}}
.cell.ai .v{{color:#58a6ff}}.cell.bh .v{{color:#f0883e}}
.card{{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:18px;margin-bottom:16px}}
.card-title{{font-size:15px;color:#f0f6fc;font-weight:600;margin-bottom:12px;padding-bottom:8px;border-bottom:1px solid #21262d}}
.verdict{{padding:14px;border-radius:8px;background:#21262d;border-left:3px solid #58a6ff;font-size:14px;color:#c9d1d9}}
.legend{{font-size:11px;color:#8b949e;margin-top:6px}}
</style></head><body><div class="container">
<div class="header">
<h1>🧯 历史极端压力测试 · {sc["name"]}</h1>
<div class="desc">{sc["desc"]}</div>
<div class="desc" style="color:#484f58">{dates[0] if dates else ""} → {dates[-1] if dates else ""} · 数据: {data_source}</div>
</div>
<div class="grid">
<div class="cell ai"><div class="v">{ai_ret:+.1f}%</div><div class="l">议会信号收益</div></div>
<div class="cell bh"><div class="v">{bh_ret:+.1f}%</div><div class="l">买入持有收益</div></div>
<div class="cell ai"><div class="v">{ai_dd:.1f}%</div><div class="l">议会最大回撤</div></div>
<div class="cell bh"><div class="v">{bh_dd:.1f}%</div><div class="l">持有最大回撤</div></div>
</div>
<div class="card">
<div class="card-title">📉 净值对比（议会信号 vs 买入持有）</div>
<svg width="100%" viewBox="0 0 {W} {H}" preserveAspectRatio="none">
<line x1="0" y1="{H-15}" x2="{W}" y2="{H-15}" stroke="#30363d"/>
{pol2(bh_nav,"#f0883e")}
{pol2(ai_nav,"#58a6ff")}
</svg>
<div class="legend"><span style="color:#58a6ff">━ 议会信号</span> · <span style="color:#f0883e">━ 买入持有</span></div>
</div>
<div class="card">
<div class="card-title">🚦 议会风险等级演变（1积极→5避险）</div>
<div style="display:flex;gap:1px;">{risk_band}</div>
</div>
<div class="card">
<div class="card-title">📋 结论</div>
<div class="verdict">{verdict}</div>
</div>
<div style="text-align:center;color:#484f58;font-size:11px;padding:12px">
⚠️ 规则信号为议会裁决的可解释近似(MA趋势+波动+恐慌减仓), 未tune参数; 非实盘业绩, 仅供演示与研究</div>
</div></body></html>"""

    return {"report_html": html}


if __name__ == "__main__":
    import json as _json
    src = open(__file__, encoding="utf-8").read()
    ns = {}
    exec(src, ns)

    # 1) 平台历史行情工具输出 (行式) — 模拟阶段内真实路径
    rows = []
    c = 3000.0
    for i in range(120):
        c *= 1 + (0.001 if i < 60 else -0.004)
        rows.append({"date": f"2020-{i//28+1:02d}-{i%28+1:02d}", "open": round(c*0.99, 2),
                     "close": round(c, 2), "high": round(c*1.01, 2), "low": round(c*0.98, 2),
                     "volume": 1000000 + i})
    platform_out = _json.dumps({"data": rows}, ensure_ascii=False)
    html = ns["main"]("2015", platform_out)["report_html"]
    assert "平台股票历史行情工具" in html, "平台数据源未生效"
    print(f"平台数据源: 生效 (HTML含'平台股票历史行情工具')")

    # 2) 列式格式兜底
    col_out = _json.dumps({"date": [r["date"] for r in rows],
                           "close": [r["close"] for r in rows]}, ensure_ascii=False)
    html2 = ns["main"]("2018", col_out)["report_html"]
    assert "平台股票历史行情工具" in html2, "列式格式未识别"
    print(f"列式数据源: 生效")

    # 3) 不传 arg2 → 腾讯真拉 (本机有网, 真实数据)
    for sc in ("2015", "2018", "2020", "2024"):
        try:
            html3 = ns["main"](sc)["report_html"]
            src_label = "腾讯" if "腾讯" in html3 else "?"
            print(f"{sc}(无arg2): HTML长度={len(html3)} 数据源={src_label}")
        except Exception as e:
            print(f"{sc}: 失败 {type(e).__name__}: {e}")