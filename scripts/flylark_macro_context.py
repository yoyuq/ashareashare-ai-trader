# -*- coding: utf-8 -*-
"""飞书宏观数据接入层 — 把平台内置宏观工具变成大师能消化的宏观上下文.

平台内置【经济金融数据】工具: 中美国债收益率 / PMI / 社会消费品零售 / 货币供应量 /
PPI / CPI / GDP / 社会融资规模增量。但工作流里没有任何节点消费它们 → 宏观派大师的
macro_context 一直是空的(LLM脑补)、裁判长无法判断"宏观剧变期"(决定派系权重)。

本节点把多个宏观工具的输出汇总成结构化 macro_context, 并给出:
  - 宏观评分 macro_score (宽松/中性/收紧) — 供大师/裁判长参考
  - 流动性/通胀/景气三个子维度
  - 是否"宏观剧变期"标记 macro_turmoil — 供裁判长: 剧变期宏观派×1.5

用法: 替换飞书一个代码节点, 放在宏观派大师之前:
  入参 arg1..arg8 = 各宏观工具输出(JSON或文本, 含指标名+最新值+环比/同比)
  出参 macro_context (文本, 喂给宏观派大师 + 裁判长)

平台工具输出格式以平台实际为准, 本节点容错解析: 能识别就识别, 识别不了则原样透传。
"""
import json
import re


# ===== 指标别名: 平台工具名 → 内部缩写 =====
ALIAS = {
    "cpi": "cpi", "居民消费价格": "cpi", "同比": "",
    "ppi": "ppi", "工业生产者出厂价格": "ppi",
    "pmi": "pmi", "制造业采购经理": "pmi", "采购经理指数": "pmi",
    "gdp": "gdp", "国内生产总值": "gdp",
    "m0": "m0", "m1": "m1", "m2": "m2", "货币供应": "m2",
    "社融": "srf", "社会融资": "srf", "社会融资规模": "srf",
    "社零": "retail", "消费品零售": "retail", "零售": "retail",
    "国债收益率": "yield", "国债": "yield", "收益率": "yield",
}


# 平台工具 JSON 里"指标名"可能用的字段名
_NAME_FIELDS = ("name", "indicator", "指标", "指标名", "item", "title", "名称")
# "数值"可能用的字段名
_VALUE_FIELDS = ("value", "val", "数值", "最新值", "现值", "current", "amount", "data")


def _num(v):
    """把数值字段转 float, 容忍 字符串数值 / (val,unit)。"""
    if isinstance(v, tuple):
        v = v[0]
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v.replace("%", "").replace(",", "").strip())
        except ValueError:
            return None
    return None


def _parse_block(arg):
    """尽量从平台工具输出中提取 {指标名: 数值}。"""
    arg = (arg or "").strip()
    if not arg:
        return None
    # 尝试 JSON
    try:
        obj = json.loads(arg)
    except Exception:
        obj = None
    if isinstance(obj, dict):
        # 优先找显式 "name/指标" + "value/数值" 字段对
        pair_out = {}
        name = None
        for k in obj:
            if any(f in k.lower() for f in _NAME_FIELDS) and isinstance(obj[k], str):
                name = obj[k]
                break
        if name is not None:
            for k in obj:
                if any(f in k.lower() for f in _VALUE_FIELDS):
                    n = _num(obj[k])
                    if n is not None:
                        pair_out[name.lower()] = n
            if pair_out:
                return pair_out
        # 否则递归收集所有数值(带路径), 用于兜底
        out = {}
        def _walk(v, prefix=""):
            if isinstance(v, dict):
                for k, val in v.items():
                    _walk(val, prefix + k.lower())
            elif isinstance(v, (int, float)):
                out[prefix.lower()] = float(v)
            elif isinstance(v, list):
                for i, x in enumerate(v[:3]):
                    _walk(x, prefix + f"[{i}]")
        _walk(obj)
        return out or None
    if isinstance(obj, list):
        # 列表: 每项可能是 {name,value}
        out = {}
        for item in obj[:10]:
            if isinstance(item, dict):
                sub = _parse_block(json.dumps(item, ensure_ascii=False))
                if sub:
                    out.update(sub)
        return out or None
    # 文本: 提取 "指标名: 数值" 或 "指标名 数值%"
    out = {}
    for m in re.finditer(r'([一-龥A-Za-z0-9_：: ]+?)[：:]\s*(-?\d+(?:\.\d+)?)\s*([%万亿元‰]?)', arg):
        name, val = m.group(1).strip(), float(m.group(2))
        out[name.lower()] = val
    return out or ({"raw": arg} if arg.strip() else None)


def main(arg1: str, arg2: str, arg3: str, arg4: str = "",
         arg5: str = "", arg6: str = "", arg7: str = "", arg8: str = "") -> dict:
    raw = {}
    for i, a in enumerate([arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8], 1):
        if a:
            raw[f"tool{i}"] = _parse_block(a)

    # ===== 从原始块里识别关键宏观指标 =====
    # 归一化: 把每个工具块里的数值都收集起来, 用于后续评分
    indicators = {}   # 缩写 -> 最新值(float, 原始)
    for blkname, blk in raw.items():
        if not isinstance(blk, dict):
            continue
        for k, v in blk.items():
            kl = k.lower()
            for alias, code in ALIAS.items():
                if alias and alias in kl and code:
                    # 取数值(可能是 (val,unit) 元组)
                    num = v[0] if isinstance(v, tuple) else v
                    if isinstance(num, (int, float)):
                        indicators.setdefault(code, float(num))
                    break

    # ===== 宏观评分 (可解释规则, 不完全依赖识别出的指标) =====
    # 用识别到的指标做方向判断; 缺失的维度如实标注"平台未提供"
    def _dir_of(code, lo, hi):
        """值高于 hi → 偏热/扩张; 低于 lo → 偏冷/收缩; 否则中性。返回 (方向, 值) 或 (None,None)"""
        v = indicators.get(code)
        if v is None:
            return None, None
        if v >= hi:
            return "扩张", v
        if v <= lo:
            return "收缩", v
        return "中性", v

    # 通胀: CPI YoY>3 过热, <0 通缩; PPI 同向前瞻
    cpi_dir, cpi_v = _dir_of("cpi", 0.0, 3.0)
    ppi_dir, ppi_v = _dir_of("ppi", -2.0, 2.0)
    # 景气: PMI>50 扩张, <50 收缩
    pmi_dir, pmi_v = _dir_of("pmi", 50.0, 50.0)
    if pmi_dir is None and pmi_v is not None:
        pmi_dir = "扩张" if pmi_v >= 50 else "收缩"
    # 流动性: M2 同比 高=宽松
    m2_dir, m2_v = _dir_of("m2", 6.0, 12.0)
    # 社融: 增量
    srf_dir, srf_v = _dir_of("srf", 0.0, 0.0)

    # 宽松计分: 通胀低(+宽松) 景气弱(+宽松空间) 流动性高(+宽松)
    loose_pts, tight_pts = 0, 0
    if cpi_dir == "收缩": loose_pts += 1
    elif cpi_dir == "扩张": tight_pts += 1
    if pmi_dir == "收缩": loose_pts += 1
    elif pmi_dir == "扩张": tight_pts += 1
    if m2_dir == "扩张": loose_pts += 1
    elif m2_dir == "收缩": tight_pts += 1

    if loose_pts > tight_pts + 1:
        macro_score = "宽松"        # 低通胀+弱景气+高流动性 → 政策有空间/正在宽松
    elif tight_pts > loose_pts + 1:
        macro_score = "收紧"        # 高通胀+强过景气+流动性收缩 → 政策收紧
    else:
        macro_score = "中性"

    # ===== 是否宏观剧变期 (供裁判长派系加权) =====
    # 剧变信号: 通胀过热(CPI>3) 或 通缩(CPI<0) 或 PPI深负(-3) 或 PMI<45 或 流动性骤变
    turmoil_signals = []
    if cpi_v is not None and (cpi_v > 3.0 or cpi_v < 0):
        turmoil_signals.append(f"CPI同比{cpi_v}%(通胀异常)")
    if ppi_v is not None and ppi_v < -3.0:
        turmoil_signals.append(f"PPI同比{ppi_v}%(深度通缩)")
    if pmi_v is not None and pmi_v < 45.0:
        turmoil_signals.append(f"PMI{pmi_v}(景气急转)")
    if srf_v is not None and srf_v < 0:
        turmoil_signals.append(f"社融增量{srf_v}(信用收缩)")
    macro_turmoil = len(turmoil_signals) >= 2   # 2个以上异常信号 → 视为宏观剧变期

    # ===== 组装 macro_context 文本 (喂给大师) =====
    lines = ["【宏观背景 · 平台经济金融数据】", ""]
    if indicators:
        for code, label in [("cpi", "CPI同比"), ("ppi", "PPI同比"), ("pmi", "PMI"),
                            ("m2", "M2同比"), ("srf", "社融增量"), ("gdp", "GDP同比"),
                            ("retail", "社零同比"), ("yield", "国债收益率")]:
            if code in indicators:
                lines.append(f"- {label}: {indicators[code]:.2f}")
    else:
        lines.append("- 本次未识别到具体的宏观指标数值(平台输出格式解析失败或未提供)")
    lines.append(f"- 宏观综合评分: {macro_score}")
    if macro_turmoil:
        lines.append(f"- ⚠️ 宏观剧变期信号: {'; '.join(turmoil_signals)}")
    lines.append("")
    lines.append("(数据来自平台内置经济金融数据工具, 仅供宏观研判参考, 非投资建议)")
    macro_context = "\n".join(lines)

    return {
        "macro_context": macro_context,
        "macro_score": macro_score,
        "macro_turmoil": macro_turmoil,
        "turmoil_signals": turmoil_signals,
        "indicators_recognized": {k: v for k, v in indicators.items()},
    }


if __name__ == "__main__":
    # 本地冒烟: 模拟平台 CPI/PPI/PMI/M2/社融 工具输出
    def tool(json_str):
        return json_str

    cpi = tool('{"name":"CPI同比","value":2.1,"yoy":2.1,"date":"2026-07"}')
    ppi = tool('{"name":"PPI同比","value":-1.8,"yoy":-1.8,"date":"2026-07"}')
    pmi = tool('{"name":"PMI","value":49.2,"date":"2026-07"}')
    m2 = tool('{"name":"M2同比","value":8.5,"yoy":8.5,"date":"2026-07"}')
    srf = tool('{"name":"社会融资规模增量","value":1.2,"unit":"万亿","date":"2026-07"}')

    r = main(cpi, ppi, pmi, m2, srf)
    print("宏观评分:", r["macro_score"], "| 剧变期:", r["macro_turmoil"], r["turmoil_signals"])
    print("识别指标:", r["indicators_recognized"])
    print("--- macro_context ---")
    print(r["macro_context"])

    print("\n===== 剧变场景: 高通胀+PMI急跌 =====")
    cpi2 = tool('{"name":"CPI同比","value":4.5,"yoy":4.5,"date":"2026-07"}')
    pmi2 = tool('{"name":"PMI","value":43.0,"date":"2026-07"}')
    r2 = main(cpi2, tool('{"name":"PPI","value":-4.2}'), pmi2, '')
    print("宏观评分:", r2["macro_score"], "| 剧变期:", r2["macro_turmoil"], r2["turmoil_signals"])