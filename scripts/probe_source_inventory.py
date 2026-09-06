"""Probe public A-share data capabilities with real requests and reproducible evidence.

Run with the project's Python interpreter; no credentials or project configuration
are loaded. Each source runs in an isolated, bounded process. Failed HTTP probes
retry once with all proxy environment variables removed, like attention_collector.
The Markdown links the complete JSON evidence for auditing individual attempts.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ("600519", "600036", "601318", "600030", "600276", "000001", "000858", "000333", "300750", "688981")
PROXY_KEYS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy", "NO_PROXY", "no_proxy")
DC_URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"
PIT_FIELDS = ("NOTICE_DATE", "HOLD_NOTICE_DATE", "ACTUAL_PUBLISH_DATE", "FIRST_APPOINT_DATE", "UPDATE_DATE", "EITIME", "pubDate", "发布时间", "公告日期")


@dataclass(frozen=True)
class Source:
    key: str
    category: str
    interface: str
    content: str
    frequency: str
    kind: str
    date_field: str = ""
    aliases: tuple[str, ...] = ()
    caveat: str = ""
    scope: str = "stock"
    report: str = ""


def dc(key, category, report, content, frequency, field, aliases=(), caveat="", scope="stock"):
    return Source(key, category, report, content, frequency, "dc", field, aliases, caveat, scope, report)


SOURCES = (
    Source("daily_em", "行情", "ak.stock_zh_a_hist", "东财不复权 OHLCV/成交额/换手", "日", "daily_em", "日期", caveat="从 1990-01-01 请求；不复权避免混入今天的复权因子；价格仍可能校正"),
    Source("daily_tx", "行情", "ak.stock_zh_a_hist_tx", "腾讯不复权 OHLCV", "日", "daily_tx", "date", caveat="从 1990-01-01 请求；AKShare 按年多次请求，耗时较长"),
    Source("minute_em", "行情", "ak.stock_zh_a_hist_min_em", "东财 5 分钟 K 线", "5 分钟", "minute_em", "时间", caveat="请求从 1990 年开始；只能量到服务器保留窗口，不能视为上市以来分钟历史"),
    Source("quote_tx", "行情", "qt.gtimg.cn/q=…", "腾讯实时价/昨收/成交量/PE/PB", "实时快照", "quote_tx", "timestamp", caveat="仅当前快照；周末可能返回上个交易日，时间戳不是采集时刻"),
    Source("daily_bs", "行情", "bs.query_history_k_data_plus", "Baostock 不复权日线/PE/PB/停牌/ST", "日", "daily_bs", "date", caveat="从 1990-01-01 请求；TCP 不走 HTTP 代理；用 next/get_row_data，避免旧 get_data 的 DataFrame.append 不兼容；空串不是零"),
    Source("profile_em", "基本面", "ak.stock_individual_info_em", "总股本/流通股/市值/行业/上市日期", "当前快照", "profile_em", caveat="上市日期是属性，不是可回溯历史起点；覆盖成功不代表所有属性完整"),
    dc("holder", "基本面", "RPT_HOLDERNUM_DET", "股东户数及变动", "不定期/报告期", "END_DATE", ("ak.stock_zh_a_gdhs_detail_em",), "HOLD_NOTICE_DATE 可对齐公告；历史记录可能被修订"),
    dc("financial", "基本面", "RPT_LICO_FN_CPD", "业绩报表/营收/利润/EPS", "季/年", "REPORTDATE", ("ak.stock_yjbb_em",), "直连保留 NOTICE_DATE/REPORTDATE；公告日不等于所有数值的首发版本"),
    dc("balance", "基本面", "RPT_DMSK_FN_BALANCE", "资产负债表主要项目", "季/年", "REPORT_DATE", ("ak.stock_zcfz_em",), "直连不套行业过滤；银行与非金融企业字段适用性不同"),
    dc("income", "基本面", "RPT_DMSK_FN_INCOME", "利润表主要项目", "季/年", "REPORT_DATE", ("ak.stock_lrb_em",), "累计口径不是单季口径；历史值可能追溯调整"),
    dc("cashflow", "基本面", "RPT_DMSK_FN_CASHFLOW", "现金流量表主要项目", "季/年", "REPORT_DATE", ("ak.stock_xjll_em",), "累计口径；当前可见版本不构成完整 PIT"),
    Source("profit_bs", "基本面", "bs.query_profit_data", "Baostock 盈利能力指标", "季", "profit_bs", "statDate", caveat="从 2007Q1 逐季找首个非空季；只探最早非空季，不能把查询下界当服务起点"),
    Source("flow_em", "资金流", "ak.stock_individual_fund_flow", "主力/超大单/大中小单净流入", "日", "flow_em", "日期", caveat="返回滚动保留窗口；资金分类为供应商算法口径，无公告日"),
    dc("north", "资金流", "RPT_MUTUAL_HOLDSTOCKNDATE_STA", "北向个股持股/增减/持股市值", "交易日", "TRADE_DATE", ("ak.stock_hsgt_individual_em",), "披露政策变化后序列可能停止更新；只记录实际返回的保留历史"),
    dc("appointment", "事件", "RPT_PUBLIC_BS_APPOIN", "定期报告预约/变更/实际披露日", "事件", "REPORT_DATE", ("ak.stock_yysj_em",), "预约修改字段不含每次修改获知时间，不能重建每个历史时点的预约快照"),
    dc("forecast", "事件", "RPT_PUBLIC_OP_NEWPREDICT", "业绩预告区间/类型/原因", "事件", "REPORT_DATE", ("ak.stock_yjyg_em",), "同一报告期多指标/多版本；必须以公告日对齐且检查修订"),
    dc("lhb", "事件", "RPT_DAILYBILLBOARD_DETAILSNEW", "龙虎榜公开交易明细", "交易日事件", "TRADE_DATE", ("ak.stock_lhb_detail_em",), "非每只股票每天都有事件；D1/D5 等事后涨跌幅不可当时点特征"),
    dc("blocktrade", "事件", "RPT_BLOCKTRADE_STA", "大宗交易每日统计", "交易日事件", "TRADE_DATE", ("ak.stock_dzjy_mrtj",), "事件覆盖率不是全市场覆盖率；含事后收益列，使用时需排除"),
    Source("hotrank", "舆情", "ak.stock_hot_rank_detail_em", "东方财富历史人气/粉丝比例", "日", "hotrank", "时间", caveat="排名与粉丝为两次 HTTP 请求；按位置拼接，需注意日期对齐；无首发版本时间"),
    dc("comment", "舆情", "RPT_DMSK_TS_STOCKNEW", "千股千评/关注/排名/综合得分", "日快照", "TRADE_DATE", ("ak.stock_comment_em",), "只探到当前快照；不能从日期列推断存在日度历史"),
    Source("news", "舆情", "ak.stock_news_em", "按股票代码搜索财经新闻", "新闻事件", "news", "发布时间", caveat="当前封装 pageSize=10，相关性排序；实测最早只是返回页最早，非完整历史；关键词命中不等于实体准确关联"),
    dc("gdp", "宏观", "RPT_ECONOMY_GDP", "中国 GDP/产业增加值", "季", "REPORT_DATE", ("ak.macro_china_gdp",), "宏观修订序列，不等于历史首发版本；非个股数据", "macro"),
    dc("cpi", "宏观", "RPT_ECONOMY_CPI", "全国/城乡 CPI", "月", "REPORT_DATE", ("ak.macro_china_cpi",), "统计期不等于发布日期；非个股数据", "macro"),
    dc("pmi", "宏观", "RPT_ECONOMY_PMI", "制造业/非制造业 PMI", "月", "REPORT_DATE", ("ak.macro_china_pmi",), "统计期不等于发布日期；非个股数据", "macro"),
    dc("money", "宏观", "RPT_ECONOMY_CURRENCY_SUPPLY", "M0/M1/M2 货币供应量", "月", "REPORT_DATE", ("ak.macro_china_money_supply",), "定义变更/修订不可由当前序列还原；非个股数据", "macro"),
)
COMPARISONS = (
    Source("alias_holder", "基本面", "ak.stock_zh_a_gdhs_detail_em", "同底层封装比较", "不定期", "alias_holder"),
    Source("alias_financial", "基本面", "ak.stock_yjbb_em", "同底层封装比较", "季/年", "alias_financial", scope="batch"),
    Source("alias_gdp", "宏观", "ak.macro_china_gdp", "同底层封装比较", "季", "alias_gdp", scope="macro"),
)
SOURCE_MAP = {source.key: source for source in (*SOURCES, *COMPARISONS)}


def safe_error(error: Exception) -> str:
    """Retain failure semantics without propagating proxy credentials or queries."""
    message = re.sub(r"https?://[^\s'\"<>]+", "[URL]", str(error))
    message = re.sub(r"(with url: )[^\s]+", r"\1[request path]", message)
    return f"{type(error).__name__}: {message}".replace("\n", " ")[:500]


@contextmanager
def transport(mode: str, proxy: str, timeout: float):
    """Scope env and requests monkeypatches to a single isolated worker attempt."""
    import requests

    old_env = {key: os.environ.get(key) for key in PROXY_KEYS}
    original = requests.sessions.Session.request
    old_timeout = socket.getdefaulttimeout()
    for key in PROXY_KEYS:
        os.environ.pop(key, None)
    if mode == "proxy":
        os.environ.update(HTTP_PROXY=proxy, HTTPS_PROXY=proxy)

    def bounded_request(session, method, url, **kwargs):
        kwargs["timeout"] = timeout
        kwargs["proxies"] = {"http": proxy, "https": proxy} if mode == "proxy" else {"http": None, "https": None}
        session.trust_env = False
        return original(session, method, url, **kwargs)

    requests.sessions.Session.request = bounded_request
    socket.setdefaulttimeout(timeout)
    try:
        yield
    finally:
        requests.sessions.Session.request = original
        socket.setdefaulttimeout(old_timeout)
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def fetch_dc(source: Source, code: str, page_limit: int):
    """Fetch oldest pages, validating schema, sort order and stock identity."""
    import pandas as pd
    import requests

    params = {"reportName": source.report, "columns": "ALL", "sortColumns": source.date_field,
              "sortTypes": "1", "pageSize": 500, "pageNumber": 1, "source": "WEB", "client": "WEB"}
    if code != "macro":
        params["filter"] = f'(SECURITY_CODE="{code}")'
    if source.key == "north":
        params["filter"] += '(INTERVAL_TYPE="1")'
    rows, pages, total = [], 1, None
    for page in range(1, page_limit + 1):
        params["pageNumber"] = page
        response = requests.get(DC_URL, params=params)
        response.raise_for_status()
        payload = response.json()
        result = payload.get("result")
        if not payload.get("success") or not isinstance(result, dict):
            raise RuntimeError(f"datacenter success={payload.get('success')}: {payload.get('message')}")
        data = result.get("data") or []
        if any(source.date_field not in row for row in data):
            raise ValueError(f"missing sort/date field {source.date_field}")
        if code != "macro" and any(str(row.get("SECURITY_CODE")) != code for row in data):
            raise ValueError("server ignored SECURITY_CODE filter")
        rows.extend(data)
        total, pages = result.get("count"), int(result.get("pages") or 0)
        if page >= pages:
            break
    frame = pd.DataFrame(rows)
    if not frame.empty:
        dates = pd.to_datetime(frame[source.date_field], errors="coerce")
        if dates.isna().any() or not dates.is_monotonic_increasing:
            raise ValueError("ascending historical boundary could not be validated")
    return frame, {"url": DC_URL, "params": params, "server_count": total, "server_pages": pages,
                   "pages_fetched": min(pages, page_limit), "truncated": pages > page_limit,
                   "boundary_method": "server ascending date sort; oldest page(s)"}


def fetch(source: Source, code: str, page_limit: int):
    import pandas as pd
    import requests

    market = "sh" if code.startswith("6") else "sz"
    today = date.today().strftime("%Y%m%d")
    if source.kind == "dc":
        return fetch_dc(source, code, page_limit)
    if source.kind == "quote_tx":
        response = requests.get(f"https://qt.gtimg.cn/q={market}{code}")
        response.raise_for_status()
        response.encoding = "gbk"
        values = response.text.split('"')[1].split("~")
        if len(values) < 47 or values[2] != code or float(values[3]) <= 0:
            raise ValueError("invalid Tencent quote payload")
        frame = pd.DataFrame([{"code": values[2], "price": values[3], "previous_close": values[4],
                               "volume": values[6], "timestamp": datetime.strptime(values[30], "%Y%m%d%H%M%S").isoformat(),
                               "pe": values[39], "pb": values[46]}])
        return frame, {"boundary_method": "current quote timestamp; no history request"}
    if source.kind in ("daily_bs", "profit_bs"):
        import baostock as bs

        login = bs.login()
        if login.error_code != "0":
            raise RuntimeError(f"Baostock login: {login.error_code} {login.error_msg}")
        try:
            queries = []
            if source.kind == "daily_bs":
                result = bs.query_history_k_data_plus(f"{market}.{code}", "date,code,open,high,low,close,volume,amount,tradestatus,peTTM,pbMRQ,isST", start_date="1990-01-01", end_date=date.today().isoformat(), frequency="d", adjustflag="3")
                queries.append("1990-01-01.." + date.today().isoformat())
            else:
                result = None
                for year in range(2007, date.today().year + 1):
                    for quarter in range(1, 5):
                        result = bs.query_profit_data(f"{market}.{code}", year=year, quarter=quarter)
                        queries.append(f"{year}Q{quarter}")
                        if result.error_code != "0":
                            raise RuntimeError(f"Baostock {result.error_code}: {result.error_msg}")
                        if result.data:
                            break
                    if result.data:
                        break
            if result.error_code != "0":
                raise RuntimeError(f"Baostock {result.error_code}: {result.error_msg}")
            rows = []
            while result.next():
                rows.append(result.get_row_data())
            if result.error_code != "0":
                raise RuntimeError(f"Baostock row fetch {result.error_code}: {result.error_msg}")
            frame = pd.DataFrame(rows, columns=result.fields)
            return frame, {"queries": queries, "boundary_method": "earliest returned from explicit query floor"}
        finally:
            bs.logout()
    import akshare as ak

    if source.kind == "alias_holder":
        params = {"symbol": code}
        frame = ak.stock_zh_a_gdhs_detail_em(**params)
    elif source.kind == "alias_financial":
        params = {"date": f"{date.today().year - 1}1231"}
        frame = ak.stock_yjbb_em(**params)
    elif source.kind == "alias_gdp":
        params = {}
        frame = ak.macro_china_gdp()
    elif source.kind == "daily_em":
        params = {"symbol": code, "start_date": "19900101", "end_date": today, "adjust": ""}
        frame = ak.stock_zh_a_hist(**params)
    elif source.kind == "daily_tx":
        params = {"symbol": market + code, "start_date": "19900101", "end_date": today, "adjust": ""}
        frame = ak.stock_zh_a_hist_tx(**params)
    elif source.kind == "minute_em":
        params = {"symbol": code, "start_date": "1990-01-01 00:00:00", "end_date": date.today().isoformat() + " 23:59:59", "period": "5", "adjust": ""}
        frame = ak.stock_zh_a_hist_min_em(**params)
    elif source.kind == "profile_em":
        params = {"symbol": code}
        frame = ak.stock_individual_info_em(**params)
    elif source.kind == "flow_em":
        params = {"stock": code, "market": market}
        frame = ak.stock_individual_fund_flow(**params)
    elif source.kind == "hotrank":
        params = {"symbol": market.upper() + code}
        frame = ak.stock_hot_rank_detail_em(**params)
    elif source.kind == "news":
        params = {"symbol": code}
        frame = ak.stock_news_em(**params)
    else:
        raise ValueError(f"unknown source kind: {source.kind}")
    return frame, {"params": params, "boundary_method": "minimum in actual returned rows; query window/retention limits apply"}


def summarize_frame(frame, source: Source) -> dict:
    import pandas as pd

    if frame is None or frame.empty:
        raise ValueError("empty response; no fabricated replacement")
    columns = [str(column) for column in frame.columns]
    result = {"rows": len(frame), "columns": columns, "first": None, "last": None,
              "pit_fields": {field: int(frame[field].notna().sum()) for field in PIT_FIELDS if field in frame}}
    if source.date_field:
        if source.date_field not in frame:
            raise ValueError(f"missing date field {source.date_field}; columns={columns}")
        dates = pd.to_datetime(frame[source.date_field], errors="coerce")
        if dates.isna().all():
            raise ValueError(f"date field {source.date_field} contains no valid timestamp")
        result.update(first=dates.min().isoformat(), last=dates.max().isoformat(), invalid_dates=int(dates.isna().sum()))
    result["payload_sha256"] = hashlib.sha256(frame.to_json(orient="split", date_format="iso", force_ascii=False).encode("utf-8")).hexdigest()
    if source.scope == "batch":
        code_field = next((field for field in ("股票代码", "代码", "SECURITY_CODE", "证券代码") if field in frame), None)
        if code_field is None:
            raise ValueError("batch wrapper has no security code column")
        actual = set(frame[code_field].astype(str).str.zfill(6))
        result["returned_codes"] = sorted(actual)
    return result


def probe_sample(source, code, proxy, timeout, page_limit):
    attempts = []
    observed_at = datetime.now(timezone.utc).isoformat()
    modes = ("tcp_direct",) if source.kind.endswith("_bs") else ("proxy", "direct")
    for mode in modes:
        start = time.monotonic()
        try:
            with transport(mode, proxy, timeout):
                frame, metadata = fetch(source, code, page_limit)
                summary = summarize_frame(frame, source)
            attempts.append({"mode": mode, "ok": True, "seconds": round(time.monotonic() - start, 3)})
            return {"code": code, "status": "ok", "observed_at": observed_at, "attempts": attempts, **summary, **metadata}
        except Exception as error:
            attempts.append({"mode": mode, "ok": False, "seconds": round(time.monotonic() - start, 3), "error": safe_error(error)})
    return {"code": code, "status": "unavailable", "observed_at": observed_at, "attempts": attempts, "rows": 0, "first": None, "last": None}


def worker(args):
    source = SOURCE_MAP[args.worker]
    codes = [source.scope] if source.scope in ("macro", "batch") else args.codes
    for code in codes:
        # Third-party progress/login output must not corrupt JSON Lines stdout.
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            evidence = probe_sample(source, code, args.proxy, args.timeout, args.pages)
        if "returned_codes" in evidence:
            returned = set(evidence.pop("returned_codes"))
            evidence["coverage_codes"] = {sample: sample in returned for sample in args.codes}
        print(json.dumps(evidence, ensure_ascii=False), flush=True)
        time.sleep(args.delay)
    return 0


def run_source(source, args):
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", source.key, "--codes", *args.codes,
               "--timeout", str(args.timeout), "--pages", str(args.pages), "--delay", str(args.delay), "--proxy", args.proxy]
    env = dict(os.environ, PYTHON_DOTENV_DISABLED="1", PYTHONIOENCODING="utf-8")
    try:
        process = subprocess.run(command, capture_output=True, encoding="utf-8", env=env, timeout=args.source_timeout, check=False)
        output, process_error = process.stdout, None if process.returncode == 0 else f"worker exit {process.returncode}"
    except subprocess.TimeoutExpired as error:
        output = error.stdout or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        process_error = f"worker deadline exceeded ({args.source_timeout}s); unfinished samples not claimed tested"
    records = []
    for line in output.splitlines():
        if not line.startswith("{"):
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            process_error = process_error or "worker produced a partial JSON record"
    observed = {record["code"] for record in records}
    for code in ([source.scope] if source.scope in ("macro", "batch") else args.codes):
        if code not in observed:
            records.append({"code": code, "status": "incomplete", "attempts": [], "rows": 0, "first": None, "last": None,
                            "error": process_error or "worker produced no record"})
    return {"source": asdict(source), "samples": records, "process_error": process_error}


def exported_catalog():
    """Enumerate names without executing any data retrieval or reading secrets."""
    import akshare as ak

    catalog = {category: [] for category in ("行情", "基本面", "资金流", "事件", "舆情", "宏观")}
    excluded = []
    for name in sorted(dir(ak)):
        if not callable(getattr(ak, name)) or not name.startswith(("stock_", "macro_china_")):
            continue
        if any(token in name for token in ("_hk", "_us", "_b_", "_us_", "_hk_", "_b_spot")):
            excluded.append(name)
            continue
        if name.startswith("macro_china_"):
            category = "宏观"
        elif any(token in name for token in ("fund_flow", "hsgt", "margin", "ggt", "market_fund")):
            category = "资金流"
        elif any(token in name for token in ("hot_", "comment", "news", "weibo", "sns", "research")):
            category = "舆情"
        elif any(token in name for token in ("lhb", "dzjy", "disclosure", "notice", "report_disclosure", "yjyg", "yjkb", "yysj", "dividend", "bonus", "restricted", "ipo", "new_", "ggcg", "gpzy", "repurchase", "register", "add_stock", "stop", "staq", "delist", "zdht", "gsrl")):
            category = "事件"
        elif any(token in name for token in ("financial", "finance", "fundamental", "zcfz", "lrb", "xjll", "yjbb", "gdhs", "gdfx", "holder", "hold", "profile", "individual_info", "industry", "ipo_info", "zygc", "js_we", "valuation", "pe_", "pb_", "a_lg")):
            category = "基本面"
        else:
            category = "行情"
        catalog[category].append(name)
    return catalog, excluded


def markdown(evidence):
    settings = evidence["settings"]
    lines = ["# 数据源能力盘点（真实网络探测）", "", f"探测时间：{evidence['observed_at']} 至 {evidence['completed_at']}；Python {evidence['python']}；AKShare {evidence['akshare']}；Baostock {evidence['baostock']}。", "",
             "本表只记录数据能力，不作策略判断。历史起点全部来自本次返回数据，不引用文档声称的起始日。起点是固定样本中的最早值，不等于全市场最早；覆盖率只代表这 10 只跨沪深主板、创业板、科创板样本，不外推至北交所、退市股或全市场。", "",
             "固定样本：" + "、".join(evidence["codes"]) + "。宏观没有股票维度，覆盖记为不适用，不把重复请求冒充 10 只覆盖。", "",
             f"每个 HTTP 样本先用指定本地代理，失败后清除全部代理设置直连一次；Baostock 是 TCP 直连。调用失败/空数据标不可用；进程超时且未留下证据的样本标未完成。稳定性仅为本次调用的成功率/耗时，未压测频率上限；本次源内间隔 {settings['delay']} 秒、{settings['workers']} 个独立进程，单 HTTP 请求 {settings['timeout']} 秒。", "",
             "东财端点直连指绕过 AKShare 封装直接请求 API；网络仍可能经代理，以 attempts.mode 为准，不能据此宣称无需代理。按原始日期字段升序、pageSize=500 分页；本次最多取 " + str(settings["pages"]) + " 页以验证历史起点和股票过滤，server_count/server_pages/truncated 留在证据中。截断数据的 last 仅为返回页末日，不代表数据库最新日。财报最早报告期可能早于上市，报告期不等于当时已公开。带公告/更新时间字段只能做日期对齐，未证明版本历史完整，均不标完整 PIT。", "",
             "重跑（不加载项目配置或密钥）：", "", "```powershell", "$env:PYTHON_DOTENV_DISABLED='1'", ".venv\\Scripts\\python.exe scripts/probe_source_inventory.py", "# 单独复测不会覆盖正式全量矩阵：", ".venv\\Scripts\\python.exe scripts/probe_source_inventory.py --only holder balance news --output docs/data_source_inventory_partial.md", "```", "",
             f"完整机器证据：[JSON]({evidence['evidence_file']})。--codes 必须提供至少 10 个不同六位 A 股代码；--pages 控制东财历史页数，--source-timeout 控制单源截止时间。脚本拒绝写入 reports/ 和密钥文件。", "",
             "## 实测矩阵", "", "| 类别 / 接口或端点 | 数据内容 | 历史起点（实测） | 频率粒度 | PIT 质量（实测字段） | 覆盖度（抽样实测） | 调用稳定性 | 已知坑 |", "|---|---|---|---|---|---|---|---|"]
    for item in evidence["results"]:
        source, samples = item["source"], item["samples"]
        good = [sample for sample in samples if sample["status"] == "ok"]
        dates = [sample["first"] for sample in good if sample.get("first")]
        snapshot = source["key"] in ("quote_tx", "profile_em", "comment")
        earliest = ("仅快照；返回日 " if snapshot else "样本最早 ") + min(dates)[:10] if dates else ("仅当前属性，无历史" if good and snapshot else "不可用，无法实测")
        if source["key"] == "news" and dates:
            earliest = "返回搜索页最早 " + min(dates)[:10]
        pit = sorted({field for sample in good for field, count in sample.get("pit_fields", {}).items() if count})
        pit_text = ("有 " + "/".join(pit) + "；版本 PIT 未证实") if pit else ("未返回公告/发布时间字段" if good else "不可用，未验证")
        incomplete = sum(sample["status"] == "incomplete" for sample in samples)
        coverage = (f"宏观序列 {len(good)}/1，{sum(sample['rows'] for sample in good)} 期；个股 N/A" if source["scope"] == "macro" else f"{len(good)}/{len(samples)} 返回非空") + (f"；{incomplete} 未完成" if incomplete else "")
        attempts = [attempt for sample in samples for attempt in sample["attempts"]]
        successful = [attempt for attempt in attempts if attempt["ok"]]
        modes = "/".join(sorted({attempt["mode"] for attempt in successful})) or "均失败"
        slowest = max((attempt["seconds"] for attempt in attempts), default=0)
        failures = [attempt["error"] for attempt in attempts if not attempt["ok"]]
        stability = f"{modes}；尝试成功 {len(successful)}/{len(attempts)}；最大 {slowest:.2f}s"
        if failures:
            stability += "；" + failures[-1][:95]
        cells = [f"{source['category']} / `{source['interface']}` ([证据](#evidence-{source['key']}))", source["content"], earliest, source["frequency"], pit_text, coverage, stability, source["caveat"]]
        lines.append("| " + " | ".join(str(cell).replace("|", "\\|").replace("\n", " ") for cell in cells) + " |")
    lines.extend(["", "## 同底层重复与保留依据", "", "下列 AKShare 封装与对应直连使用相同 reportName，矩阵只保留直连一行。实测比较代表性的逐股股东户数、全市场业绩报表、宏观 GDP 封装（结果如下）；其余同族按原始字段保留和受控分页选择直连，未声称逐个证明所有环境中最稳。腾讯、东财、Baostock 日线来自不同服务；巨潮 stock_report_disclosure 也不是东财 stock_yysj_em 的同底层别名。", "",
                  "对照两侧返回范围不同（封装业绩报表取去年年报全市场、直连取每只样本历史），下列开销为本次累计耗时，不是严格同负载性能测试。", "",
                  "| 封装代表 | 封装本次结果 | 端点本次结果 | 直连保留依据 |", "|---|---|---|---|"])
    for comparison in evidence.get("comparisons", []):
        good = [sample for sample in comparison["samples"] if sample["status"] == "ok"]
        coverage = next((sample.get("coverage_codes") for sample in good if "coverage_codes" in sample), None)
        description = f"调用返回 {len(good)}/{len(comparison['samples'])}"
        if coverage:
            description += f"；固定股票覆盖 {sum(coverage.values())}/{len(coverage)}"
        attempts = [attempt for sample in comparison["samples"] for attempt in sample["attempts"]]
        description += f"；累计 {sum(attempt['seconds'] for attempt in attempts):.2f}s"
        direct = next(item for item in evidence["results"] if item["source"]["key"] == comparison["source"]["key"].removeprefix("alias_"))
        direct_attempts = [attempt for sample in direct["samples"] for attempt in sample["attempts"]]
        direct_good = sum(sample["status"] == "ok" for sample in direct["samples"])
        wrapper_good = sum(coverage.values()) if coverage else len(good)
        relation = "本次覆盖相同" if direct_good == wrapper_good else ("本次端点覆盖更高" if direct_good > wrapper_good else "本次封装覆盖更高，需复核端点代表行")
        direct_result = f"非空 {direct_good}/{len(direct['samples'])}；累计 {sum(attempt['seconds'] for attempt in direct_attempts):.2f}s"
        lines.append(f"| `{comparison['source']['interface']}` | {description} | {direct_result} | {relation}；端点保留原始日期/公告列、排序及分页证据；未声称稳定性显著更优 |")
    lines.append("")
    for item in evidence["results"]:
        source = item["source"]
        if source["aliases"]:
            lines.append(f"- `{source['interface']}` ← " + ", ".join(f"`{alias}`" for alias in source["aliases"]))
    lines.extend(["", "新增直连验证范围是除已知股东户数和预约披露外的矩阵 reportName；是否实际可用见各行，失败端点不计已验证可用。", "", "## AKShare 公开接口枚举（符号级索引）", "", "这是已安装版本导出名的可重跑枚举，按功能关键字分组，混合市场接口可能包含非 A 股。排除明确 HK/US/B 股命名。它用于界定后续扩展范围：只有实测矩阵及同底层映射具有本次能力证据，其余全部**未测**，不宣称可用或具有历史覆盖。", ""])
    for category, names in evidence["catalog"].items():
        lines.extend([f"### {category}（{len(names)} 个导出符号）", "", ", ".join(f"`{name}`" for name in names), ""])
    lines.extend(["## 证据摘要", "", "status=ok 表示有实际非空响应，字段非空数见 pit_fields；status=unavailable 见 attempts.error。payload_sha256 为本次规范化响应校验和，只保留工程证据，不收录新闻正文或财务原文。", ""])
    for item in evidence["results"]:
        source = item["source"]
        lines.extend([f"<a id=\"evidence-{source['key']}\"></a>", f"### {source['interface']}", "", "| 样本 | 状态 | 返回行数 | 返回最早 / 最晚 | 传输尝试 |", "|---|---|---|---|---|"])
        for sample in item["samples"]:
            attempts = "; ".join(f"{attempt['mode']}={'ok' if attempt['ok'] else 'FAIL'} {attempt['seconds']}s" for attempt in sample["attempts"])
            lines.append(f"| {sample['code']} | {sample['status']} | {sample['rows']} | {sample.get('first') or '—'} / {sample.get('last') or '—'} | {attempts or sample.get('error', '未执行')} |")
        lines.append("")
    lines.extend(["接口语义参考（历史与覆盖数字均来自上面的实测）：[AKShare 股票文档](https://akshare.akfamily.xyz/data/stock/stock.html)、[AKShare 宏观文档](https://akshare.akfamily.xyz/data/macro/macro.html)、[Baostock 官方文档](http://www.baostock.com/baostock/index.php/Python_API文档)。同底层映射依据安装包对应函数源码中的 reportName。", ""])
    return "\n".join(lines)


def safe_output(path: Path) -> Path:
    original = path.absolute()
    resolved = path.resolve()
    for candidate in (original, resolved):
        if any(part.lower() == "reports" for part in candidate.parts) or candidate.name.lower() in (".env", ".mcp.json"):
            raise ValueError("output must not access reports/ or secret configuration files")
    return resolved


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "data_source_inventory.md")
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--only", nargs="+", choices=sorted(source.key for source in SOURCES))
    parser.add_argument("--codes", nargs="+", default=list(SAMPLES))
    parser.add_argument("--proxy", default="http://127.0.0.1:7897")
    parser.add_argument("--timeout", type=float, default=6)
    parser.add_argument("--source-timeout", type=float, default=240)
    parser.add_argument("--delay", type=float, default=0.15)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--pages", type=int, default=2)
    parser.add_argument("--worker", choices=sorted(SOURCE_MAP), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if len(set(args.codes)) < 10 or len(set(args.codes)) != len(args.codes) or any(not re.fullmatch(r"(?:60|68|00|30)\d{4}", code) for code in args.codes):
        parser.error("--codes requires at least 10 distinct six-digit Shanghai/Shenzhen A-share codes")
    if min(args.timeout, args.source_timeout, args.workers, args.pages) <= 0 or args.delay < 0:
        parser.error("timeouts, workers and pages must be positive; delay must be nonnegative")
    try:
        args.output = safe_output(args.output)
        args.json_output = safe_output(args.json_output or args.output.with_name(args.output.stem + "_evidence.json"))
        if args.output == args.json_output:
            raise ValueError("Markdown and JSON outputs must have different paths")
    except ValueError as error:
        parser.error(str(error))
    if args.only and args.output == (ROOT / "docs" / "data_source_inventory.md").resolve() and not args.worker:
        parser.error("--only requires a different --output so partial probes cannot overwrite the full inventory")
    return args


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    os.environ["PYTHON_DOTENV_DISABLED"] = "1"
    args = parse_args(argv)
    if args.worker:
        return worker(args)
    observed_at = datetime.now(timezone.utc).isoformat()
    selected = [source for source in SOURCES if not args.only or source.key in args.only]
    comparisons = [source for source in COMPARISONS if source.key.removeprefix("alias_") in {item.key for item in selected}]
    results = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_source, source, args): source for source in (*selected, *comparisons)}
        for future in as_completed(futures):
            source = futures[future]
            result = future.result()
            results[source.key] = result
            good = sum(sample["status"] == "ok" for sample in result["samples"])
            print(f"{source.key}: {good}/{len(result['samples'])} nonempty", flush=True)
    catalog, excluded = exported_catalog()
    evidence = {"schema_version": 1, "observed_at": observed_at, "completed_at": datetime.now(timezone.utc).isoformat(),
                "python": sys.version.split()[0], "akshare": importlib.metadata.version("akshare"),
                "baostock": importlib.metadata.version("baostock"), "codes": args.codes,
                "evidence_file": os.path.relpath(args.json_output, args.output.parent).replace("\\", "/"),
                "settings": {"timeout": args.timeout, "source_timeout": args.source_timeout, "workers": args.workers, "delay": args.delay, "pages": args.pages, "proxy": "specified proxy; credentials never recorded"},
                "results": [results[source.key] for source in selected], "comparisons": [results[source.key] for source in comparisons], "catalog": catalog, "excluded_names": excluded}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown(evidence), encoding="utf-8")
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")
    return 2 if any(sample["status"] == "incomplete" for result in evidence["results"] for sample in result["samples"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
