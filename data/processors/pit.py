"""
PIT (Point-in-Time) 数据处理器

消除前视偏差(Look-ahead Bias)和幸存者偏差(Survivorship Bias):

前视偏差:
  例: 2024-01-15使用2023Q4财报数据→错误!
  2023Q4财报要到2024年4月底才披露,1月根本看不到。
  PIT修正: 2024-01-15只能看到2023Q3数据。

幸存者偏差:
  当前(2026年)的沪深300成分股 ≠ 2016年的沪深300成分股。
  2016年回测时,绝不能用今天的成分股列表。

实现方式:
  - 财报PIT: 维护 (财报截止日, 实际披露日) 映射 → 回测时按当时可用的最新数据取
  - 成分股PIT: 维护每个指数各时间截面上的成分股快照
"""

from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger


class PITProcessor:
    """
    Point-in-Time数据处理器

    核心逻辑:
    1. 财报PIT: for a given query_date, find the latest report with report_date
       <= query_date AND actual_disclose_date <= query_date
    2. 成分股PIT: for a given query_date, use the index constituents as of that date
    """

    def __init__(self):
        # 财报披露延迟表: (report_period → actual_disclose_date)
        # A股财报披露规则:
        #   Q1(3/31): 最晚4/30披露
        #   Q2/半年报(6/30): 最晚8/31披露
        #   Q3(9/30): 最晚10/31披露
        #   年报(12/31): 最晚次年4/30披露
        self._disclosure_schedule: Optional[pd.DataFrame] = None
        # 指数成分股历史快照
        self._index_constituents: Dict[str, Dict[date, List[str]]] = {}

    # ═══════════════════════════════════════════════════════════════
    # 财报PIT
    # ═══════════════════════════════════════════════════════════════

    def get_latest_available_report(
        self,
        symbol: str,
        query_date: date,
        financials_df: pd.DataFrame,
    ) -> Optional[pd.Series]:
        """
        获取query_date时点下可用的最新财报

        逻辑:
        1. 找到所有 report_date <= query_date 的报告期
        2. 对于每个报告期, actual_disclose_date <= query_date 才可用
        3. 返回可用中 report_date 最大的那条

        Args:
            symbol: 股票代码
            query_date: 查询时点(即"今天")
            financials_df: 该股票的所有财报数据, 需含 report_date 和 disclose_date 列

        Returns:
            query_date时点下可用的最新一行财报数据, 或 None
        """
        if financials_df.empty:
            return None

        df = financials_df.copy()

        # 确保有披露日期列
        if "disclose_date" not in df.columns:
            # 如果数据本身没有披露日期, 用保守估计:
            # 最晚截止日后推作为实际可见日期
            df["disclose_date"] = df["report_date"].apply(
                self._estimate_disclose_date
            )

        # 两个条件同时满足:
        # 1. 报告期 <= 查询日 (报告已截止)
        # 2. 披露日 <= 查询日 (数据已公开)
        df["report_date_dt"] = pd.to_datetime(df["report_date"]).dt.date
        df["disclose_date_dt"] = pd.to_datetime(df["disclose_date"]).dt.date

        available = df[
            (df["report_date_dt"] <= query_date) &
            (df["disclose_date_dt"] <= query_date)
        ]

        if available.empty:
            return None

        # 返回最新的
        latest = available.loc[available["report_date_dt"].idxmax()]
        return latest

    def build_pit_financials_series(
        self,
        symbol: str,
        dates: List[date],
        financials_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        为一系列查询日期构建PIT财务时间序列

        Args:
            symbol: 股票代码
            dates: 查询日期列表(通常是每个回测交易日)
            financials_df: 该股票全部财报历史

        Returns:
            DataFrame indexed by query_date, 每个日期对应的"当时可用"最新财务数据
        """
        records = []
        for d in dates:
            latest = self.get_latest_available_report(
                symbol, d, financials_df
            )
            if latest is not None:
                rec = latest.to_dict()
                rec["query_date"] = d
                records.append(rec)

        if not records:
            return pd.DataFrame()

        result = pd.DataFrame(records)
        return result.set_index("query_date")

    @staticmethod
    def _estimate_disclose_date(report_date: date) -> date:
        """
        保守估计财报实际披露日期(最晚法定截止日)

        规则:
        - 12/31年报 → 次年4/30
        - 3/31 Q1 → 4/30
        - 6/30 半年报 → 8/31
        - 9/30 Q3 → 10/31
        """
        m = report_date.month
        d = report_date.day

        if m == 12 and d == 31:
            # 年报: 次年4月30日
            return date(report_date.year + 1, 4, 30)
        elif m == 3 and d == 31:
            # Q1: 当年4月30日
            return date(report_date.year, 4, 30)
        elif m == 6 and d == 30:
            # 半年报: 当年8月31日
            return date(report_date.year, 8, 31)
        elif m == 9 and d == 30:
            # Q3: 当年10月31日
            return date(report_date.year, 10, 31)
        else:
            # 其他: 报告期+60天
            return report_date + timedelta(days=60)

    # ═══════════════════════════════════════════════════════════════
    # 成分股PIT (幸存者偏差消除)
    # ═══════════════════════════════════════════════════════════════

    def register_index_snapshot(
        self,
        index_code: str,
        snapshot_date: date,
        constituents: List[str],
    ):
        """注册某指数在某日期的成分股快照"""
        if index_code not in self._index_constituents:
            self._index_constituents[index_code] = {}
        self._index_constituents[index_code][snapshot_date] = constituents

    def get_constituents_as_of(
        self,
        index_code: str,
        query_date: date,
    ) -> List[str]:
        """
        获取指数在query_date时的成分股列表(无前视偏差)

        查找 <= query_date 的最新快照
        """
        if index_code not in self._index_constituents:
            return []

        snapshots = self._index_constituents[index_code]
        available_dates = sorted(
            [d for d in snapshots.keys() if d <= query_date],
            reverse=True,
        )
        if not available_dates:
            return []

        return snapshots[available_dates[0]]

    # ═══════════════════════════════════════════════════════════════
    # 股票存在性检查 (退市股处理)
    # ═══════════════════════════════════════════════════════════════

    def is_listed_on(
        self,
        symbol: str,
        query_date: date,
        stock_info_df: pd.DataFrame,
    ) -> bool:
        """
        检查某股票在query_date当天是否上市且未退市

        必须保留退市股的历史数据,否则回测会错误地夸大收益
        (因为回测只看到了活下来的好股票)
        """
        info = stock_info_df[stock_info_df["symbol"] == symbol]
        if info.empty:
            return False

        ipo = info["ipo_date"].iloc[0]
        if isinstance(ipo, str):
            ipo = datetime.strptime(ipo, "%Y-%m-%d").date()

        delist = info.get("delist_date")
        if delist is not None:
            delist_val = delist.iloc[0]
            if pd.notna(delist_val):
                if isinstance(delist_val, str):
                    delist_val = datetime.strptime(delist_val, "%Y-%m-%d").date()
                return ipo <= query_date <= delist_val

        return ipo <= query_date

    # ═══════════════════════════════════════════════════════════════
    # 股票池前视偏差过滤
    # ═══════════════════════════════════════════════════════════════

    def filter_universe_pit(
        self,
        candidate_symbols: List[str],
        query_date: date,
        stock_info_df: pd.DataFrame,
        min_listed_days: int = 60,
        min_daily_amount: Optional[float] = None,
    ) -> List[str]:
        """
        按PIT原则过滤股票池:

        1. 排除query_date当日未上市的
        2. 排除query_date当日已退市的
        3. 排除上市不足min_listed_days的(次新股需要足够的K线数据)
        4. (可选) 排除日均成交额过小的
        """
        valid = []
        for sym in candidate_symbols:
            if not self.is_listed_on(sym, query_date, stock_info_df):
                continue

            # 检查上市天数
            info = stock_info_df[stock_info_df["symbol"] == sym]
            ipo = info["ipo_date"].iloc[0]
            if isinstance(ipo, str):
                ipo = datetime.strptime(ipo, "%Y-%m-%d").date()
            if (query_date - ipo).days < min_listed_days:
                continue

            valid.append(sym)

        return valid
