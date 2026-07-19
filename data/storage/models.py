"""
数据库模型 — SQLAlchemy ORM (PostgreSQL + TimescaleDB)

分层存储策略:
  - realtime_quotes: 最新行情快照(非时序,只保留最新一条)
  - kline_daily: 日K线时序主表(TimescaleDB hypertable)
  - kline_minute: 分钟K线时序表
  - stock_info: 股票基本信息(维度表)
  - financials: 财务指标表
  - signal_log: 交易信号归档(v2.0)
  - alternative_data: 另类数据快照(v2.1)
"""

from datetime import date, datetime
from typing import Optional

from sqlalchemy import (
    Column,
    Date,
    DateTime,
    Float,
    Integer,
    BigInteger,
    String,
    Text,
    Boolean,
    Index,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session


class Base(DeclarativeBase):
    pass


# ═══════════════════════════════════════════════════════════════
# 股票基本信息 (维度表)
# ═══════════════════════════════════════════════════════════════

class StockInfo(Base):
    __tablename__ = "stock_info"

    symbol = Column(String(20), primary_key=True, comment="股票代码(sh.600000)")
    name = Column(String(20), nullable=False, comment="股票名称")
    exchange = Column(String(10), comment="交易所(sh/sz/bj)")
    ipo_date = Column(Date, comment="上市日期")
    delist_date = Column(Date, nullable=True, comment="退市日期")
    industry = Column(String(50), comment="申万一级行业")
    sub_industry = Column(String(50), comment="申万二级行业")
    total_mv = Column(Float, comment="总市值(亿元)")
    float_mv = Column(Float, comment="流通市值(亿元)")
    is_st = Column(Boolean, default=False, comment="是否ST")
    board = Column(String(20), comment="板块(主板/创业板/科创板/北交所)")
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)


# ═══════════════════════════════════════════════════════════════
# 日K线 (TimescaleDB hypertable)
# ═══════════════════════════════════════════════════════════════

class KlineDaily(Base):
    __tablename__ = "kline_daily"

    symbol = Column(String(20), primary_key=True, comment="股票代码")
    date = Column(Date, primary_key=True, comment="交易日期")
    open = Column(Float, comment="开盘价")
    high = Column(Float, comment="最高价")
    low = Column(Float, comment="最低价")
    close = Column(Float, comment="收盘价")
    pre_close = Column(Float, comment="前收盘价")
    volume = Column(Float, comment="成交量(股)")
    amount = Column(Float, comment="成交额(元)")
    turnover = Column(Float, comment="换手率(%)")
    pct_change = Column(Float, comment="涨跌幅(%)")
    amplitude = Column(Float, comment="振幅(%)")
    is_st = Column(Boolean, default=False, comment="当日是否ST")
    is_trade = Column(Boolean, default=True, comment="是否有交易")
    adj_factor = Column(Float, default=1.0, comment="复权因子")
    source = Column(String(20), comment="数据来源")

    __table_args__ = (
        Index("idx_kline_daily_date", "date"),
        Index("idx_kline_daily_symbol_date", "symbol", "date"),
        {"comment": "A股日K线时序数据"},
    )


# ═══════════════════════════════════════════════════════════════
# 分钟K线
# ═══════════════════════════════════════════════════════════════

class KlineMinute(Base):
    __tablename__ = "kline_minute"

    symbol = Column(String(20), primary_key=True)
    datetime = Column(DateTime, primary_key=True, comment="交易时间")
    frequency = Column(String(10), primary_key=True, comment="频率(1/5/15/30/60min)")
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)
    amount = Column(Float)

    __table_args__ = (
        Index("idx_kline_minute_datetime", "datetime"),
        {"comment": "A股分钟K线时序数据"},
    )


# ═══════════════════════════════════════════════════════════════
# 实时行情快照 (非时序,每只股票一条记录)
# ═══════════════════════════════════════════════════════════════

class RealtimeQuote(Base):
    __tablename__ = "realtime_quote"

    symbol = Column(String(20), primary_key=True)
    name = Column(String(20))
    price = Column(Float, comment="最新价")
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    pre_close = Column(Float)
    volume = Column(Float, comment="成交量")
    amount = Column(Float, comment="成交额")
    pct_change = Column(Float, comment="涨跌幅")
    turnover = Column(Float, comment="换手率")
    bid1 = Column(Float, comment="买一价")
    ask1 = Column(Float, comment="卖一价")
    bid_vol1 = Column(Float, comment="买一量")
    ask_vol1 = Column(Float, comment="卖一量")
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)


# ═══════════════════════════════════════════════════════════════
# 财务指标
# ═══════════════════════════════════════════════════════════════

class Financials(Base):
    __tablename__ = "financials"

    symbol = Column(String(20), primary_key=True)
    report_date = Column(Date, primary_key=True, comment="报告期")
    eps = Column(Float, comment="每股收益")
    bps = Column(Float, comment="每股净资产")
    roe = Column(Float, comment="净资产收益率(%)")
    roa = Column(Float, comment="总资产收益率(%)")
    gross_margin = Column(Float, comment="毛利率(%)")
    net_margin = Column(Float, comment="净利率(%)")
    revenue_yoy = Column(Float, comment="营收同比(%)")
    profit_yoy = Column(Float, comment="净利润同比(%)")
    debt_ratio = Column(Float, comment="资产负债率(%)")
    current_ratio = Column(Float, comment="流动比率")
    pe_ttm = Column(Float, comment="市盈率TTM")
    pb = Column(Float, comment="市净率")
    total_mv = Column(Float, comment="总市值")

    __table_args__ = (
        Index("idx_financials_report_date", "report_date"),
    )


# ═══════════════════════════════════════════════════════════════
# v2.0: 交易信号归档
# ═══════════════════════════════════════════════════════════════

class SignalLog(Base):
    __tablename__ = "signal_log"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False)
    signal_date = Column(Date, nullable=False, comment="信号发出日期")
    direction = Column(String(10), comment="long/short/flat")
    confidence = Column(Float, comment="置信度(0-1)")
    entry_price = Column(Float, comment="建议入场价")
    stop_loss = Column(Float, comment="止损价")
    take_profit = Column(Float, comment="止盈价")
    strategy_id = Column(String(50), comment="策略ID")
    market_regime = Column(String(30), comment="市场状态")
    debate_bull_score = Column(Float, comment="多头评分(v2.1)")
    debate_bear_score = Column(Float, comment="空头评分(v2.1)")
    debate_result = Column(Text, comment="辩论摘要(v2.1)")
    report_snippet = Column(Text, comment="研判报告摘要")
    actual_result_pct = Column(Float, nullable=True, comment="N日后实际涨跌幅(回看用)")
    reviewed = Column(Boolean, default=False, comment="是否已回看")
    created_at = Column(DateTime, default=datetime.now)

    __table_args__ = (
        Index("idx_signal_symbol_date", "symbol", "signal_date"),
        Index("idx_signal_strategy", "strategy_id", "signal_date"),
        Index("idx_signal_reviewed", "reviewed"),
    )


# ═══════════════════════════════════════════════════════════════
# v2.1: 另类数据快照
# ═══════════════════════════════════════════════════════════════

class AlternativeDataSnapshot(Base):
    __tablename__ = "alternative_data_snapshot"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False)
    snapshot_date = Column(Date, nullable=False)
    irm_qa_count = Column(Integer, comment="互动易问答条数")
    irm_avg_response_hours = Column(Float, comment="平均回复速度")
    shareholder_trend = Column(String(20), comment="筹码趋势")
    shareholder_change_pct = Column(Float, comment="股东户数变化%")
    margin_trend = Column(String(20), comment="融资趋势")
    avg_margin_buy_ratio = Column(Float, comment="平均融资买入占比")
    heat_rank = Column(Integer, comment="热度排名")
    upcoming_unlock_mv_ratio = Column(Float, comment="解禁市值占比")
    block_trade_premium = Column(Float, comment="大宗交易平均溢价率")
    created_at = Column(DateTime, default=datetime.now)

    __table_args__ = (
        Index("idx_altdata_symbol_date", "symbol", "snapshot_date"),
    )


# ═══════════════════════════════════════════════════════════════
# v2.1: 因子IC衰减日志
# ═══════════════════════════════════════════════════════════════

class FactorICLog(Base):
    __tablename__ = "factor_ic_log"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    factor_name = Column(String(50), nullable=False)
    calc_date = Column(Date, nullable=False)
    ic_value = Column(Float, comment="当期IC值")
    ic_cumulative = Column(Float, comment="累积IC")
    ir = Column(Float, comment="信息比率(IR)")
    decay_prob = Column(Float, comment="贝叶斯衰减概率")
    is_active = Column(Boolean, default=True, comment="因子是否活跃")
    retired_date = Column(Date, nullable=True, comment="退休日期")
    created_at = Column(DateTime, default=datetime.now)

    __table_args__ = (
        Index("idx_ic_factor_date", "factor_name", "calc_date"),
        UniqueConstraint("factor_name", "calc_date", name="uq_factor_ic_date"),
    )


# ═══════════════════════════════════════════════════════════════
# 数据库引擎工厂
# ═══════════════════════════════════════════════════════════════

def create_db_engine(database_url: Optional[str] = None):
    """创建数据库引擎(优先TimescaleDB,备选标准PostgreSQL)"""
    if database_url is None:
        import os
        user = os.getenv("POSTGRES_USER", "ashare")
        password = os.getenv("POSTGRES_PASSWORD", "ashare")
        host = os.getenv("POSTGRES_HOST", "localhost")
        port = os.getenv("POSTGRES_PORT", "5432")
        db = os.getenv("POSTGRES_DB", "ashare_trader")
        database_url = f"postgresql+psycopg://{user}:{password}@{host}:{port}/{db}"

    engine = create_engine(
        database_url,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
        echo=False,
    )
    return engine


def init_db(engine) -> None:
    """创建所有表"""
    Base.metadata.create_all(engine)


def get_session(engine) -> Session:
    """获取数据库Session"""
    return Session(engine)
