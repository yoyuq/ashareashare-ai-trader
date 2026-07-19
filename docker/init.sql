-- A股智能分析Agent — 数据库初始化脚本
-- 自动创建TimescaleDB扩展和hypertable

-- 启用TimescaleDB扩展
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- 日K线时序表 (需在应用创建表后执行)
-- SELECT create_hypertable('kline_daily', 'date', chunk_time_interval => INTERVAL '7 days');
-- CREATE INDEX IF NOT EXISTS idx_kline_daily_symbol_date ON kline_daily (symbol, date DESC);

-- 分钟K线时序表
-- SELECT create_hypertable('kline_minute', 'datetime', chunk_time_interval => INTERVAL '1 day');
-- CREATE INDEX IF NOT EXISTS idx_kline_minute_symbol_datetime ON kline_minute (symbol, datetime DESC);

-- 数据保留策略(可选):
-- SELECT add_retention_policy('kline_daily', INTERVAL '10 years');
-- SELECT add_retention_policy('kline_minute', INTERVAL '2 years');

-- 注意: 表结构由SQLAlchemy ORM自动创建,此文件仅作TimescaleDB扩展初始化
