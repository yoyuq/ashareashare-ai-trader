"""
数据库初始化脚本

用途:
  1. 创建PostgreSQL数据库和TimescaleDB扩展
  2. 通过SQLAlchemy ORM自动创建所有表
  3. 将kline_daily/kline_minute转换为TimescaleDB hypertable

运行:
  python scripts/init_database.py
"""

import asyncio
import os
import sys
from datetime import date
from pathlib import Path

# 添加项目根目录到path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

from loguru import logger

from data.storage.models import Base, create_db_engine, init_db


def setup_timescaledb(engine):
    """将kline表转换为TimescaleDB hypertable"""
    from sqlalchemy import text

    with engine.connect() as conn:
        try:
            # 启用TimescaleDB扩展
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb"))
            conn.commit()
            logger.info("TimescaleDB扩展已启用")

            # 将kline_daily转为hypertable
            try:
                conn.execute(text(
                    "SELECT create_hypertable('kline_daily', 'date', "
                    "chunk_time_interval => INTERVAL '7 days', "
                    "if_not_exists => TRUE)"
                ))
                conn.commit()
                logger.info("kline_daily → hypertable 完成")
            except Exception as e:
                logger.warning(f"kline_daily hypertable创建: {e}")

            # 将kline_minute转为hypertable
            try:
                conn.execute(text(
                    "SELECT create_hypertable('kline_minute', 'datetime', "
                    "chunk_time_interval => INTERVAL '1 day', "
                    "if_not_exists => TRUE)"
                ))
                conn.commit()
                logger.info("kline_minute → hypertable 完成")
            except Exception as e:
                logger.warning(f"kline_minute hypertable创建: {e}")

            # 设置数据保留策略(可选)
            try:
                conn.execute(text(
                    "SELECT add_retention_policy('kline_daily', "
                    "INTERVAL '10 years', if_not_exists => TRUE)"
                ))
                conn.execute(text(
                    "SELECT add_retention_policy('kline_minute', "
                    "INTERVAL '2 years', if_not_exists => TRUE)"
                ))
                conn.commit()
                logger.info("数据保留策略已设置")
            except Exception as e:
                logger.warning(f"保留策略设置: {e} (如未使用TimescaleDB社区版则忽略)")

        except Exception as e:
            logger.error(f"TimescaleDB设置失败: {e}")


def main():
    logger.info("开始初始化数据库...")

    # 创建引擎
    engine = create_db_engine()
    logger.info(f"数据库引擎创建完成: {engine.url}")

    # 创建所有表
    init_db(engine)
    logger.info("ORM表创建完成")

    # 设置TimescaleDB
    setup_timescaledb(engine)

    # 验证
    with engine.connect() as conn:
        from sqlalchemy import text
        result = conn.execute(text(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public'"
        ))
        tables = [row[0] for row in result]
        logger.info(f"现有表: {tables}")

    logger.info("数据库初始化完成!")


if __name__ == "__main__":
    main()
