"""
pytest 全局配置 (v2.9)
"""
import sys
from pathlib import Path

import pytest

# 确保项目根在 path 中
sys.path.insert(0, str(Path(__file__).parent.parent))


def pytest_configure(config):
    """统一注册自定义 marker (P2-6)。

    网络/慢测试标记集中于此, 替代 pyproject.toml 中分散的 markers 列表,
    保证 `pytest -m "not network"` 默认跳过联网测试时无 unknown-marker 告警。
    """
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "network: marks tests requiring network access")


@pytest.fixture(scope="session")
def project_root():
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def sample_ohlcv():
    """生成标准模拟K线数据 (200天)"""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(42)
    n = 200
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    close = 50 + np.cumsum(rng.normal(0.02, 0.5, n))
    close = np.maximum(close, 1)

    df = pd.DataFrame({
        "date": dates,
        "open": close + rng.uniform(-0.3, 0.3, n),
        "high": close + rng.uniform(0.1, 0.8, n),
        "low": close - rng.uniform(0.1, 0.8, n),
        "close": close,
        "volume": rng.integers(100000, 5000000, n).astype(float),
    })
    # 确保 high >= open/close >= low
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)
    return df
