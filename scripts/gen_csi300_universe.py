"""生成 CSI300 大市值子集 (幸存者偏差稳健性参考).

用当前快照 full_market_cache.json 的 total_mv 取前 N 只 (排除北交所 bj),
作为 CSI300 大市值 proxy. 大市值=退市少 → 幸存者偏差最小化.
输出: replay_data/csi300_universe.txt (每行一个 symbol)

用法:
  python scripts/gen_csi300_universe.py [--top 300]
"""

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=300)
    args = ap.parse_args()

    cache = Path("simulation_data/full_market_cache.json")
    if not cache.exists():
        print(f"缺少快照 {cache}")
        return 1
    d = json.loads(cache.read_text(encoding="utf-8"))

    rows = []
    for item in d.get("data", []):
        code = str(item.get("code", ""))
        if not code or not code.isdigit():
            continue
        prefix = "sh" if code.startswith("6") else ("bj" if code.startswith(("8", "4")) else "sz")
        if prefix == "bj":
            continue  # 排除北交所
        mv = float(item.get("total_mv", 0) or 0)
        rows.append((f"{prefix}.{code}", mv))

    rows.sort(key=lambda x: x[1], reverse=True)
    top = [s for s, mv in rows[: args.top]]
    out = Path("replay_data/csi300_universe.txt")
    out.write_text("\n".join(top), encoding="utf-8")
    print(f"CSI300 大市值子集: {len(top)} 只 → {out}")
    print(f"  市值中位: {rows[args.top//2][1]:.0f} 亿 | 龙头: {top[0]}")


if __name__ == "__main__":
    raise SystemExit(main())