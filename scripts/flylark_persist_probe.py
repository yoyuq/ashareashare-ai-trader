# -*- coding: utf-8 -*-
"""飞书沙箱文件持久性测试 — 决定进化记忆用文件还是多维表格.

用法: 粘到「代码节点」, 配 入参 arg1=任意; 出参 result1 → probe_persist.
**连续运行两次**, 对比两次返回的 history_count:
  - 第二次 > 第一次 → 文件跨运行持久 (进化记忆可用文件! 最简单)
  - 第二次仍为 0 → 每次运行新容器, 文件不持久 (必须用多维表格)
"""
import json


def main(arg1: str) -> dict:
    import os
    from datetime import datetime

    # ===== 固定探针文件路径 =====
    PROBE_PATH = "/app/_persist_probe.txt"

    history = []
    # ===== 尝试读上次的痕迹 =====
    try:
        if os.path.exists(PROBE_PATH):
            with open(PROBE_PATH, "r", encoding="utf-8") as f:
                history = [ln.strip() for ln in f if ln.strip()]
    except Exception as e:
        read_err = f"{type(e).__name__}: {e}"
    else:
        read_err = None

    # ===== 追加本次运行痕迹 =====
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    history.append(f"run:{now} arg1={arg1 or 'None'}")
    try:
        with open(PROBE_PATH, "w", encoding="utf-8") as f:
            f.write("\n".join(history) + "\n")
    except Exception as e:
        write_err = f"{type(e).__name__}: {e}"
    else:
        write_err = None

    # ===== 判断持久性 =====
    if read_err or write_err:
        verdict = f"文件操作异常: read={read_err} write={write_err}"
    elif len(history) >= 2:
        verdict = "✅ 跨运行持久! 上次写入能读到 (进化记忆可用文件)"
    else:
        verdict = "🕐 首跑, 只写了第1条; 请**再跑一次**, 看第2条是否累积"

    result = {
        "history_count": len(history),
        "history": history,
        "probe_path": PROBE_PATH,
        "read_err": read_err,
        "write_err": write_err,
        "verdict": verdict,
    }
    return {"probe_persist": json.dumps(result, ensure_ascii=False, indent=2)}