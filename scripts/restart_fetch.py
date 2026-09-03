"""抓取进程一键重启: 杀旧进程 → 清 checkpoint fail 污染 → 加固版代码分离重启。

用法: python scripts/restart_fetch.py
"""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
CKPT = ROOT / "replay_data" / "inst_holdings_checkpoint.json"
LOG = ROOT / "reports" / "agent_loop" / "fetch_inst.log"
ERR = ROOT / "reports" / "agent_loop" / "fetch_inst.err"


def main() -> int:
    # 1. 杀所有旧 fetch 进程
    out = subprocess.run(["taskkill", "/F", "/FI", "WINDOWTITLE eq *"],
                         capture_output=True, text=True)
    r = subprocess.run(["wmic", "process", "where",
                        "name='python.exe'", "get", "processid,commandline", "/format:csv"],
                       capture_output=True, text=True)
    killed = 0
    for line in r.stdout.splitlines():
        low = line.lower()
        if "fetch_inst_holdings" in low and "restart_fetch" not in low:
            pid = line.rstrip().split(",")[-1]
            if pid.isdigit():
                subprocess.run(["taskkill", "/F", "/PID", pid], capture_output=True)
                killed += 1
                print(f"killed old fetch pid={pid}")
    time.sleep(1.0)

    # 2. 清 fail 污染 (fail≠无覆盖, 预注册口径只认 ok/empty)
    if CKPT.exists():
        c = json.loads(CKPT.read_text(encoding="utf-8"))
        n0 = len(c)
        c = {k: v for k, v in c.items() if v != "fail"}
        CKPT.write_text(json.dumps(c, ensure_ascii=False), encoding="utf-8")
        print(f"purged {n0 - len(c)} fail entries; remaining {len(c)}/730")
    else:
        print("no checkpoint (fresh start)")

    # 3. 分离重启 (DETACHED_PROCESS, 不挂会话任务)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "ab") as fo, open(ERR, "ab") as fe:
        subprocess.Popen(
            [sys.executable, str(ROOT / "scripts" / "fetch_inst_holdings.py")],
            stdout=fo, stderr=fe, cwd=str(ROOT),
            creationflags=0x00000008 | 0x00000200,  # DETACHED_PROCESS | NEW_PROCESS_GROUP
        )
    print(f"relaunched detached; log -> {LOG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
