"""D9 决策 B 落地: AITraderDailyLive 每日 → 每周一 15:05 (研究位).

保留原任务 action (run_daily_live.bat) / Principal / Settings (含电池修复),
仅把 CalendarTrigger 从 ScheduleByDay 换成 ScheduleByWeek Monday,
StartBoundary 重置到下周一以重新武装触发器。经 schtasks /create /xml /f 原地覆盖。
"""
from __future__ import annotations

import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

TASK = "AITraderDailyLive"


def next_monday(today: date) -> date:
    return today + timedelta(days=(7 - today.weekday()) % 7 or 7)


def main() -> int:
    q = subprocess.run(
        ["schtasks", "/query", "/tn", TASK, "/xml"],
        capture_output=True,
    )
    if q.returncode != 0:
        print(q.stderr.decode("utf-16-le", errors="replace") or q.stderr)
        return 1
    # 管道输出是 ASCII/UTF-8 (声明头里的 UTF-16 是任务 XML 内容的编码, 不是管道编码)
    try:
        xml = q.stdout.decode("utf-8")
    except UnicodeDecodeError:
        xml = q.stdout.decode("gbk", errors="replace")
    xml = xml.replace("\r\n", "\n")

    import re
    pat = re.compile(
        r"<ScheduleByDay>\s*<DaysInterval>1</DaysInterval>\s*</ScheduleByDay>"
    )
    new_block = (
        "<ScheduleByWeek>"
        "<DaysOfWeek><Monday /></DaysOfWeek>"
        "<WeeksInterval>1</WeeksInterval>"
        "</ScheduleByWeek>"
    )
    if not pat.search(xml):
        i = xml.find("ScheduleByDay")
        if i == -1:
            i = xml.find("Triggers")
        print("FATAL: 未匹配 ScheduleByDay 块 (任务定义与预期不符, 不改动)")
        print("原文片段:", repr(xml[max(0, i - 80):i + 200]))
        return 2
    xml = pat.sub(new_block, xml)

    # 重置 StartBoundary 到下周一 15:05 (重新武装触发器)
    nm = next_monday(date.today()).isoformat()
    idx = xml.find("<StartBoundary>")
    end = xml.find("</StartBoundary>", idx)
    xml = xml[:idx] + f"<StartBoundary>{nm}T15:05:00" + xml[end:]

    xml_file = r"C:\Users\hjl\ashare-ai-trader\replay_data\schtasks_fix\daily_live_weekly.xml"
    Path(xml_file).parent.mkdir(parents=True, exist_ok=True)
    with open(xml_file, "w", encoding="utf-16") as f:  # utf-16 带 BOM, schtasks /xml 要求
        f.write(xml)

    c = subprocess.run(
        ["schtasks", "/create", "/tn", TASK, "/xml", xml_file, "/f"],
        capture_output=True,
    )
    if c.returncode != 0:
        print(c.stdout.decode("utf-16-le", errors="replace"), c.stderr.decode("utf-16-le", errors="replace"))
        return 3
    print(f"OK: {TASK} 触发器改为 每周一 15:05 (首次 {nm})")

    # self-check
    v = subprocess.run(["schtasks", "/query", "/tn", TASK, "/v", "/fo", "LIST"],
                       capture_output=True)
    txt = v.stdout.decode("gbk", errors="replace")
    for line in txt.splitlines():
        if any(k in line for k in ("下次运行", "Next Run", "计划类型", "Schedule Type",
                                   "要运行的任务", "Task To Run", "状态", "Status",
                                   "上次运行", "Last Run")):
            print("  " + line.strip())
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
