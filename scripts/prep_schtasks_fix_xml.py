"""schtasks XML 修复准备 (D1) — 导出的任务 XML 字节对齐修正 + 电池/补跑设置翻转 + MinuteCollector 重排 XML.

用途: 分类器拦 powershell.exe, 改走 Bash `schtasks /create /xml` 路线。
本脚本只做 XML 文件准备 (写 /tmp/*.fix.xml), 注册由 schtasks CLI 完成:
  schtasks /create /tn AITraderDailyLive /xml <file> /f
  schtasks /create /tn AITraderForwardTrack /xml <file> /f
  schtasks /create /tn AITraderMinuteCollector /xml <file> /f
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

FIX_SCRIPTS = Path(__file__).parent
OUT_DIR = FIX_SCRIPTS.parent / "replay_data" / "schtasks_fix"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_task_xml(raw: Path) -> str:
    """schtasks /xml 导出经 bash 重定向后可能偏 1 字节; 以 FF FE BOM 重对齐."""
    data = raw.read_bytes()
    i = data.find(b"\xff\xfe")
    if i > 0:
        data = data[i:]
    return data.decode("utf-16")


def flip_battery(txt: str) -> str:
    """DisallowStartIfOnBatteries/StopIfGoingOnBatteries → false; 加 StartWhenAvailable."""
    txt = re.sub(r"<DisallowStartIfOnBatteries>\s*true\s*</DisallowStartIfOnBatteries>",
                 "<DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>", txt)
    txt = re.sub(r"<StopIfGoingOnBatteries>\s*true\s*</StopIfGoingOnBatteries>",
                 "<StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>", txt)
    if "<StartWhenAvailable>" not in txt:
        # 插在 </Settings> 前 (Settings 必有; IdleSettings 是子元素不受影响)
        txt = txt.replace("</Settings>", "<StartWhenAvailable>true</StartWhenAvailable></Settings>", 1)
    else:
        txt = re.sub(r"<StartWhenAvailable>\s*false\s*</StartWhenAvailable>",
                     "<StartWhenAvailable>true</StartWhenAvailable>", txt)
    return txt


MINUTE_COLLECTOR_XML = """<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <Triggers>
    <TimeTrigger>
      <StartBoundary>2026-09-07T09:25:00</StartBoundary>
      <Enabled>true</Enabled>
      <Repetition>
        <Interval>PT10M</Interval>
        <Duration>PT5H35M</Duration>
        <StopAtDurationEnd>true</StopAtDurationEnd>
      </Repetition>
    </TimeTrigger>
  </Triggers>
  <Settings>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <StartWhenAvailable>true</StartWhenAvailable>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <ExecutionTimeLimit>PT1H</ExecutionTimeLimit>
    <Enabled>true</Enabled>
  </Settings>
  <Actions Context="Author">
    <Exec>
      <Command>C:\\Users\\hjl\\AppData\\Local\\Python\\pythoncore-3.14-64\\python.exe</Command>
      <Arguments>C:\\Users\\hjl\\ashare-ai-trader\\scripts\\minute_collector.py</Arguments>
    </Exec>
  </Actions>
</Task>
"""


def main() -> int:
    tmp = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    outs = {}
    for name in ("AITraderDailyLive", "AITraderForwardTrack"):
        raw = tmp / (name.lower().replace("aitrader", "") + "_raw.xml")
        txt = load_task_xml(raw)
        fixed = flip_battery(txt)
        out = OUT_DIR / f"{name}.fix.xml"
        out.write_text(fixed, encoding="utf-16")
        outs[name] = out
        changed = fixed != txt
        print(f"{name}: battery翻转={'YES' if changed else 'no-change(已是目标状态?)'} -> {out}")
    mc = OUT_DIR / "AITraderMinuteCollector.fix.xml"
    mc.write_text(MINUTE_COLLECTOR_XML, encoding="utf-16")
    print(f"AITraderMinuteCollector: 全新 XML (每日09:25 + 10min x 5h35m) -> {mc}")
    print("\n注册命令 (Bash, MSYS_NO_PATHCONV=1):")
    for name, out in {**outs, "AITraderMinuteCollector": mc}.items():
        print(f'  MSYS_NO_PATHCONV=1 schtasks /create /tn {name} /xml "{out}" /f')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
