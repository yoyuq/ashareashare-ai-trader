@echo off
REM ============================================================
REM  抓取 2015/2016/2017/2021/2022/2023 全市场日K (Baostock, 断点续跑)
REM  由 Windows 任务计划触发 (独立于 Claude 沙箱, 避免长跑进程被回收)
REM  脚本内置: 串行节流 + 每200只重连 + 中途保存 + 断点续跑 + 按上市日期过滤
REM ============================================================
chcp 65001 >nul
cd /d "C:\Users\hjl\ashare-ai-trader"

set "PYTHONIOENCODING=utf-8"

REM 追加日志 (含日期), 失败也记录退出码
set "LOG=replay_data\fetch_full_cycle.log"
echo [%date% %time%] ===== fetch full cycle start ===== >> "%LOG%"
python -u scripts\fetch_full_cycle_windows.py >> "%LOG%" 2>&1
set "EXIT=%ERRORLEVEL%"
echo [%date% %time%] ===== fetch full cycle end (exit=%EXIT%) ===== >> "%LOG%"
exit /b %EXIT%
