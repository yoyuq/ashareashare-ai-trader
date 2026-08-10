@echo off
REM ============================================================
REM  A股 AI 交易 — 每日实盘(纸面)运行脚本  (v5.4)
REM  完整 LLM 模式 + 自我进化系统
REM  由 Windows 任务计划每天 15:30 触发 (见 register_daily_task.ps1)
REM ============================================================
chcp 65001 >nul
cd /d "C:\Users\hjl\ashare-ai-trader"

REM 数据源(akshare/东财)需要代理; 腾讯/baostock 免代理回退
set "HTTP_PROXY=http://127.0.0.1:7897"
set "HTTPS_PROXY=http://127.0.0.1:7897"
set "PYTHONIOENCODING=utf-8"

REM 追加日志 (含日期), 失败也记录退出码
set "LOG=simulation_data\daily_live.log"
echo [%date% %time%] ===== daily run start ===== >> "%LOG%"
python -m simulation.daily_runner >> "%LOG%" 2>&1
set "EXIT=%ERRORLEVEL%"
echo [%date% %time%] ===== daily run end (exit=%EXIT%) ===== >> "%LOG%"
exit /b %EXIT%