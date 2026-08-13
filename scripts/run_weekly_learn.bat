@echo off
REM ============================================================
REM  A股 AI 交易 — 每周自主学习闭环  (v5.6)
REM  研究官联网(Bocha) → RAG查重 → 忠实测试(真实回放) → 向量库留/删
REM  由 Windows 任务计划每周一 15:30 触发 (见 register_weekly_learn.ps1)
REM  博查/DeepSeek 均为国内 API, 无需代理 (冒烟已验证)
REM ============================================================
chcp 65001 >nul
cd /d "C:\Users\hjl\ashare-ai-trader"

set "PYTHONIOENCODING=utf-8"

REM 追加日志 (含日期), 失败也记录退出码
set "LOG=reports\weekly_learn.log"
echo [%date% %time%] ===== weekly learn start ===== >> "%LOG%"
python scripts\learn_external.py --auto 3 >> "%LOG%" 2>&1
set "EXIT=%ERRORLEVEL%"
echo [%date% %time%] ===== weekly learn end (exit=%EXIT%) ===== >> "%LOG%"
exit /b %EXIT%
