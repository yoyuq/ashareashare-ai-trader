@echo off
REM ============================================================
REM  Forward paper-validation daily monitor (v5.6 #108/#109)
REM  refresh full-market snapshot -> track registered edge(s) vs
REM  matched universe vs SH index. Isolated from daily_runner so
REM  the experiment probe never mixes into the core trading path.
REM  Triggered daily 15:20 by Windows Task Scheduler (AITraderForwardTrack).
REM ============================================================
chcp 65001 >nul
cd /d "C:\Users\hjl\ashare-ai-trader"

set "PYTHONIOENCODING=utf-8"

REM Tencent quote feed needs no proxy; do NOT set HTTP_PROXY (proxy outage would kill the monitor)
set "LOG=simulation_data\forward_validation.log"
echo [%date% %time%] ===== forward track start ===== >> "%LOG%"

python scripts\refresh_market_cache.py >> "%LOG%" 2>&1
set "RC1=%ERRORLEVEL%"

python scripts\forward_track.py >> "%LOG%" 2>&1
set "RC2=%ERRORLEVEL%"

echo [%date% %time%] ===== forward track end (refresh=%RC1%, track=%RC2%) ===== >> "%LOG%"
exit /b %RC2%
