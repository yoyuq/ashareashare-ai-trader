<#
.SYNOPSIS
  注册 Windows 任务计划: 每周一 15:30 自动运行自主学习闭环.
.DESCRIPTION
  用 schtasks 注册每周一 15:30 触发 run_weekly_learn.bat (研究官联网 + 忠实测试 + 向量库留/删).
  15:30 选在每日实盘(AITraderDailyLive 15:05)之后, 降低 DeepSeek/CPU 争用.
  幂等: 重复运行会覆盖同名任务 (参数 /f).
  需以当前用户运行 (能访问 .env / 网络). 博查/DeepSeek 国内 API 无需代理.
.PARAMETER Time
  触发时间 (HH:mm), 默认 15:30.
.PARAMETER Unregister
  仅删除任务 (用于停用).
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\register_weekly_learn.ps1
  powershell -ExecutionPolicy Bypass -File scripts\register_weekly_learn.ps1 -Time 16:00
  powershell -ExecutionPolicy Bypass -File scripts\register_weekly_learn.ps1 -Unregister
#>
param(
    [string]$Time = "15:30",
    [switch]$Unregister
)

$TaskName = "AITraderWeeklyLearn"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$Bat = Join-Path $ScriptDir "run_weekly_learn.bat"

if ($Unregister) {
    schtasks /Delete /TN $TaskName /F
    Write-Host "已删除任务: $TaskName"
    exit 0
}

if (-not (Test-Path $Bat)) {
    Write-Error "未找到启动脚本: $Bat"
    exit 1
}

# 校验时间格式
if ($Time -notmatch '^\d{2}:\d{2}$') {
    Write-Error "时间格式应为 HH:mm, 收到: $Time"
    exit 1
}

# schtasks 用 /tr 调 bat; 工作目录由 bat 内 cd /d 修复
schtasks /Create /TN $TaskName `
    /TR "`"$Bat`"" `
    /SC WEEKLY /D MON /ST $Time /F /RL LIMITED

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "已注册任务: $TaskName"
    Write-Host "  触发: 每周一 $Time"
    Write-Host "  脚本: $Bat"
    Write-Host "  日志: $ProjectRoot\reports\weekly_learn.log"
    Write-Host ""
    Write-Host "立即测试一次: schtasks /Run /TN $TaskName"
    Write-Host "查看计划:    schtasks /Query /TN $TaskName /V /FO LIST"
    Write-Host "删除任务:    $MyInvocation.MyCommand.Path -Unregister"
} else {
    Write-Host "注册失败 (exit=$LASTEXITCODE)"
}
