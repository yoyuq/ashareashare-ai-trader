<#
.SYNOPSIS
  注册 Windows 任务计划: 每个交易日 15:20 自动跑前瞻失效监控 (refresh缓存 + forward_track).
.DESCRIPTION
  用 schtasks 注册 AITraderForwardTrack, 触发 run_forward_track.bat.
  15:00 收盘后、daily_runner (15:05) 之后跑, 用当天收盘价跟踪已注册 edge (冷落 beta).
  幂等: 重复运行覆盖同名任务 (/f). 与 daily_runner 隔离 (实验探针不混进核心交易路径).
  腾讯行情免代理, bat 内不设 HTTP_PROXY.
.PARAMETER Time
  触发时间 (HH:mm), 默认 15:20.
.PARAMETER Unregister
  仅删除任务 (用于停用).
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\register_forward_track.ps1
  powershell -ExecutionPolicy Bypass -File scripts\register_forward_track.ps1 -Unregister
#>
param(
    [string]$Time = "15:20",
    [switch]$Unregister
)

$TaskName = "AITraderForwardTrack"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$Bat = Join-Path $ScriptDir "run_forward_track.bat"

if ($Unregister) {
    schtasks /Delete /TN $TaskName /F
    Write-Host "已删除任务: $TaskName"
    exit 0
}

if (-not (Test-Path $Bat)) {
    Write-Error "未找到启动脚本: $Bat"
    exit 1
}

if ($Time -notmatch '^\d{2}:\d{2}$') {
    Write-Error "时间格式应为 HH:mm, 收到: $Time"
    exit 1
}

schtasks /Create /TN $TaskName `
    /TR "`"$Bat`"" `
    /SC DAILY /ST $Time /F /RL LIMITED

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "已注册任务: $TaskName"
    Write-Host "  触发: 每天 $Time"
    Write-Host "  脚本: $Bat"
    Write-Host "  日志: $ProjectRoot\simulation_data\forward_validation.log"
    Write-Host ""
    Write-Host "立即测试一次: schtasks /Run /TN $TaskName"
    Write-Host "查看计划:    schtasks /Query /TN $TaskName /V /FO LIST"
    Write-Host "删除任务:    $MyInvocation.MyCommand.Path -Unregister"
} else {
    Write-Host "注册失败 (exit=$LASTEXITCODE)"
}
