# Fix AITrader scheduled tasks (ASCII-only version, 2026-09-06)
# Defects fixed:
#   1. AITraderMinuteCollector registered as one-shot TimeTrigger -> re-register:
#      daily 09:25, repeat every 10min until 15:00 (StopAtDurationEnd)
#   2. AITraderDailyLive / AITraderForwardTrack: DisallowStartIfOnBatteries=True
#      -> battery-powered laptop rejects trigger (0x800710E0) -> flip to
#      AllowStartIfOnBatteries + StartWhenAvailable
# Idempotent. Run as current user (hjl), no admin needed.

$ErrorActionPreference = "Stop"
$Python = "C:\Users\hjl\AppData\Local\Python\pythoncore-3.14-64\python.exe"

# ---- 1. MinuteCollector: daily 09:25 + 10min repetition x5h35m (XML, most reliable) ----
$xml = @"
<?xml version="1.0" encoding="UTF-16"?>
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
      <Command>"$Python"</Command>
      <Arguments>"C:\Users\hjl\ashare-ai-trader\scripts\minute_collector.py"</Arguments>
    </Exec>
  </Actions>
</Task>
"@
$xmlFile = Join-Path $env:TEMP "aitrader_minute_collector.xml"
$xml | Out-File -FilePath $xmlFile -Encoding Unicode
schtasks /create /tn "AITraderMinuteCollector" /xml "$xmlFile" /f | Out-Null
if ($LASTEXITCODE -ne 0) { throw "schtasks create failed with exit code $LASTEXITCODE" }
Write-Host "OK: AITraderMinuteCollector re-registered (daily 09:25 + 10min x5h35m)"

# ---- 2/3. DailyLive + ForwardTrack: flip battery settings + catch-up on missed trigger ----
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable
foreach ($name in "AITraderDailyLive", "AITraderForwardTrack") {
    Set-ScheduledTask -TaskName $name -Settings $settings | Out-Null
    Write-Host "OK: $name battery/catch-up settings flipped"
}

# ---- Self-check ----
Write-Host ""
Write-Host "=== SELF-CHECK ==="
foreach ($n in "AITraderDailyLive", "AITraderForwardTrack", "AITraderMinuteCollector") {
    $t = Get-ScheduledTask -TaskName $n
    $info = Get-ScheduledTaskInfo -TaskName $n
    $rep = ""
    if ($t.Triggers[0].Repetition.Interval) { $rep = " rep=$($t.Triggers[0].Repetition.Interval) dur=$($t.Triggers[0].Repetition.Duration)" }
    Write-Host ("{0}: enabled={1} battery-off={2} startAvail={3}{4} lastRun={5} lastResult={6}" -f $n, $t.Settings.Enabled, (-not $t.Settings.DisallowStartIfOnBatteries), $t.Settings.StartWhenAvailable, $rep, $info.LastRunTime, $info.LastTaskResult)
}
