[CmdletBinding()]
param(
    [ValidateSet('Status', 'Canary', 'Arm', 'Disarm', 'SupervisorTick', 'Flatten')]
    [string]$Action = 'Status'
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$rustRoot = Join-Path $repoRoot 'rust_live'
$runRoot = Join-Path $rustRoot 'run'
$reportRoot = Join-Path $rustRoot 'reports\live_active'
# The live configuration is edited, not generated. Change the parameters in
# config/cashcat.toml (it defaults to the grid's `sweep1_flat300` row) and the
# canary and the live service both pick them up. There is no promotion step and
# no derived active-config file.
$liveConfig = Join-Path $rustRoot 'config\cashcat.toml'
$armMarker = Join-Path $runRoot 'cashcat-live.arm'
$canaryPass = Join-Path $runRoot 'live-canary-pass.json'
$composeFile = Join-Path $repoRoot 'docker-compose.live.yml'
$taskName = 'CASHCAT Quota-Aware Live Supervisor'

New-Item -ItemType Directory -Force -Path $runRoot, $reportRoot | Out-Null

function Invoke-LiveFlatten {
    if (-not (Test-Path -LiteralPath $liveConfig)) {
        throw "Live config is missing: $liveConfig"
    }
    docker compose -f $composeFile run --rm --no-deps cashcat-live `
        --config /opt/mm/config/cashcat.toml live-flatten
    if ($LASTEXITCODE -ne 0) {
        throw 'live-flatten failed; live remains stopped'
    }
}

# Liveness only. It restarts a dead or unhealthy container after flattening; it
# does not judge whether the strategy is making money.
#
# It used to: it stopped live unless the best `eligible_for_promotion` row on the
# dry-run leaderboard had positive `promotion_pnl_usdc`. That was coherent while
# `promote-best` generated the live config from exactly that row. Since the live
# config is hand-edited (2026-09-10) the two are unrelated -- config/cashcat.toml
# ships `sweep1_flat300`, whose paper lot-age exit makes it permanently
# ineligible, so the gate was reading a different strategy's P&L (`wide60` on
# 2026-09-10) to decide this one's fate. Removed rather than re-pointed at a
# config-declared row name, which would be a human claim that goes stale on the
# next edit of the config.
#
# The economic stop that remains is first-hand:
# `production_max_daily_realized_loss_usdc` (1 USDC) pauses new placements off the
# live account's own realised P&L, in
# hyperliquid_live.rs::placement_pause_reason. Stopping live is a decision, and
# it is `-Action Disarm`.
function Invoke-SupervisorTick {
    if (-not (Test-Path -LiteralPath $armMarker)) {
        return
    }
    $container = docker inspect cashcat-live --format '{{.State.Status}}|{{if .State.Health}}{{.State.Health.Status}}{{end}}' 2>$null
    if ($LASTEXITCODE -ne 0 -or $container -match 'exited|dead|unhealthy') {
        docker compose -f $composeFile stop cashcat-live | Out-Null
        Invoke-LiveFlatten
        docker compose -f $composeFile up -d --no-deps cashcat-live
        if ($LASTEXITCODE -ne 0) {
            throw 'supervisor could not restart cashcat-live after flattening'
        }
    }
}

switch ($Action) {
    'Status' {
        [pscustomobject]@{
            Armed = Test-Path -LiteralPath $armMarker
            CanaryPassed = Test-Path -LiteralPath $canaryPass
            LiveConfig = $liveConfig
            Container = docker inspect cashcat-live --format '{{json .State}}' 2>$null
        } | ConvertTo-Json -Depth 8
    }
    'Flatten' {
        docker compose -f $composeFile stop cashcat-live | Out-Null
        Invoke-LiveFlatten
    }
    'Canary' {
        $stamp = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()
        $reportName = "canary-$stamp.json"
        docker compose -f $composeFile run --rm --no-deps cashcat-live `
            --config /opt/mm/config/cashcat.toml live `
            --duration-seconds 7200 --report "/opt/mm/reports/live_active/$reportName"
        $runExit = $LASTEXITCODE
        Invoke-LiveFlatten
        if ($runExit -ne 0) {
            throw "two-hour canary exited with code $runExit"
        }
        $reportPath = Join-Path $reportRoot $reportName
        $report = Get-Content -LiteralPath $reportPath -Raw | ConvertFrom-Json
        $durationMs = [int64]$report.finished_at_ms - [int64]$report.started_at_ms
        if ($durationMs -lt 7190000 -or
            [int64]$report.execution.fills -lt 1 -or
            [int64]$report.execution.unknown_outcomes -ne 0 -or
            [int64]$report.execution.orders_rejected -ne 0 -or
            [int64]$report.account.inventory_units -ne 0 -or
            $report.stop_reason -ne 'duration_elapsed' -or
            $report.shutdown_succeeded -ne $true -or
            $report.operationally_valid -ne $true) {
            throw 'canary evidence did not satisfy duration/fill/operational/flatness gates'
        }
        $evidence = [ordered]@{
            schema_version = 1
            passed_at = [DateTimeOffset]::Now.ToString('O')
            report = $reportPath
            sha256 = (Get-FileHash -LiteralPath $reportPath -Algorithm SHA256).Hash.ToLowerInvariant()
            duration_ms = $durationMs
            fills = [int64]$report.execution.fills
            final_inventory_units = [int64]$report.account.inventory_units
            address_requests_used = [int64]$report.execution.address_requests_used
            address_requests_cap = [int64]$report.execution.address_requests_cap
        }
        $evidence | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $canaryPass
    }
    'Arm' {
        if (-not (Test-Path -LiteralPath $canaryPass)) {
            throw 'A successful two-hour canary is required before arming.'
        }
        Set-Content -LiteralPath $armMarker -Value ([DateTimeOffset]::Now.ToString('O'))
        $taskAction = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument (
            "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`" -Action SupervisorTick"
        )
        $startup = New-ScheduledTaskTrigger -AtStartup
        $repeat = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) `
            -RepetitionInterval ([TimeSpan]::FromMinutes(1)) `
            -RepetitionDuration ([TimeSpan]::FromDays(3650))
        Register-ScheduledTask -TaskName $taskName -Action $taskAction `
            -Trigger @($startup, $repeat) -RunLevel Highest -Force | Out-Null
        Invoke-SupervisorTick
    }
    'Disarm' {
        if (Test-Path -LiteralPath $armMarker) {
            Remove-Item -LiteralPath $armMarker -Force
        }
        Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue
        docker compose -f $composeFile stop cashcat-live | Out-Null
        Invoke-LiveFlatten
    }
    'SupervisorTick' { Invoke-SupervisorTick }
}
