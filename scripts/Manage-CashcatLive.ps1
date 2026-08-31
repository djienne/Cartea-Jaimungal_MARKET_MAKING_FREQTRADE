[CmdletBinding()]
param(
    [ValidateSet('Status', 'Promote', 'Canary', 'Arm', 'Disarm', 'SupervisorTick', 'Flatten')]
    [string]$Action = 'Status'
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$rustRoot = Join-Path $repoRoot 'rust_live'
$runRoot = Join-Path $rustRoot 'run'
$reportRoot = Join-Path $rustRoot 'reports\live_active'
$binary = Join-Path $rustRoot 'target\release\mm-live.exe'
$baseConfig = Join-Path $rustRoot 'config\cashcat_dryrun_realistic.toml'
$gridSpec = Join-Path $rustRoot 'config\grid_cashcat.toml'
$leaderboard = Join-Path $rustRoot 'reports\grid_live\leaderboard.json'
$activeConfig = Join-Path $runRoot 'cashcat-active-live.toml'
$manifest = Join-Path $runRoot 'cashcat-promotion.json'
$armMarker = Join-Path $runRoot 'cashcat-live.arm'
$canaryPass = Join-Path $runRoot 'live-canary-pass.json'
$lastCheck = Join-Path $runRoot 'last-promotion-check.txt'
$composeFile = Join-Path $repoRoot 'docker-compose.live.yml'
$taskName = 'CASHCAT Quota-Aware Live Supervisor'

New-Item -ItemType Directory -Force -Path $runRoot, $reportRoot | Out-Null

function Invoke-MmLive {
    param([string[]]$Arguments)
    if (-not (Test-Path -LiteralPath $binary)) {
        throw "Release binary is missing: $binary"
    }
    & $binary @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "mm-live failed with exit code $LASTEXITCODE"
    }
}

function Invoke-LiveFlatten {
    if (-not (Test-Path -LiteralPath $activeConfig)) {
        throw "Active live config is missing: $activeConfig"
    }
    docker compose -f $composeFile run --rm --no-deps cashcat-live `
        --config /opt/mm/run/cashcat-active-live.toml live-flatten
    if ($LASTEXITCODE -ne 0) {
        throw 'live-flatten failed; live remains stopped'
    }
}

function Invoke-Promotion {
    $backupConfig = "$activeConfig.previous"
    $backupManifest = "$manifest.previous"
    if (Test-Path -LiteralPath $activeConfig) {
        Copy-Item -LiteralPath $activeConfig -Destination $backupConfig -Force
    }
    if (Test-Path -LiteralPath $manifest) {
        Copy-Item -LiteralPath $manifest -Destination $backupManifest -Force
    }
    try {
        Invoke-MmLive @(
            '--config', $baseConfig,
            'promote-best',
            '--grid', $gridSpec,
            '--leaderboard', $leaderboard,
            '--output', $activeConfig,
            '--manifest', $manifest,
            '--min-elapsed-seconds', '43200'
        )
        $promotion = Get-Content -LiteralPath $manifest -Raw | ConvertFrom-Json
        if (-not $promotion.changed) {
            return $false
        }
        if (-not (Test-Path -LiteralPath $armMarker)) {
            # Initial promotion only stages the config. The two-hour canary and
            # explicit Arm gate must run before any continuous live service.
            return $true
        }
        docker compose -f $composeFile stop cashcat-live
        Invoke-LiveFlatten
        docker compose -f $composeFile up -d --no-deps cashcat-live
        if ($LASTEXITCODE -ne 0) {
            throw 'failed to start the promoted live service'
        }
        return $true
    }
    catch {
        if (Test-Path -LiteralPath $backupConfig) {
            Copy-Item -LiteralPath $backupConfig -Destination $activeConfig -Force
        }
        if (Test-Path -LiteralPath $backupManifest) {
            Copy-Item -LiteralPath $backupManifest -Destination $manifest -Force
        }
        docker compose -f $composeFile stop cashcat-live | Out-Null
        throw
    }
}

function Invoke-SupervisorTick {
    if (-not (Test-Path -LiteralPath $armMarker)) {
        return
    }
    $board = Get-Content -LiteralPath $leaderboard -Raw | ConvertFrom-Json
    $best = $board.rows | Where-Object eligible_for_promotion |
        Sort-Object promotion_pnl_usdc -Descending | Select-Object -First 1
    if ($null -eq $best -or [double]$best.promotion_pnl_usdc -le 0.0) {
        $running = docker ps --filter name=cashcat-live --format '{{.Names}}'
        if ($running) {
            docker compose -f $composeFile stop cashcat-live | Out-Null
            Invoke-LiveFlatten
        }
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
    $promotionDue = -not (Test-Path -LiteralPath $lastCheck)
    if (-not $promotionDue) {
        $promotionDue = (Get-Date) - (Get-Item -LiteralPath $lastCheck).LastWriteTime -ge [TimeSpan]::FromHours(12)
    }
    if ($promotionDue) {
        Invoke-Promotion | Out-Null
        Set-Content -LiteralPath $lastCheck -Value ([DateTimeOffset]::Now.ToString('O'))
    }
}

switch ($Action) {
    'Status' {
        [pscustomobject]@{
            Armed = Test-Path -LiteralPath $armMarker
            CanaryPassed = Test-Path -LiteralPath $canaryPass
            ActiveConfig = Test-Path -LiteralPath $activeConfig
            Promotion = if (Test-Path -LiteralPath $manifest) {
                Get-Content -LiteralPath $manifest -Raw | ConvertFrom-Json
            } else { $null }
            Container = docker inspect cashcat-live --format '{{json .State}}' 2>$null
        } | ConvertTo-Json -Depth 8
    }
    'Promote' { Invoke-Promotion | Out-Null }
    'Flatten' {
        docker compose -f $composeFile stop cashcat-live | Out-Null
        Invoke-LiveFlatten
    }
    'Canary' {
        if (-not (Test-Path -LiteralPath $activeConfig)) {
            throw 'Promote a corrected 12-hour winner before running the canary.'
        }
        $stamp = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()
        $reportName = "canary-$stamp.json"
        docker compose -f $composeFile run --rm --no-deps cashcat-live `
            --config /opt/mm/run/cashcat-active-live.toml live `
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
            -not [bool]$report.scientifically_valid) {
            throw 'canary evidence did not satisfy duration/fill/validity/flatness gates'
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
