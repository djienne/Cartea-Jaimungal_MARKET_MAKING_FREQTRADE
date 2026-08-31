param(
    [Parameter(Mandatory = $true)]
    [string]$Output,
    [int]$Cpu = 4,
    [int]$Runs = 7
)

$ErrorActionPreference = "Stop"
if ($Runs -lt 3 -or ($Runs % 2) -eq 0) { throw "Runs must be an odd number >= 3" }
$reports = @()
try {
    $env:MM_BENCH_CPU = "$Cpu"
    for ($index = 0; $index -lt $Runs; $index++) {
        $temporary = [System.IO.Path]::GetTempFileName()
        try {
            $env:MM_BENCH_OUTPUT = $temporary
            cargo run --locked --release --bin hot-path-bench | Out-Host
            if ($LASTEXITCODE -ne 0) { throw "hot-path benchmark failed" }
            $reports += Get-Content -Raw -LiteralPath $temporary | ConvertFrom-Json
        } finally {
            Remove-Item -LiteralPath $temporary -Force -ErrorAction SilentlyContinue
        }
    }
} finally {
    Remove-Item Env:MM_BENCH_OUTPUT -ErrorAction SilentlyContinue
    Remove-Item Env:MM_BENCH_CPU -ErrorAction SilentlyContinue
}

$first = $reports[0]
foreach ($report in $reports) {
    if ($report.schema_version -ne 3 -or $report.pinned_cpu -ne $Cpu) { throw "benchmark schema/core mismatch" }
    foreach ($field in @("profile", "opt_level", "target", "target_features", "rustc")) {
        if ($report.build.$field -ne $first.build.$field) { throw "build changed across runs: $field" }
    }
    if ($report.cpu_model -ne $first.cpu_model) { throw "CPU model changed across runs" }
}

function Median([double[]]$values) {
    $sorted = $values | Sort-Object
    return [double]$sorted[[int]($sorted.Count / 2)]
}

$study = [ordered]@{
    study_schema_version = 1
    benchmark_schema_version = 3
    runs = $Runs
    pinned_cpu = $Cpu
    cpu_model = $first.cpu_model
    build = $first.build
    metrics = [ordered]@{
        policy_kernel_p50_ns = Median @($reports | ForEach-Object { $_.policy_kernel_batch_mean_ns_per_decision.p50 })
        policy_kernel_p99_ns = Median @($reports | ForEach-Object { $_.policy_kernel_batch_mean_ns_per_decision.p99 })
        hot_step_p50_ns = Median @($reports | ForEach-Object { $_.hot_step_batch_mean_ns_per_decision.p50 })
        hot_step_p99_ns = Median @($reports | ForEach-Object { $_.hot_step_batch_mean_ns_per_decision.p99 })
        monitoring_overhead_percent = Median @($reports | ForEach-Object { $_.monitoring_overhead_percent })
        hjb_p50_ms = Median @($reports | ForEach-Object { $_.hjb_solve_ms.p50 })
    }
}
$study | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $Output -Encoding utf8
Write-Host "wrote $Output"
