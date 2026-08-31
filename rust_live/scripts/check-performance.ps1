param(
    [Parameter(Mandatory = $true)]
    [string]$Baseline,
    [int]$Cpu = 4,
    [int]$Runs = 7
)

$ErrorActionPreference = "Stop"
$temporary = [System.IO.Path]::GetTempFileName()
try {
    & "$PSScriptRoot\run-performance-study.ps1" -Output $temporary -Cpu $Cpu -Runs $Runs
    if ($LASTEXITCODE -ne 0) { throw "performance study failed" }
    $reference = Get-Content -Raw -LiteralPath $Baseline | ConvertFrom-Json
    $candidate = Get-Content -Raw -LiteralPath $temporary | ConvertFrom-Json
    foreach ($field in @("study_schema_version", "benchmark_schema_version", "pinned_cpu", "cpu_model")) {
        if ($candidate.$field -ne $reference.$field) { throw "incompatible baseline: $field" }
    }
    foreach ($field in @("profile", "opt_level", "target", "target_features", "rustc")) {
        if ($candidate.build.$field -ne $reference.build.$field) { throw "incompatible baseline build: $field" }
    }
    foreach ($name in @("policy_kernel_p50_ns", "hot_step_p50_ns")) {
        $regression = $candidate.metrics.$name / $reference.metrics.$name - 1.0
        if ($regression -gt 0.05) { throw "$name regressed by $($regression.ToString('P2')); limit is 5%" }
    }
    foreach ($name in @("policy_kernel_p99_ns", "hot_step_p99_ns")) {
        $regression = $candidate.metrics.$name / $reference.metrics.$name - 1.0
        if ($regression -gt 0.10) { throw "$name regressed by $($regression.ToString('P2')); limit is 10%" }
    }
    $overheadIncrease = $candidate.metrics.monitoring_overhead_percent - $reference.metrics.monitoring_overhead_percent
    if ($overheadIncrease -gt 5.0) { throw "monitoring overhead increased by $overheadIncrease percentage points; limit is 5" }
    Write-Host "Performance gates passed against $Baseline"
} finally {
    Remove-Item -LiteralPath $temporary -Force -ErrorAction SilentlyContinue
}
