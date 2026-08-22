param(
    [Parameter(Mandatory = $true)]
    [string]$Baseline,
    [int]$Cpu = -1
)

$ErrorActionPreference = "Stop"
$temporary = [System.IO.Path]::GetTempFileName()
try {
    $env:MM_BENCH_OUTPUT = $temporary
    if ($Cpu -ge 0) {
        $env:MM_BENCH_CPU = "$Cpu"
    } else {
        Remove-Item Env:MM_BENCH_CPU -ErrorAction SilentlyContinue
    }
    cargo run --locked --release --bin hot-path-bench | Out-Host
    if ($LASTEXITCODE -ne 0) {
        throw "hot-path benchmark failed"
    }
    $reference = Get-Content -Raw -LiteralPath $Baseline | ConvertFrom-Json
    $candidate = Get-Content -Raw -LiteralPath $temporary | ConvertFrom-Json

    $p50Regression = $candidate.quote_batch_ns_per_decision.p50 / $reference.quote_batch_ns_per_decision.p50 - 1.0
    $p99Regression = $candidate.quote_batch_ns_per_decision.p99 / $reference.quote_batch_ns_per_decision.p99 - 1.0
    if ($p50Regression -gt 0.05) {
        throw ("quote p50 regressed by {0:P2}; limit is 5%" -f $p50Regression)
    }
    if ($p99Regression -gt 0.10) {
        throw ("quote p99 regressed by {0:P2}; limit is 10%" -f $p99Regression)
    }
    if ($candidate.monitoring_overhead_percent -gt 5.0) {
        throw ("latency monitoring overhead is {0:N2}%; limit is 5%" -f $candidate.monitoring_overhead_percent)
    }
    Write-Host ("Performance gates passed: p50={0:P2}, p99={1:P2}, monitoring={2:N2}%" -f $p50Regression, $p99Regression, $candidate.monitoring_overhead_percent)
} finally {
    Remove-Item Env:MM_BENCH_OUTPUT -ErrorAction SilentlyContinue
    Remove-Item Env:MM_BENCH_CPU -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $temporary -Force -ErrorAction SilentlyContinue
}
