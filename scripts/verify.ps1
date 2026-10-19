<#
.SYNOPSIS
    Build everything, run every benchmark, and check correctness + performance.

.DESCRIPTION
    This is the regression gate. It answers two questions:

      1. Is every kernel still CORRECT?  (relL2 vs the CPU reference)
      2. Did any kernel get SLOWER?      (GFLOP/s vs the recorded baseline)

    Correctness failures are hard errors (exit 1). Performance regressions
    are warnings, because GFLOP/s varies run to run - only you can decide
    whether a 10% dip is noise or a real regression.

    Baselines live in the $baselines table below. UPDATE THEM whenever you
    legitimately make something faster, otherwise the gate only ever warns.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\verify.ps1

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\verify.ps1 -SkipBuild
#>
[CmdletBinding()]
param(
    # Skip the nvcc build step (use when you already built).
    [switch]$SkipBuild,

    # Matrix size to test. Default is the shape recorded in the baselines.
    [int[]]$MatmulSize = @(1024, 1024, 1024),

    # Warn when a kernel drops below this fraction of its baseline GFLOP/s.
    [double]$PerfTolerance = 0.80
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$bin = Join-Path $root "bin"

# ---------------------------------------------------------------------------
# Baselines. Keys must match the implementation names printed by the benches.
#   GFlops   : reference GFLOP/s on this machine
#   MaxRelL2 : correctness ceiling. FP16 kernels get a looser one on purpose.
# ---------------------------------------------------------------------------
$baselines = [ordered]@{
    # ---- matmul, 1024^3 FP32, RTX 2080 Super ----
    "v0 naive"          = @{ GFlops =  770; MaxRelL2 = 1e-5 }
    "v1 tiled 16x16"    = @{ GFlops = 1233; MaxRelL2 = 1e-5 }
    "v1 tiled 32x32"    = @{ GFlops = 1303; MaxRelL2 = 1e-5 }
    "v2 reg-tile 64x64" = @{ GFlops = 3215; MaxRelL2 = 1e-5 }
    "v3 wmma fp16"      = @{ GFlops = 3135; MaxRelL2 = 1e-3 }   # FP16 inputs
    "cuBLAS (library)"  = @{ GFlops = 9090; MaxRelL2 = 1e-5 }

    # ---- conv, C=64 224x224 OC=64 3x3 ----
    "v0 naive direct"        = @{ GFlops =  435; MaxRelL2 = 1e-5 }
    "v1 shared-memory"       = @{ GFlops =  440; MaxRelL2 = 1e-5 }
    "v2 im2col + GEMM"       = @{ GFlops = 2461; MaxRelL2 = 1e-5 }
    "v3 depthwise-separable" = @{ GFlops = 11572; MaxRelL2 = 1e-5 }
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
function Write-Step([string]$text) {
    Write-Host ""
    Write-Host "=== $text ===" -ForegroundColor Cyan
}

# Parse one benchmark row into a structured object.
# Matches e.g. "v2 reg-tile 64x64   0.6680   3215.0   4.2x   6.34e-08"
# and tolerates a trailing "(equal-FLOP baseline 1.08 ms)" note.
$rowPattern = '^(?<name>.+?)\s{2,}(?<ms>\d+\.\d+)\s+(?<gflops>\d+(?:\.\d+)?)\s+(?<speedup>\d+(?:\.\d+)?)x\s+(?<rell2>\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)'

function Parse-BenchOutput([string[]]$lines) {
    $rows = @()
    foreach ($line in $lines) {
        if ($line -match $rowPattern) {
            $rows += [pscustomobject]@{
                Name    = $matches["name"].Trim()
                Ms      = [double]$matches["ms"]
                GFlops  = [double]$matches["gflops"]
                RelL2   = [double]$matches["rell2"]
            }
        }
    }
    return $rows
}

function Invoke-Bench([string]$exeName, [string[]]$exeArgs) {
    $exe = Join-Path $bin $exeName
    if (-not (Test-Path -LiteralPath $exe)) {
        Write-Error "$exeName not found at $exe. Run scripts\build_direct.ps1 first (or drop -SkipBuild)."
    }
    Write-Host "Running $exeName $($exeArgs -join ' ')"
    $out = & $exe @exeArgs 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Error "$exeName exited with code $LASTEXITCODE"
    }
    return ,$out
}

# ---------------------------------------------------------------------------
# 1. Build
# ---------------------------------------------------------------------------
if ($SkipBuild) {
    Write-Step "Skipping build (-SkipBuild)"
} else {
    Write-Step "Building all targets"
    & powershell -ExecutionPolicy Bypass -File (Join-Path $root "scripts\build_direct.ps1")
    if ($LASTEXITCODE -ne 0) { Write-Error "build failed with code $LASTEXITCODE"; exit 1 }
}

# ---------------------------------------------------------------------------
# 2. Run benchmarks
# ---------------------------------------------------------------------------
$allRows = @()

Write-Step "Benchmark: matmul $($MatmulSize -join 'x')"
$mmOut = Invoke-Bench "bench_matmul.exe" @($MatmulSize | ForEach-Object { "$_" })
$mmOut | ForEach-Object { Write-Host $_ }
$allRows += Parse-BenchOutput $mmOut

Write-Step "Benchmark: conv (default shape)"
$cvOut = Invoke-Bench "bench_conv.exe" @()
$cvOut | ForEach-Object { Write-Host $_ }
$allRows += Parse-BenchOutput $cvOut

# ---------------------------------------------------------------------------
# 3. Check
# ---------------------------------------------------------------------------
Write-Step "Verification"

$correctnessFailures = @()
$perfWarnings        = @()
$unknownRows         = @()
$checked             = 0

foreach ($row in $allRows) {
    $base = $baselines[$row.Name]
    if ($null -eq $base) {
        $unknownRows += $row.Name
        continue
    }
    $checked++

    # --- correctness: hard failure ---
    if ($row.RelL2 -gt $base.MaxRelL2) {
        $correctnessFailures += [pscustomobject]@{
            Name  = $row.Name
            Got   = $row.RelL2
            Limit = $base.MaxRelL2
        }
        continue   # no point also warning about perf on a wrong kernel
    }

    # --- performance: warning only ---
    $floor = $base.GFlops * $PerfTolerance
    if ($row.GFlops -lt $floor) {
        $perfWarnings += [pscustomobject]@{
            Name     = $row.Name
            Got      = $row.GFlops
            Expected = $base.GFlops
            Pct      = [math]::Round(100.0 * $row.GFlops / $base.GFlops, 1)
        }
    }
}

# --- report table ---
$fmt = "{0,-26} {1,10} {2,10} {3,9} {4,12}  {5}"
Write-Host ""
Write-Host ($fmt -f "implementation", "GFLOP/s", "baseline", "%", "relL2", "verdict")
Write-Host ("-" * 84)
foreach ($row in $allRows) {
    $base = $baselines[$row.Name]
    if ($null -eq $base) {
        Write-Host ($fmt -f $row.Name, $row.GFlops, "-", "-", $row.RelL2, "not tracked")
        continue
    }
    $pct = [math]::Round(100.0 * $row.GFlops / $base.GFlops, 1)

    $verdict = "OK"
    $color   = "Green"
    if ($row.RelL2 -gt $base.MaxRelL2) {
        $verdict = "CORRECTNESS FAIL"
        $color   = "Red"
    } elseif ($row.GFlops -lt $base.GFlops * $PerfTolerance) {
        $verdict = "SLOW"
        $color   = "Yellow"
    }
    Write-Host ($fmt -f $row.Name, $row.GFlops, $base.GFlops, "$pct%", $row.RelL2, $verdict) `
        -ForegroundColor $color
}

Write-Host ""
Write-Host "Checked $checked of $($allRows.Count) benchmark rows."

if ($unknownRows.Count -gt 0) {
    Write-Host ""
    Write-Host "NOTE: $($unknownRows.Count) row(s) not present in the baseline table:" -ForegroundColor DarkGray
    $unknownRows | ForEach-Object { Write-Host "  - $_" -ForegroundColor DarkGray }
    Write-Host "  Add them to `$baselines in scripts\verify.ps1 if they are new implementations." -ForegroundColor DarkGray
}

# ---------------------------------------------------------------------------
# 4. Verdict
# ---------------------------------------------------------------------------
Write-Step "Result"

$exitCode = 0

if ($correctnessFailures.Count -gt 0) {
    Write-Host "CORRECTNESS FAILURES ($($correctnessFailures.Count)):" -ForegroundColor Red
    foreach ($f in $correctnessFailures) {
        Write-Host ("  {0,-26} relL2 = {1:G3}  (ceiling {2:G3})" -f $f.Name, $f.Got, $f.Limit) -ForegroundColor Red
    }
    Write-Host ""
    Write-Host "A wrong kernel is never acceptable. Fix it before touching performance." -ForegroundColor Red
    $exitCode = 1
}

if ($perfWarnings.Count -gt 0) {
    Write-Host "PERFORMANCE REGRESSIONS ($($perfWarnings.Count)):" -ForegroundColor Yellow
    foreach ($w in $perfWarnings) {
        Write-Host ("  {0,-26} {1,8} GFLOP/s  (baseline {2}, {3}%)" -f $w.Name, $w.Got, $w.Expected, $w.Pct) -ForegroundColor Yellow
    }
    Write-Host ""
    Write-Host "Warnings only. GPU clocks and host load move these numbers, so re-run once before believing it." -ForegroundColor Yellow
}

if ($exitCode -eq 0) {
    Write-Host "PASS: all $checked tracked kernels are correct and within tolerance." -ForegroundColor Green
    if ($perfWarnings.Count -gt 0) {
        Write-Host "(with $($perfWarnings.Count) performance warning(s) above)" -ForegroundColor Yellow
    }
}

exit $exitCode
