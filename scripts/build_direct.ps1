# Build all targets by invoking nvcc directly.
#
# Use this when the Ninja/CMake path is unavailable - for example inside a
# confined agent sandbox where Ninja cannot capture child-process output and
# deadlocks. In a normal terminal prefer scripts\build.ps1 or VS2026.

$ErrorActionPreference = "Stop"

$vcvars = "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
$root = Split-Path -Parent $PSScriptRoot
$bin = Join-Path $root "bin"
New-Item -ItemType Directory -Path $bin -Force | Out-Null

# -Xcompiler=/utf-8 uses the = form on purpose: embedding a quoted argument
# inside a PowerShell string here just fights the quote parser.
$flags = "-arch=sm_75 -O3 --use_fast_math -lineinfo --expt-relaxed-constexpr -Xcompiler=/utf-8"

# Every Src / Inc entry is relative to $root, so no `cd` is needed.
$targets = @(
    @{ Name = "bench_matmul"
       Src = @("src\matmul\bench_matmul.cu", "src\matmul\matmul.cu", "src\common\reference.cu")
       Inc = @("src\common"); Libs = @("-lcublas") },
    @{ Name = "bench_conv"
       Src = @("src\conv2d\bench_conv.cu", "src\conv2d\conv.cu",
               "src\matmul\matmul.cu", "src\common\reference.cu")
       Inc = @("src\common", "src\matmul"); Libs = @() },
    @{ Name = "debug_mm"
       Src = @("src\matmul\debug_mm.cu", "src\matmul\matmul.cu", "src\common\reference.cu")
       Inc = @("src\common"); Libs = @() },
    @{ Name = "debug_conv"
       Src = @("src\conv2d\debug_conv.cu", "src\conv2d\conv.cu",
               "src\matmul\matmul.cu", "src\common\reference.cu")
       Inc = @("src\common", "src\matmul"); Libs = @() }
)

foreach ($t in $targets) {
    $srcArgs = ($t.Src | ForEach-Object { "`"$root\$_`"" }) -join " "
    $incArgs = ($t.Inc | ForEach-Object { "-I`"$root\$_`"" }) -join " "
    $libArgs = ($t.Libs -join " ")
    $out = "`"$bin\$($t.Name).exe`""

    $cmd = "`"$vcvars`" >nul 2>&1 && nvcc $flags $incArgs $srcArgs -o $out $libArgs"
    Write-Host "Building $($t.Name) ..."
    cmd /c $cmd
    if ($LASTEXITCODE -ne 0) { Write-Error "Failed to build $($t.Name)"; exit $LASTEXITCODE }
}

Write-Host ""
Write-Host "Build OK:"
Get-ChildItem "$bin\*.exe" | ForEach-Object { Write-Host "  $($_.FullName)" }
