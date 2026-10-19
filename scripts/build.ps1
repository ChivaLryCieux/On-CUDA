# Configure + build both benchmarks.
#
# Run from a normal PowerShell:
#   powershell -ExecutionPolicy Bypass -File scripts\build.ps1
#
# Or open the folder in VS2026 (File > Open > CMake...) which configures the
# same CMakeLists.txt and gives you kernel debugging with device breakpoints.
#
# Why the PATH juggling: nvcc does NOT locate cl.exe on its own (unlike what
# you might expect), so the MSVC environment must be loaded first. The bundled
# Ninja is also not on PATH by default.

$ErrorActionPreference = "Stop"

$vsRoot = "C:\Program Files\Microsoft Visual Studio\18\Community"
$vcvars = Join-Path $vsRoot "VC\Auxiliary\Build\vcvars64.bat"
$cmake  = Join-Path $vsRoot "Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
$ninja  = Join-Path $vsRoot "Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja"

foreach ($p in @($vcvars, $cmake, $ninja)) {
    if (-not (Test-Path $p)) { Write-Error "Not found: $p"; exit 1 }
}

$buildDir = Join-Path $PSScriptRoot "..\build"
$srcDir   = Join-Path $PSScriptRoot ".."

# Put Ninja on PATH from PowerShell and let cmd inherit it. Do NOT try to
# `set "PATH=..."` inside the cmd /c chain: cmd's outer-quote stripping mangles
# that form and silently destroys PATH.
$env:PATH = "$ninja;$env:PATH"

# Load vcvars64, then configure and build inside that shell.
$cmd = "`"$vcvars`" >nul 2>&1 && " +
       "`"$cmake`" -S `"$srcDir`" -B `"$buildDir`" -G Ninja -DCMAKE_BUILD_TYPE=Release && " +
       "`"$cmake`" --build `"$buildDir`" --parallel"

Write-Host "Configuring and building..."
cmd /c $cmd
if ($LASTEXITCODE -ne 0) {
    Write-Error "Build failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "Build OK. Binaries:"
Get-ChildItem (Join-Path $buildDir "src\*\*.exe") | ForEach-Object { Write-Host "  $($_.FullName)" }
