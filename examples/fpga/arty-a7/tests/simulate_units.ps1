param([string]$VivadoBin = $env:VIVADO_BIN)
$ErrorActionPreference = "Stop"
if (!$VivadoBin) { $VivadoBin = Split-Path (Get-Command xvlog.bat -ErrorAction Stop).Source }
$root = Split-Path $PSScriptRoot
$out = Join-Path $root "build\unit-xsim"
New-Item -ItemType Directory -Path $out -Force | Out-Null
Copy-Item (Join-Path $root "board\*.sv") -Destination $out
Copy-Item (Join-Path $PSScriptRoot "tb_*.sv") -Destination $out
Push-Location $out
try {
  & (Join-Path $VivadoBin "xvlog.bat") -sv axi_bram.sv axi_uart.sv uart_rx.sv uart_loader.sv tb_axi_bram.sv tb_axi_uart.sv tb_uart_rx.sv tb_uart_loader.sv
  if ($LASTEXITCODE -ne 0) { throw "Unit RTL compilation failed" }
  foreach ($test in @("tb_axi_bram", "tb_axi_uart", "tb_uart_rx", "tb_uart_loader")) {
    & (Join-Path $VivadoBin "xelab.bat") $test -s $test -timescale 1ns/1ps
    if ($LASTEXITCODE -ne 0) { throw "Unit elaboration failed: $test" }
    & (Join-Path $VivadoBin "xsim.bat") $test -runall -log "$test.log"
    if ($LASTEXITCODE -ne 0) { throw "Unit simulation failed: $test" }
    $log = Get-Content -LiteralPath "$test.log" -Raw
    if ($log -notmatch '(?m)^PASS:' -or $log -match '(?m)^(Fatal|Error):') { throw "Unit test failed: $test" }
  }
} finally { Pop-Location }
