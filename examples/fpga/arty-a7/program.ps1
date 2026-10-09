param(
  [string]$VivadoBin = $env:VIVADO_BIN,
  [string]$Bitstream = ""
)
$ErrorActionPreference = "Stop"
if (!$VivadoBin) { $VivadoBin = Split-Path (Get-Command vivado.bat -ErrorAction Stop).Source }
if (!$Bitstream) { $Bitstream = Join-Path $PSScriptRoot "build\arty\pebble_a7.bit" }
if (!(Test-Path -LiteralPath $Bitstream -PathType Leaf)) { throw "Bitstream not found: $Bitstream" }
Push-Location $PSScriptRoot
try {
  & (Join-Path $VivadoBin "vivado.bat") -mode batch -source board/program_arty.tcl -tclargs $Bitstream
  if ($LASTEXITCODE -ne 0) { throw "FPGA programming failed; inspect Vivado log" }
} finally { Pop-Location }
