# Resume whiteboard drawing training from the newest checkpoint.
# Usage: .\resume_training.ps1 [-Iterations 5000] [-NumEnvs 64] [-Gcode examples\portrait.gcode]
param(
  [int]$Iterations = 5000,
  [int]$NumEnvs = 64,
  [string]$Gcode = ""
)

Set-Location $PSScriptRoot
$env:PYTHONIOENCODING = "utf-8"
if ($Gcode) { $env:MJLAB_DRAWING_GCODE = (Resolve-Path $Gcode).Path }

# Newest run folder that actually contains checkpoints.
$run = Get-ChildItem logs\rsl_rl\g1_drawing -Directory |
  Where-Object { (Get-ChildItem $_.FullName -Filter model_*.pt).Count -gt 0 } |
  Sort-Object LastWriteTime | Select-Object -Last 1
if (-not $run) { throw "No run with checkpoints found." }
Write-Host "Resuming from run $($run.Name)"

.\.venv\Scripts\python.exe -m mjlab.scripts.train Mjlab-Drawing-Flat-Unitree-G1-PF `
  --env.scene.num-envs $NumEnvs `
  --agent.resume True `
  --agent.load-run $run.Name `
  --agent.max-iterations $Iterations

