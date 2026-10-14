# Train one demo shape, then copy the policy for the real robot into deploy\<shape>\.
# Usage: .\train_shape.ps1 -Shape eight  -Iterations 6000
#        .\train_shape.ps1 -Shape circle -Iterations 6000
# Each call continues from the newest checkpoint unless you pick a starting run:
#        .\train_shape.ps1 -Shape circle -Iterations 5000 -BaseRun 2026-10-11_22-38-05
param(
  [Parameter(Mandatory)][ValidateSet("eight", "circle", "face")][string]$Shape,
  [int]$Iterations = 6000,
  [int]$NumEnvs = 64,
  [string]$BaseRun = ""
)

Set-Location $PSScriptRoot
$env:PYTHONIOENCODING = "utf-8"
$gcodeFiles = @{ circle = "examples\circle.gcode"; face = "examples\face_outline.gcode" }
if ($gcodeFiles.ContainsKey($Shape)) {
  $env:MJLAB_DRAWING_GCODE = (Resolve-Path $gcodeFiles[$Shape]).Path
} else {
  Remove-Item Env:MJLAB_DRAWING_GCODE -ErrorAction SilentlyContinue
}
# The circle is kept small so the whole thing is within reach of the right arm.
if ($Shape -eq "circle") { $env:MJLAB_DRAWING_GCODE_SIZE = "0.24" } else { Remove-Item Env:MJLAB_DRAWING_GCODE_SIZE -ErrorAction SilentlyContinue }

if ($BaseRun) {
  $base = Get-Item "logs\rsl_rl\g1_drawing\$BaseRun" -ErrorAction Stop
} else {
  $base = Get-ChildItem logs\rsl_rl\g1_drawing -Directory |
    Where-Object { (Get-ChildItem $_.FullName -Filter model_*.pt).Count -gt 0 } |
    Sort-Object LastWriteTime | Select-Object -Last 1
}
if (-not $base) { throw "No run with checkpoints found." }
Write-Host "Training '$Shape' from $($base.Name) for $Iterations iterations"

.\.venv\Scripts\python.exe -m mjlab.scripts.train Mjlab-Drawing-Flat-Unitree-G1-PF `
  --env.scene.num-envs $NumEnvs `
  --agent.resume True `
  --agent.load-run $base.Name `
  --agent.max-iterations $Iterations

.\export_policy.ps1 -Shape $Shape
