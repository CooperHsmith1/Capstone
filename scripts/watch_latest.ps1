# Watch a policy in the browser (Viser).
# Usage: .\watch_latest.ps1                    -> newest checkpoint, figure eight
#        .\watch_latest.ps1 -Shape eight|circle|face   -> that shape's exported policy
#        .\watch_latest.ps1 -Gcode examples\face_outline.gcode   -> newest checkpoint + G-code
param([string]$Shape = "", [string]$Gcode = "", [int]$NumEnvs = 1)

Set-Location $PSScriptRoot
$env:PYTHONIOENCODING = "utf-8"

$gcodeFiles = @{ circle = "examples\circle.gcode"; face = "examples\face_outline.gcode" }
if ($Shape -and $gcodeFiles.ContainsKey($Shape)) {
  $Gcode = $gcodeFiles[$Shape]
} elseif ($Shape -eq "eight") {
  Remove-Item Env:MJLAB_DRAWING_GCODE -ErrorAction SilentlyContinue
}
if ($Gcode) {
  $env:MJLAB_DRAWING_GCODE = (Resolve-Path $Gcode).Path
} else {
  Remove-Item Env:MJLAB_DRAWING_GCODE -ErrorAction SilentlyContinue
}
if ($Gcode -match "circle") { $env:MJLAB_DRAWING_GCODE_SIZE = "0.24" } else { Remove-Item Env:MJLAB_DRAWING_GCODE_SIZE -ErrorAction SilentlyContinue }

if ($Shape) {
  $latest = Get-ChildItem "deploy\$Shape" -Filter model_*.pt -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
  if (-not $latest) { Write-Error "No exported policy in deploy\$Shape yet. Run .\export_policy.ps1 -Shape $Shape"; exit 1 }
} else {
  $latest = Get-ChildItem logs\rsl_rl\g1_drawing -Recurse -Filter model_*.pt |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
}
if (-not $latest) { Write-Error "No checkpoints found."; exit 1 }
Write-Host "Playing $($latest.FullName)"
Write-Host "Open the URL printed below (usually http://localhost:8080)"

.\.venv\Scripts\python.exe -m mjlab.scripts.play Mjlab-Drawing-Flat-Unitree-G1-PF `
  --checkpoint-file $latest.FullName --num-envs $NumEnvs --viewer viser
