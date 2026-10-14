# Copy the newest checkpoint (.pt + ONNX) into deploy\<shape>\ for the real robot.
# Usage: .\export_policy.ps1 -Shape circle
param([Parameter(Mandatory)][ValidateSet("eight", "circle", "face")][string]$Shape)

Set-Location $PSScriptRoot
$run = Get-ChildItem logs\rsl_rl\g1_drawing -Directory |
  Where-Object { (Get-ChildItem $_.FullName -Filter model_*.pt).Count -gt 0 } |
  Sort-Object LastWriteTime | Select-Object -Last 1
if (-not $run) { throw "No run with checkpoints found." }

$pt = Get-ChildItem $run.FullName -Filter model_*.pt |
  Sort-Object { [int]($_.BaseName -replace 'model_', '') } | Select-Object -Last 1
$out = "deploy\$Shape"
New-Item -ItemType Directory -Force $out | Out-Null
Copy-Item $pt.FullName $out -Force

$onnx = Get-ChildItem $run.FullName -Filter *.onnx | Sort-Object LastWriteTime | Select-Object -Last 1
if ($onnx) {
  Copy-Item $onnx.FullName "$out\policy.onnx" -Force
} else {
  Write-Warning "No ONNX in the run folder; exporting from $($pt.Name) is not available here."
}
if ($Shape -eq "circle") { Copy-Item examples\circle.gcode $out -Force }
if ($Shape -eq "face") { Copy-Item examples\face_outline.gcode $out -Force }

# Keep the whole run (TensorBoard events, params, newest checkpoints) with the shape.
$data = "$out\training_data"
New-Item -ItemType Directory -Force $data | Out-Null
Copy-Item "$($run.FullName)\events.out.tfevents.*" $data -Force
Copy-Item "$($run.FullName)\params" $data -Recurse -Force
Write-Host "Training data -> $data (view: tensorboard --logdir $data)"
Write-Host "Exported $($pt.Name) from $($run.Name) -> $out"
