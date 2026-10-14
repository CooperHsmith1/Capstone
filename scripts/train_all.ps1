# Train every demo shape back to back and export each policy to deploy\<shape>\.
# Usage: .\train_all.ps1                      (eight, circle, face; 8000 iterations each)
#        .\train_all.ps1 -Iterations 12000
#        .\train_all.ps1 -Shapes circle,face   (only some)
param(
  [int]$Iterations = 8000,
  [int]$NumEnvs = 1024,
  [string[]]$Shapes = @("eight", "circle", "face")
)

Set-Location $PSScriptRoot
foreach ($s in $Shapes) {
  .\train_shape.ps1 -Shape $s -Iterations $Iterations -NumEnvs $NumEnvs
}
