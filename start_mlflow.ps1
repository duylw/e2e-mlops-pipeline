# Get the directory of this script to build absolute paths dynamically
$ProjectDir = $PSScriptRoot
if (-not $ProjectDir) {
    $ProjectDir = (Get-Item .).FullName
}

# Define database and artifact paths relative to the project directory
$DbPath = "$ProjectDir/tmp/mlflow.db"
$ProjectDirUri = $ProjectDir -replace '\\', '/'
$ArtifactPath = "file:///$ProjectDirUri/tmp/mlflow-artifacts"

# Ensure the tmp directory exists
if (-not (Test-Path "$ProjectDir/tmp")) {
    New-Item -ItemType Directory -Force -Path "$ProjectDir/tmp" | Out-Null
}

Write-Host "Starting MLflow server..." -ForegroundColor Green
Write-Host "Backend Store: sqlite:///$DbPath" -ForegroundColor Cyan
Write-Host "Artifact Root: $ArtifactPath" -ForegroundColor Cyan

# Run MLflow server
& "$ProjectDir/.venv/Scripts/mlflow" server --port 5000 --backend-store-uri "sqlite:///$DbPath" --default-artifact-root "$ArtifactPath"
