@echo off
setlocal
:: Get the directory of this batch file
set "PROJECT_DIR=%~dp0"
:: Remove trailing backslash
set "PROJECT_DIR=%PROJECT_DIR:~0,-1%"

set "DB_PATH=%PROJECT_DIR%\tmp\mlflow.db"
:: Replace backslashes with forward slashes for SQLite URI
set "DB_URI_PATH=%DB_PATH:\=/%"
set "ARTIFACT_PATH=file:///%PROJECT_DIR:\=/%/tmp/mlflow-artifacts"

if not exist "%PROJECT_DIR%\tmp" mkdir "%PROJECT_DIR%\tmp"

echo Starting MLflow server...
echo Backend Store: sqlite:///%DB_URI_PATH%
echo Artifact Root: %ARTIFACT_PATH%

"%PROJECT_DIR%\.venv\Scripts\mlflow" server --port 5000 --backend-store-uri "sqlite:///%DB_URI_PATH%" --default-artifact-root "%ARTIFACT_PATH%"
