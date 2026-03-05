@echo off
setlocal
set PYTHONUTF8=1
cd /d "%~dp0\.."
set "OCC_PYTHON=%CD%\.occ_env\python.exe"

REM OCC가 설치된 .occ_env가 있으면 그 Python으로 실행 (DLL/경로 문제 방지)
if exist "%OCC_PYTHON%" (
    echo [OCC] Using .occ_env Python: %OCC_PYTHON%
    REM CONDA_PREFIX 설정으로 train_pipeline이 .occ_env를 OCC 후보로 사용
    set "CONDA_PREFIX=%CD%\.occ_env"
    REM OpenCASCADE DLL 검색 경로 추가 (DLL load failed while importing _gp 방지)
    set "PATH=%CD%\.occ_env\Library\bin;%PATH%"
    REM 프로젝트 의존성 확인 (yaml, loguru 등); 없으면 requirements.txt 설치
    "%OCC_PYTHON%" -c "import yaml, loguru" 2>nul
    if errorlevel 1 (
        echo [OCC] Installing project requirements into .occ_env...
        "%OCC_PYTHON%" -m pip install -r requirements.txt -q --no-warn-script-location
    )
    "%OCC_PYTHON%" train_pipeline.py %*
    exit /b %ERRORLEVEL%
)

REM 그 외에는 현재 python 사용 (conda base 등 OCC 경로는 train_pipeline이 자동 탐색)
python train_pipeline.py %*
exit /b %ERRORLEVEL%
