@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

for %%I in ("%SCRIPT_DIR%\..") do set "ROOT=%%~fI"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"

set "BUILD_OPENVINO=0"
set "BUILD_GENAI=0"
set "INVALID_ARG="

if "%~1"=="" (
    set "BUILD_OPENVINO=1"
    set "BUILD_GENAI=1"
) else (
    for %%A in (%*) do (
        call :parse_arg "%%~A"
    )
)

if defined INVALID_ARG (
    echo [ERROR] Invalid build target: %INVALID_ARG%
    call :usage
    exit /b 1
)

if "%BUILD_OPENVINO%%BUILD_GENAI%"=="00" (
    echo [ERROR] No valid build target selected.
    call :usage
    exit /b 1
)

call :ensure_vs_env
if errorlevel 1 exit /b 1

if "%BUILD_OPENVINO%"=="1" (
    echo [BUILD] openvino
    cmake -S "%ROOT%\openvino" -B "%ROOT%\openvino\build" -DCMAKE_BUILD_TYPE=Release -DENABLE_PYTHON=ON
    if errorlevel 1 (
        echo [ERROR] CMake configure failed for openvino.
        exit /b 1
    )

    cmake --build "%ROOT%\openvino\build"  --config Release
    if errorlevel 1 (
        echo [ERROR] Build failed for openvino.
        exit /b 1
    )
)

if "%BUILD_GENAI%"=="1" (
    echo [BUILD] openvino.genai
    cmake -S "%ROOT%\openvino.genai" -B "%ROOT%\openvino.genai\build" -DCMAKE_BUILD_TYPE=Release -DOpenVINO_DIR="%ROOT%\openvino\build" -DENABLE_PYTHON=ON
    if errorlevel 1 (
        echo [ERROR] CMake configure failed for openvino.genai.
        echo         Make sure openvino build directory exists and is valid.
        exit /b 1
    )

    cmake --build "%ROOT%\openvino.genai\build" --config Release
    if errorlevel 1 (
        echo [ERROR] Build failed for openvino.genai.
        exit /b 1
    )
)

echo [OK] Build finished.
exit /b 0

:parse_arg
set "ARG=%~1"
if "%ARG%"=="" exit /b 0

for %%T in (%ARG:,= %) do (
    if "%%~T"=="0" (
        set "BUILD_OPENVINO=1"
    ) else if "%%~T"=="1" (
        set "BUILD_GENAI=1"
    ) else (
        set "INVALID_ARG=%%~T"
    )
)
exit /b 0

:ensure_vs_env
where cl >nul 2>nul
if not errorlevel 1 (
    where ninja >nul 2>nul
    if not errorlevel 1 (
        where cmake >nul 2>nul
        if not errorlevel 1 (
            echo [INFO] VC toolchain, cmake, and ninja found in PATH.
            exit /b 0
        )
    )
)

set "VSDEV_CMD="
if defined VSINSTALLDIR (
    if exist "%VSINSTALLDIR%\Common7\Tools\VsDevCmd.bat" (
        set "VSDEV_CMD=%VSINSTALLDIR%\Common7\Tools\VsDevCmd.bat"
    )
)

if not defined VSDEV_CMD (
    set "VSWHERE="
    for /f "usebackq delims=" %%I in (`where vswhere 2^>nul`) do (
        if not defined VSWHERE set "VSWHERE=%%I"
    )

    if not defined VSWHERE (
        if not "%ProgramFiles(x86)%"=="" if exist "%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" (
            set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
        )
    )
    if not defined VSWHERE (
        if not "%ProgramFiles%"=="" if exist "%ProgramFiles%\Microsoft Visual Studio\Installer\vswhere.exe" (
            set "VSWHERE=%ProgramFiles%\Microsoft Visual Studio\Installer\vswhere.exe"
        )
    )
    if not defined VSWHERE (
        if exist "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe" (
            set "VSWHERE=C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
        )
    )
    if not defined VSWHERE (
        if exist "C:\Program Files\Microsoft Visual Studio\Installer\vswhere.exe" (
            set "VSWHERE=C:\Program Files\Microsoft Visual Studio\Installer\vswhere.exe"
        )
    )
    if not defined VSWHERE (
        echo [ERROR] Cannot find vswhere.exe.
        echo         Please install Visual Studio 2022 with C++ build tools.
        exit /b 1
    )

    set "VS_INSTALL="
    for /f "delims=" %%I in ('"!VSWHERE!" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2^>nul') do (
        set "VS_INSTALL=%%I"
    )

    if not defined VS_INSTALL (
        echo [ERROR] Visual Studio 2022 with VC tools not found.
        exit /b 1
    )

    set "VSDEV_CMD=!VS_INSTALL!\Common7\Tools\VsDevCmd.bat"
)

if not exist "%VSDEV_CMD%" (
    echo [ERROR] VsDevCmd.bat not found: %VSDEV_CMD%
    exit /b 1
)

echo [INFO] Initializing VS build environment...
call "%VSDEV_CMD%" -arch=x64 -host_arch=x64 >nul
if errorlevel 1 (
    echo [ERROR] Failed to initialize VS developer environment.
    exit /b 1
)

where cl >nul 2>nul
if errorlevel 1 (
    echo [ERROR] cl.exe still not found after VS environment setup.
    exit /b 1
)

where cmake >nul 2>nul
if errorlevel 1 (
    echo [ERROR] cmake.exe not found after VS environment setup.
    exit /b 1
)

where ninja >nul 2>nul
if errorlevel 1 (
    echo [ERROR] ninja.exe not found after VS environment setup.
    echo         Please install Ninja or CMake Ninja support in VS.
    exit /b 1
)

exit /b 0

:usage
echo Usage:
echo   build.bat            ^(build openvino + openvino.genai^)
echo   build.bat 0          ^(build openvino only^)
echo   build.bat 1          ^(build openvino.genai only^)
echo   build.bat 0,1        ^(build both^)
echo   build.bat 0 1        ^(build both^)
exit /b 0
