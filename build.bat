@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

for %%I in ("%SCRIPT_DIR%\..") do set "ROOT=%%~fI"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"

set "OPENVINO_SRC=%ROOT%\openvino"
set "OPENVINO_BUILD=%OPENVINO_SRC%\build"
set "GENAI_SRC=%ROOT%\openvino.genai"
set "GENAI_BUILD=%GENAI_SRC%\build"
set "WHEEL_OUTPUT_DIR=%ROOT%\wheel"
set "WHEEL_VENV_DIR=%SCRIPT_DIR%\.wheel-build-venv"
set "WHEEL_SCRIPT=%SCRIPT_DIR%\scripts\wheel.py"

set "BUILD_OPENVINO=1"
set "BUILD_GENAI=1"
set "BUILD_WHEEL=0"
set "INVALID_ARG="
set "SHOW_USAGE=0"

if not "%~1"=="" (
    call :parse_args %*
    if errorlevel 1 exit /b 1
)

if "%SHOW_USAGE%"=="1" (
    call :usage
    exit /b 0
)

if defined INVALID_ARG (
    echo [ERROR] Invalid argument: %INVALID_ARG%
    call :usage
    exit /b 1
)

call :ensure_vs_env
if errorlevel 1 exit /b 1

if "%BUILD_OPENVINO%"=="1" (
    call :build_openvino
    if errorlevel 1 exit /b 1
)

if "%BUILD_GENAI%"=="1" (
    call :build_genai
    if errorlevel 1 exit /b 1
)

if "%BUILD_WHEEL%"=="1" (
    call :build_wheels
    if errorlevel 1 exit /b 1
)

echo [OK] Build finished.
exit /b 0

:build_openvino
echo [BUILD] openvino
call :configure_openvino
if errorlevel 1 exit /b 1

cmake --build "%OPENVINO_BUILD%" --parallel
if errorlevel 1 (
    echo [ERROR] Build failed for openvino.
    exit /b 1
)
exit /b 0

:configure_openvino
if defined OPENVINO_ALREADY_CONFIGURED exit /b 0

set "OPENVINO_CMAKE_ARGS=-DCMAKE_BUILD_TYPE=Release"
if "%BUILD_WHEEL%"=="1" (
    call :ensure_wheel_python
    if errorlevel 1 exit /b 1
    set "OPENVINO_CMAKE_ARGS=!OPENVINO_CMAKE_ARGS! -DENABLE_PYTHON=ON -DENABLE_WHEEL=ON -DPython3_EXECUTABLE=!WHEEL_PYTHON!"
)

echo [CONFIGURE] openvino
cmake -S "%OPENVINO_SRC%" -B "%OPENVINO_BUILD%" -G Ninja !OPENVINO_CMAKE_ARGS!
if errorlevel 1 (
    echo [ERROR] CMake configure failed for openvino.
    exit /b 1
)

set "OPENVINO_ALREADY_CONFIGURED=1"
exit /b 0

:build_genai
if not exist "%OPENVINO_BUILD%\OpenVINOConfig.cmake" (
    echo [ERROR] CMake configure failed for openvino.genai.
    echo         Make sure openvino build directory exists and is valid.
    exit /b 1
)

echo [BUILD] openvino.genai
cmake -S "%GENAI_SRC%" -B "%GENAI_BUILD%" -G Ninja -DCMAKE_BUILD_TYPE=Release -DOpenVINO_DIR="%OPENVINO_BUILD%"
if errorlevel 1 (
    echo [ERROR] CMake configure failed for openvino.genai.
    echo         Make sure openvino build directory exists and is valid.
    exit /b 1
)

cmake --build "%GENAI_BUILD%" --parallel
if errorlevel 1 (
    echo [ERROR] Build failed for openvino.genai.
    exit /b 1
)
exit /b 0

:build_wheels
call :ensure_wheel_python
if errorlevel 1 exit /b 1

call :configure_openvino
if errorlevel 1 exit /b 1

call :prepare_wheel_output
if errorlevel 1 exit /b 1

echo [BUILD] openvino wheel
cmake --build "%OPENVINO_BUILD%" --config Release --target ie_wheel --parallel
if errorlevel 1 (
    echo [ERROR] Failed to build the openvino wheel.
    exit /b 1
)

copy /y "%OPENVINO_BUILD%\wheels\*.whl" "%WHEEL_OUTPUT_DIR%\" >nul
if errorlevel 1 (
    echo [ERROR] Failed to copy openvino wheel into %WHEEL_OUTPUT_DIR%.
    exit /b 1
)

echo [BUILD] openvino_tokenizers wheel
"%WHEEL_PYTHON%" -m pip wheel "%GENAI_SRC%\thirdparty\openvino_tokenizers" --wheel-dir "%WHEEL_OUTPUT_DIR%" --find-links "%WHEEL_OUTPUT_DIR%" --no-deps -v
if errorlevel 1 (
    echo [ERROR] Failed to build the openvino_tokenizers wheel.
    exit /b 1
)

set "OPENVINO_BUILD_DIR_FWD=%OPENVINO_BUILD%"
set "OPENVINO_BUILD_DIR_FWD=!OPENVINO_BUILD_DIR_FWD:\=/!"

echo [BUILD] openvino.genai wheel
"%WHEEL_PYTHON%" -m pip wheel "%GENAI_SRC%" --wheel-dir "%WHEEL_OUTPUT_DIR%" --find-links "%WHEEL_OUTPUT_DIR%" --no-deps --config-settings=--override=cmake.options.OpenVINO_DIR=!OPENVINO_BUILD_DIR_FWD! -v
if errorlevel 1 (
    echo [ERROR] Failed to build the openvino.genai wheel.
    exit /b 1
)

echo [DOWNLOAD] wheel runtime dependencies
"%WHEEL_PYTHON%" -m pip download --dest "%WHEEL_OUTPUT_DIR%" --only-binary=:all: "numpy<2.5.0,>=1.16.6" "openvino-telemetry>=2023.2.1"
if errorlevel 1 (
    echo [ERROR] Failed to download wheel runtime dependencies.
    exit /b 1
)

if not exist "%WHEEL_SCRIPT%" (
    echo [ERROR] wheel.py not found: %WHEEL_SCRIPT%
    exit /b 1
)

copy /y "%WHEEL_SCRIPT%" "%WHEEL_OUTPUT_DIR%\wheel.py" >nul
if errorlevel 1 (
    echo [ERROR] Failed to copy wheel.py into %WHEEL_OUTPUT_DIR%.
    exit /b 1
)

set "GENAI_WHEEL="
for %%I in ("%WHEEL_OUTPUT_DIR%\openvino_genai-*.whl") do (
    if not defined GENAI_WHEEL set "GENAI_WHEEL=%%~nxI"
)

echo [OK] Wheel output ready: %WHEEL_OUTPUT_DIR%
if defined GENAI_WHEEL (
    echo [INFO] Offline install example:
    echo        python -m pip install --no-index --find-links "%WHEEL_OUTPUT_DIR%" "%WHEEL_OUTPUT_DIR%\!GENAI_WHEEL!"
    echo [INFO] Smoke test example:
    echo        python "%WHEEL_OUTPUT_DIR%\wheel.py" --help
    echo        python "%WHEEL_OUTPUT_DIR%\wheel.py" --model "path\to\cached_model.xml" --device GPU --max-new-tokens 24
)
exit /b 0

:prepare_wheel_output
if not exist "%WHEEL_OUTPUT_DIR%" mkdir "%WHEEL_OUTPUT_DIR%"
if errorlevel 1 (
    echo [ERROR] Failed to create wheel output directory: %WHEEL_OUTPUT_DIR%
    exit /b 1
)

del /q "%WHEEL_OUTPUT_DIR%\openvino-*.whl" 2>nul
del /q "%WHEEL_OUTPUT_DIR%\openvino_genai-*.whl" 2>nul
del /q "%WHEEL_OUTPUT_DIR%\openvino_tokenizers-*.whl" 2>nul
del /q "%WHEEL_OUTPUT_DIR%\numpy-*.whl" 2>nul
del /q "%WHEEL_OUTPUT_DIR%\openvino_telemetry-*.whl" 2>nul
exit /b 0

:ensure_wheel_python
if defined WHEEL_ENV_READY exit /b 0

call :ensure_host_python
if errorlevel 1 exit /b 1

set "WHEEL_PYTHON=%WHEEL_VENV_DIR%\Scripts\python.exe"

if not exist "%WHEEL_PYTHON%" (
    echo [SETUP] Creating wheel build venv: %WHEEL_VENV_DIR%
    "%HOST_PYTHON%" -m venv "%WHEEL_VENV_DIR%"
    if errorlevel 1 (
        echo [ERROR] Failed to create the wheel build venv.
        exit /b 1
    )
)

echo [SETUP] Installing wheel build dependencies
"%WHEEL_PYTHON%" -m pip install --upgrade pip
if errorlevel 1 (
    echo [ERROR] Failed to upgrade pip in the wheel build venv.
    exit /b 1
)

"%WHEEL_PYTHON%" -m pip install --upgrade "setuptools>=70.1" wheel build packaging "py-build-cmake==0.5.0" "pybind11-stubgen==2.5.5"
if errorlevel 1 (
    echo [ERROR] Failed to install wheel build dependencies.
    exit /b 1
)

set "WHEEL_ENV_READY=1"
exit /b 0

:parse_args
if "%~1"=="" exit /b 0

set "ARG=%~1"
if /i "%ARG%"=="--wheel" (
    set "BUILD_WHEEL=1"
) else if /i "%ARG%"=="--help" (
    set "SHOW_USAGE=1"
) else if /i "%ARG%"=="-h" (
    set "SHOW_USAGE=1"
) else if /i "%ARG%"=="/?" (
    set "SHOW_USAGE=1"
) else (
    set "INVALID_ARG=%ARG%"
)

shift
goto :parse_args

:ensure_host_python
if defined HOST_PYTHON exit /b 0

for /f "usebackq delims=" %%I in (`where python 2^>nul`) do (
    if not defined HOST_PYTHON set "HOST_PYTHON=%%I"
)

if not defined HOST_PYTHON (
    echo [ERROR] python.exe not found in PATH.
    echo         Install Python 3.10+ and make sure it is available in PATH.
    exit /b 1
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
echo   build.bat
echo       Configure and build openvino, then configure and build openvino.genai.
echo.
echo   build.bat --wheel
echo       Build openvino, build openvino.genai, and create Python wheel files in:
echo       the "wheel" folder under the directory two levels above this build.bat
echo       The folder is created automatically if needed, and same-name wheel files are replaced.
echo.
echo   build.bat --help
echo       Show this help message.
exit /b 0
