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
set "GENAI_BIN_ROOT=%GENAI_BUILD%\bin"
set "WHEEL_OUTPUT_ROOT=%ROOT%\wheel"
set "WHEEL_VENV_ROOT=%SCRIPT_DIR%\.wheel-build-venv"
set "WHEEL_SCRIPT=%SCRIPT_DIR%\scripts\wheel.py"
set "TOKENIZERS_WHEEL_SCRIPT=%SCRIPT_DIR%\scripts\build_openvino_tokenizers_wheel.py"
set "WHEEL_OUTPUT_DIR="
set "WHEEL_VENV_DIR="
set "WHEEL_PYTHON="
set "WHEEL_PYTHON_SOURCE="
set "WHEEL_PYTHON_REQUEST="
set "WHEEL_PYTHON_VERSION="
set "WHEEL_TAG="

set "BUILD_OPENVINO=1"
set "BUILD_GENAI=1"
set "BUILD_WHEEL=0"
if not defined BUILD_TYPE set "BUILD_TYPE=Release"
set "INVALID_ARG="
set "ARG_ERROR="
set "SHOW_USAGE=0"
set "RAW_ARGS=%*"

if defined RAW_ARGS (
    goto :parse_args
)
goto :after_parse

:parse_args
if not defined RAW_ARGS goto :after_parse

set "ARG="
for /f "tokens=1* delims= " %%A in ("%RAW_ARGS%") do (
    set "ARG=%%A"
    set "RAW_ARGS=%%B"
)

if not defined ARG goto :after_parse
if /i "!ARG!"=="--wheel" (
    set "BUILD_WHEEL=1"
) else if /i "!ARG!"=="--python" (
    if not defined ARG_ERROR set "ARG_ERROR=Use --python=<version>."
) else if /i "!ARG:~0,9!"=="--python=" (
    set "WHEEL_PYTHON_REQUEST=!ARG:~9!"
    if "!WHEEL_PYTHON_REQUEST!"=="" if not defined ARG_ERROR set "ARG_ERROR=Missing value for --python."
) else if /i "!ARG!"=="--help" (
    set "SHOW_USAGE=1"
) else if /i "!ARG!"=="-h" (
    set "SHOW_USAGE=1"
) else if /i "!ARG!"=="/?" (
    set "SHOW_USAGE=1"
) else (
    if not defined INVALID_ARG if not defined ARG_ERROR set "INVALID_ARG=!ARG!"
)

goto :parse_args

:after_parse
if "%SHOW_USAGE%"=="1" (
    call :usage
    exit /b 0
)

if defined INVALID_ARG (
    echo [ERROR] Invalid argument: %INVALID_ARG%
    call :usage
    exit /b 1
)

if defined ARG_ERROR (
    echo [ERROR] %ARG_ERROR%
    call :usage
    exit /b 1
)

if defined WHEEL_PYTHON_REQUEST if not "%BUILD_WHEEL%"=="1" (
    echo [ERROR] --python can only be used together with --wheel.
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

set "OPENVINO_CMAKE_ARGS=-DCMAKE_BUILD_TYPE=%BUILD_TYPE%"
if "%BUILD_WHEEL%"=="1" (
    call :ensure_wheel_python
    if errorlevel 1 exit /b 1
    set "OPENVINO_CMAKE_ARGS=!OPENVINO_CMAKE_ARGS! -DENABLE_PYTHON=ON -DENABLE_WHEEL=ON -DPython3_EXECUTABLE=!WHEEL_PYTHON!"
 ) else (
    rem Clear wheel-only cache entries so a previous --wheel configure cannot break a normal build.
    set "OPENVINO_CMAKE_ARGS=!OPENVINO_CMAKE_ARGS! -DENABLE_PYTHON=OFF -DENABLE_WHEEL=OFF"
    set "OPENVINO_CMAKE_ARGS=!OPENVINO_CMAKE_ARGS! -UPython3_EXECUTABLE -UPython3_ROOT_DIR -UPython3_INCLUDE_DIR"
    set "OPENVINO_CMAKE_ARGS=!OPENVINO_CMAKE_ARGS! -UPython3_LIBRARY -UPython3_LIBRARY_RELEASE -UPython3_LIBRARY_DEBUG"
    set "OPENVINO_CMAKE_ARGS=!OPENVINO_CMAKE_ARGS! -U_Python3_EXECUTABLE -U_Python3_INCLUDE_DIR"
    set "OPENVINO_CMAKE_ARGS=!OPENVINO_CMAKE_ARGS! -U_Python3_LIBRARY_RELEASE -U_Python3_LIBRARY_DEBUG"
    set "OPENVINO_CMAKE_ARGS=!OPENVINO_CMAKE_ARGS! -UPYBIND11_PYTHON_EXECUTABLE_LAST -UFIND_PACKAGE_MESSAGE_DETAILS_Python3"
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
set "GENAI_CMAKE_ARGS=-DCMAKE_BUILD_TYPE=%BUILD_TYPE% -DOpenVINO_DIR=%OPENVINO_BUILD%"
if "%BUILD_WHEEL%"=="1" (
    call :ensure_wheel_python
    if errorlevel 1 exit /b 1
    rem Clear stale Python cache entries before switching wheel Python versions or layouts.
    set "GENAI_CMAKE_ARGS=!GENAI_CMAKE_ARGS! -DENABLE_PYTHON=ON -UPython3_EXECUTABLE -UPython3_ROOT_DIR -UPython3_INCLUDE_DIR"
    set "GENAI_CMAKE_ARGS=!GENAI_CMAKE_ARGS! -UPython3_LIBRARY -UPython3_LIBRARY_RELEASE -UPython3_LIBRARY_DEBUG"
    set "GENAI_CMAKE_ARGS=!GENAI_CMAKE_ARGS! -U_Python3_EXECUTABLE -U_Python3_INCLUDE_DIR"
    set "GENAI_CMAKE_ARGS=!GENAI_CMAKE_ARGS! -U_Python3_LIBRARY_RELEASE -U_Python3_LIBRARY_DEBUG"
    set "GENAI_CMAKE_ARGS=!GENAI_CMAKE_ARGS! -UPYBIND11_PYTHON_EXECUTABLE_LAST -UFIND_PACKAGE_MESSAGE_DETAILS_Python3"
    set "GENAI_CMAKE_ARGS=!GENAI_CMAKE_ARGS! -DPython3_EXECUTABLE=!WHEEL_PYTHON!"
) else (
    rem Keep the default build independent from Python and any wheel-only cache state.
    set "GENAI_CMAKE_ARGS=!GENAI_CMAKE_ARGS! -DENABLE_PYTHON=OFF -UPython3_EXECUTABLE -UPython3_ROOT_DIR -UPython3_INCLUDE_DIR"
    set "GENAI_CMAKE_ARGS=!GENAI_CMAKE_ARGS! -UPython3_LIBRARY -UPython3_LIBRARY_RELEASE -UPython3_LIBRARY_DEBUG"
    set "GENAI_CMAKE_ARGS=!GENAI_CMAKE_ARGS! -U_Python3_EXECUTABLE -U_Python3_INCLUDE_DIR"
    set "GENAI_CMAKE_ARGS=!GENAI_CMAKE_ARGS! -U_Python3_LIBRARY_RELEASE -U_Python3_LIBRARY_DEBUG"
    set "GENAI_CMAKE_ARGS=!GENAI_CMAKE_ARGS! -UPYBIND11_PYTHON_EXECUTABLE_LAST -UFIND_PACKAGE_MESSAGE_DETAILS_Python3"
)

cmake -S "%GENAI_SRC%" -B "%GENAI_BUILD%" -G Ninja !GENAI_CMAKE_ARGS!
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

call :stage_genai_bin_layout
if errorlevel 1 exit /b 1
exit /b 0

:build_wheels
call :ensure_wheel_python
if errorlevel 1 exit /b 1

call :configure_openvino
if errorlevel 1 exit /b 1

call :prepare_wheel_output
if errorlevel 1 exit /b 1

echo [BUILD] openvino wheel
cmake --build "%OPENVINO_BUILD%" --config %BUILD_TYPE% --target ie_wheel --parallel
if errorlevel 1 (
    echo [ERROR] Failed to build the openvino wheel.
    exit /b 1
)

call :copy_latest_openvino_wheel
if errorlevel 1 exit /b 1

set "OPENVINO_BUILD_DIR_FWD=%OPENVINO_BUILD%"
set "OPENVINO_BUILD_DIR_FWD=!OPENVINO_BUILD_DIR_FWD:\=/!"

echo [BUILD] openvino_tokenizers wheel
if not exist "%TOKENIZERS_WHEEL_SCRIPT%" (
    echo [ERROR] build_openvino_tokenizers_wheel.py not found: %TOKENIZERS_WHEEL_SCRIPT%
    exit /b 1
)

"%WHEEL_PYTHON%" "%TOKENIZERS_WHEEL_SCRIPT%" --source-dir "%GENAI_SRC%\thirdparty\openvino_tokenizers" --build-dir "%GENAI_BUILD%" --wheel-dir "%WHEEL_OUTPUT_DIR%"
if errorlevel 1 (
    echo [ERROR] Failed to build the openvino_tokenizers wheel.
    exit /b 1
)

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
if not exist "%WHEEL_OUTPUT_ROOT%" mkdir "%WHEEL_OUTPUT_ROOT%"
if errorlevel 1 (
    echo [ERROR] Failed to create wheel output root: %WHEEL_OUTPUT_ROOT%
    exit /b 1
)

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
del /q "%WHEEL_OUTPUT_DIR%\wheel.py" 2>nul
exit /b 0

:ensure_wheel_python
if defined WHEEL_ENV_READY exit /b 0

call :resolve_wheel_context
if errorlevel 1 exit /b 1

set "WHEEL_PYTHON=%WHEEL_VENV_DIR%\Scripts\python.exe"
set "WHEEL_VENV_INFO_FILE=%WHEEL_VENV_DIR%\.wheel-env-info.txt"

if not defined WHEEL_VENV_ROOT_READY (
    if exist "%WHEEL_VENV_ROOT%\pyvenv.cfg" (
        echo [SETUP] Removing legacy single-version wheel venv layout: %WHEEL_VENV_ROOT%
        rmdir /s /q "%WHEEL_VENV_ROOT%"
        if errorlevel 1 (
            echo [ERROR] Failed to remove the legacy wheel build venv directory.
            exit /b 1
        )
    )

    if not exist "%WHEEL_VENV_ROOT%" mkdir "%WHEEL_VENV_ROOT%"
    if errorlevel 1 (
        echo [ERROR] Failed to create wheel venv root: %WHEEL_VENV_ROOT%
        exit /b 1
    )

    set "WHEEL_VENV_ROOT_READY=1"
)

set "CREATE_WHEEL_VENV=0"
if exist "%WHEEL_PYTHON%" (
    set "CURRENT_WHEEL_VENV_VERSION="
    set "CURRENT_WHEEL_VENV_SOURCE="
    if not exist "%WHEEL_VENV_INFO_FILE%" (
        echo [SETUP] Existing wheel build venv is missing metadata and will be recreated: %WHEEL_VENV_DIR%
        set "CREATE_WHEEL_VENV=1"
    ) else (
        for /f "usebackq tokens=1* delims=|" %%I in ("%WHEEL_VENV_INFO_FILE%") do (
            if not defined CURRENT_WHEEL_VENV_VERSION set "CURRENT_WHEEL_VENV_VERSION=%%I"
            if not defined CURRENT_WHEEL_VENV_SOURCE set "CURRENT_WHEEL_VENV_SOURCE=%%J"
        )

        if not defined CURRENT_WHEEL_VENV_VERSION (
            echo [SETUP] Existing wheel build venv metadata is invalid and will be recreated: %WHEEL_VENV_DIR%
            set "CREATE_WHEEL_VENV=1"
        ) else if /i not "!CURRENT_WHEEL_VENV_VERSION!"=="%WHEEL_PYTHON_VERSION%" (
            echo [SETUP] Existing wheel build venv uses Python !CURRENT_WHEEL_VENV_VERSION! and will be recreated for %WHEEL_PYTHON_VERSION%.
            set "CREATE_WHEEL_VENV=1"
        ) else if not defined CURRENT_WHEEL_VENV_SOURCE (
            echo [SETUP] Existing wheel build venv metadata is incomplete and will be recreated: %WHEEL_VENV_DIR%
            set "CREATE_WHEEL_VENV=1"
        ) else if /i not "!CURRENT_WHEEL_VENV_SOURCE!"=="%WHEEL_PYTHON_SOURCE%" (
            echo [SETUP] Existing wheel build venv uses !CURRENT_WHEEL_VENV_SOURCE! and will be recreated for %WHEEL_PYTHON_SOURCE%.
            set "CREATE_WHEEL_VENV=1"
        ) else (
            echo [SETUP] Reusing wheel build venv: %WHEEL_VENV_DIR%
        )
    )
) else (
    set "CREATE_WHEEL_VENV=1"
)

if "%CREATE_WHEEL_VENV%"=="1" (
    if exist "%WHEEL_VENV_DIR%" (
        rmdir /s /q "%WHEEL_VENV_DIR%"
        if errorlevel 1 (
            echo [ERROR] Failed to remove wheel build venv: %WHEEL_VENV_DIR%
            exit /b 1
        )
    )

    echo [SETUP] Creating wheel build venv for Python %WHEEL_PYTHON_VERSION%: %WHEEL_VENV_DIR%
    uv venv --seed --python "%WHEEL_PYTHON_SOURCE%" "%WHEEL_VENV_DIR%"
    if errorlevel 1 (
        echo [ERROR] Failed to create the wheel build venv with uv.
        exit /b 1
    )

    >"%WHEEL_VENV_INFO_FILE%" echo %WHEEL_PYTHON_VERSION%^|%WHEEL_PYTHON_SOURCE%
    if errorlevel 1 (
        echo [ERROR] Failed to record wheel build venv interpreter metadata.
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

:resolve_wheel_context
if defined WHEEL_CONTEXT_READY exit /b 0

call :ensure_uv
if errorlevel 1 exit /b 1

if defined WHEEL_PYTHON_REQUEST (
    echo [SETUP] Ensuring Python %WHEEL_PYTHON_REQUEST% is available via uv
    uv python install "%WHEEL_PYTHON_REQUEST%"
    if errorlevel 1 (
        echo [ERROR] Failed to install or locate Python %WHEEL_PYTHON_REQUEST% with uv.
        exit /b 1
    )

    for /f "usebackq delims=" %%I in (`uv python find --show-version "%WHEEL_PYTHON_REQUEST%" 2^>nul`) do (
        if not defined WHEEL_PYTHON_VERSION set "WHEEL_PYTHON_VERSION=%%I"
    )
    for /f "tokens=1* delims= " %%I in ('uv python list --managed-python --only-installed "%WHEEL_PYTHON_REQUEST%" 2^>nul') do (
        if not defined WHEEL_PYTHON_SOURCE set "WHEEL_PYTHON_SOURCE=%%J"
    )

    if not defined WHEEL_PYTHON_SOURCE (
        echo [ERROR] uv could not resolve a managed Python interpreter for %WHEEL_PYTHON_REQUEST%.
        exit /b 1
    )
    if not defined WHEEL_PYTHON_VERSION (
        echo [ERROR] uv could not resolve the exact Python version for %WHEEL_PYTHON_REQUEST%.
        exit /b 1
    )
) else (
    call :ensure_host_python
    if errorlevel 1 exit /b 1

    set "WHEEL_PYTHON_SOURCE=%HOST_PYTHON%"
    call :get_python_version "%HOST_PYTHON%" WHEEL_PYTHON_VERSION
    if not defined WHEEL_PYTHON_VERSION (
        echo [ERROR] Failed to detect the default python version from PATH: %HOST_PYTHON%
        exit /b 1
    )
)

call :make_wheel_tag "%WHEEL_PYTHON_VERSION%" WHEEL_TAG
if not defined WHEEL_TAG (
    echo [ERROR] Failed to derive the wheel tag from Python %WHEEL_PYTHON_VERSION%.
    exit /b 1
)

set "WHEEL_VENV_DIR=%WHEEL_VENV_ROOT%\%WHEEL_PYTHON_VERSION%"
set "WHEEL_OUTPUT_DIR=%WHEEL_OUTPUT_ROOT%\%WHEEL_TAG%"

echo [INFO] Wheel build Python: %WHEEL_PYTHON_VERSION% (%WHEEL_TAG%)
if defined WHEEL_PYTHON_REQUEST (
    echo [INFO] Resolved from --python=%WHEEL_PYTHON_REQUEST%
) else (
    echo [INFO] Using default python from PATH: %WHEEL_PYTHON_SOURCE%
)

set "WHEEL_CONTEXT_READY=1"
exit /b 0

:copy_latest_openvino_wheel
set "OPENVINO_WHEEL="
for /f "delims=" %%I in ('dir /b /a-d /o-d "%OPENVINO_BUILD%\wheels\openvino-*-%WHEEL_TAG%-%WHEEL_TAG%-*.whl" 2^>nul') do (
    if not defined OPENVINO_WHEEL set "OPENVINO_WHEEL=%%I"
)

if not defined OPENVINO_WHEEL (
    echo [ERROR] Failed to locate the newly built openvino wheel for %WHEEL_TAG% in %OPENVINO_BUILD%\wheels.
    exit /b 1
)

copy /y "%OPENVINO_BUILD%\wheels\%OPENVINO_WHEEL%" "%WHEEL_OUTPUT_DIR%\" >nul
if errorlevel 1 (
    echo [ERROR] Failed to copy %OPENVINO_WHEEL% into %WHEEL_OUTPUT_DIR%.
    exit /b 1
)
exit /b 0

:stage_genai_bin_layout
set "GENAI_BIN_DIR=%GENAI_BIN_ROOT%\%BUILD_TYPE%"
if not exist "%GENAI_BIN_ROOT%" (
    echo [ERROR] OpenVINO GenAI bin directory not found: %GENAI_BIN_ROOT%
    exit /b 1
)

if not exist "%GENAI_BIN_DIR%" mkdir "%GENAI_BIN_DIR%"
if errorlevel 1 (
    echo [ERROR] Failed to create OpenVINO GenAI build-type bin directory: %GENAI_BIN_DIR%
    exit /b 1
)

del /q "%GENAI_BIN_DIR%\*.dll" 2>nul
if errorlevel 1 (
    echo [ERROR] Failed to clean stale OpenVINO GenAI runtime DLLs from %GENAI_BIN_DIR%.
    exit /b 1
)

for %%I in ("%GENAI_BIN_ROOT%\*.exe") do (
    if exist "%%~fI" copy /y "%%~fI" "%GENAI_BIN_DIR%\" >nul
)
if errorlevel 1 (
    echo [ERROR] Failed to stage OpenVINO GenAI executable files into %GENAI_BIN_DIR%.
    exit /b 1
)

for %%I in ("%GENAI_BIN_ROOT%\*.dll") do (
    if exist "%%~fI" copy /y "%%~fI" "%GENAI_BIN_DIR%\" >nul
)
if errorlevel 1 (
    echo [ERROR] Failed to stage OpenVINO GenAI runtime DLLs into %GENAI_BIN_DIR%.
    exit /b 1
)

echo [INFO] OpenVINO GenAI executable and runtime DLL files staged in: %GENAI_BIN_DIR%
exit /b 0

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

:ensure_uv
if defined UV_READY exit /b 0

where uv >nul 2>nul
if errorlevel 1 (
    echo [ERROR] uv.exe not found in PATH.
    echo         Install uv and make sure it is available in PATH before running build.bat --wheel.
    exit /b 1
)

set "UV_READY=1"
exit /b 0

:get_python_version
set "%~2="
for /f "usebackq delims=" %%I in (`"%~1" -c "import sys; print('.'.join(str(x) for x in sys.version_info[:3]))" 2^>nul`) do (
    set "%~2=%%I"
)
exit /b 0

:make_wheel_tag
set "%~2="
for /f "tokens=1,2 delims=." %%A in ("%~1") do (
    if not "%%A"=="" if not "%%B"=="" set "%~2=cp%%A%%B"
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
echo   build.bat --wheel [--python=3.11.9]
echo       Build openvino, build openvino.genai, and create Python wheel files in:
echo       the "wheel\cpXY" folder under the directory two levels above this build.bat
echo       The Python version defaults to the first python.exe in PATH unless --python is specified.
echo       Wheel build virtual environments are reused under ".wheel-build-venv\<python-version>".
echo.
echo   build.bat --help
echo       Show this help message.
exit /b 0
