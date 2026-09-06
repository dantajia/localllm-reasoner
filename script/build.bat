@echo off
chcp 65001 >nul 2>&1
setlocal enabledelayedexpansion

echo ============================================
echo   LocalLLM Build Script (Release)
echo ============================================
echo.

set "SCRIPT_DIR=%~dp0"
set "PROJECT_DIR=%SCRIPT_DIR%..\homework"
set "BUILD_DIR=%PROJECT_DIR%\build"
set "CONFIG=Release"

set "VS_PATH="
for /f "usebackq tokens=*" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -property installationPath 2^>nul`) do (
    set "VS_PATH=%%i"
)

if not defined VS_PATH (
    echo [ERROR] Visual Studio not found
    goto :error
)
echo [INFO] Visual Studio: %VS_PATH%

set "CMAKE_PATH="
where cmake >nul 2>&1
if %errorlevel% equ 0 (
    set "CMAKE_PATH=cmake"
) else (
    if exist "!VS_PATH!\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" (
        set "CMAKE_PATH=!VS_PATH!\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
    )
)
if not defined CMAKE_PATH (
    echo [ERROR] cmake not found
    goto :error
)
echo [INFO] CMake: %CMAKE_PATH%

set "MSBUILD_PATH="
where msbuild >nul 2>&1
if %errorlevel% equ 0 (
    set "MSBUILD_PATH=msbuild"
) else (
    if exist "!VS_PATH!\MSBuild\Current\Bin\amd64\MSBuild.exe" (
        set "MSBUILD_PATH=!VS_PATH!\MSBuild\Current\Bin\amd64\MSBuild.exe"
    ) else if exist "!VS_PATH!\MSBuild\Current\Bin\MSBuild.exe" (
        set "MSBUILD_PATH=!VS_PATH!\MSBuild\Current\Bin\MSBuild.exe"
    )
)
if not defined MSBUILD_PATH (
    echo [ERROR] MSBuild not found
    goto :error
)
echo [INFO] MSBuild: %MSBUILD_PATH%
echo.

if not exist "%BUILD_DIR%" (
    mkdir "%BUILD_DIR%"
)

echo [Step 1] Configuring CMake...
echo   Project : %PROJECT_DIR%
echo   Build   : %BUILD_DIR%
echo   Config  : %CONFIG%
echo.

"%CMAKE_PATH%" -S "%PROJECT_DIR%" -B "%BUILD_DIR%" -G "Visual Studio 17 2022" -A x64
if %errorlevel% neq 0 (
    echo [ERROR] CMake configuration failed
    goto :error
)

echo.
echo [Step 2] Building localllm (%CONFIG%)...
echo.

"%CMAKE_PATH%" --build "%BUILD_DIR%" --config %CONFIG% --target localllm
if %errorlevel% neq 0 (
    echo [ERROR] Build failed
    goto :error
)

echo.
echo [Step 3] Copying runtime DLLs...
echo.

set "EXE_DIR=%BUILD_DIR%\bin\%CONFIG%"
set "LLAMA_DLL_DIR=%PROJECT_DIR%\llama.cpp-master\build\bin\%CONFIG%"

if exist "%LLAMA_DLL_DIR%\*.dll" (
    copy /Y "%LLAMA_DLL_DIR%\*.dll" "%EXE_DIR%\" >nul
    echo   DLLs copied to %EXE_DIR%
) else (
    echo   [WARN] No DLLs found in %LLAMA_DLL_DIR%
    echo   If localllm.exe fails to start, please build llama.cpp first:
    echo     "%CMAKE_PATH%" -S "%PROJECT_DIR%\llama.cpp-master" -B "%PROJECT_DIR%\llama.cpp-master\build" -G "Visual Studio 17 2022" -A x64
    echo     "%CMAKE_PATH%" --build "%PROJECT_DIR%\llama.cpp-master\build" --config %CONFIG%
)

echo.
echo ============================================
echo   Build Successful
echo ============================================
echo   Output: %EXE_DIR%\localllm.exe
echo.
echo   Usage:
echo     localllm.exe --input ^<test_file^> --output ^<result_file^>
echo.
echo   Example:
echo     localllm.exe --input E:\test.txt --output E:\result_llm.json
echo ============================================

endlocal
exit /b 0

:error
echo.
echo ============================================
echo   Build Failed
echo ============================================
endlocal
exit /b 1
