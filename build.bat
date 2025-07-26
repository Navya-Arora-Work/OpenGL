@echo off
REM PBR Material System Build Script for Windows

set BUILD_TYPE=%1
if "%BUILD_TYPE%"=="" set BUILD_TYPE=Release

set BUILD_DIR=build

echo === PBR Material System Build ===
echo Build type: %BUILD_TYPE%
echo Build directory: %BUILD_DIR%
echo =================================

REM Create build directory
if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"
cd "%BUILD_DIR%"

REM Configure with CMake (try different generators)
cmake .. -DCMAKE_BUILD_TYPE=%BUILD_TYPE% -A x64
if errorlevel 1 (
    echo Trying MinGW generator...
    cmake .. -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=%BUILD_TYPE%
)

REM Build
cmake --build . --config %BUILD_TYPE% --parallel

if errorlevel 1 (
    echo Build failed!
    pause
    exit /b 1
)

echo.
echo Build completed successfully!
echo Executable: %BUILD_DIR%\%BUILD_TYPE%\PBRMaterialSystem.exe
echo.
echo To run: cd %BUILD_DIR% ^&^& .\%BUILD_TYPE%\PBRMaterialSystem.exe
pause
