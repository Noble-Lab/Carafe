@echo off
REM ---------------------------------------------------------------------------
REM Build the Carafe Windows installer (unsigned MSI) and bundle Osprey.
REM
REM Prerequisites (the CI workflow performs these before calling this script):
REM   1. "mvn -B -DskipTests package" has produced target\carafe-2.2.0\
REM      (carafe-2.2.0.jar plus lib\).
REM   2. The Osprey self-contained build is staged at
REM      target\carafe-2.2.0\osprey\win-x64\Osprey.exe so it is copied into
REM      the app image and resolveOspreyBinary() finds it next to the jar.
REM   3. A JDK with jpackage is on PATH, and WiX 3.x (jpackage's MSI backend) is
REM      installed and on PATH.
REM
REM This mirrors the layout Bo's installer uses (per-user install under
REM %LOCALAPPDATA%\Carafe\app\). Reconcile app name / shortcuts / vendor with the
REM canonical generate_installer_win.bat if they differ.
REM ---------------------------------------------------------------------------
setlocal

set "APP_NAME=Carafe"
REM APP_VERSION may be supplied by the CI workflow (derived from the Maven project version);
REM fall back to a literal default for local runs so a pom bump only changes one place.
if not defined APP_VERSION set "APP_VERSION=2.2.0"
set "MAIN_JAR=carafe-%APP_VERSION%.jar"
set "MAIN_CLASS=main.java.gui.CarafeLauncher"
set "INPUT_DIR=target\carafe-%APP_VERSION%"
set "DEST_DIR=target\installer"
set "ICON=src\main\resources\carafe.ico"
set "VENDOR=MacCoss Lab, University of Washington"

if not exist "%INPUT_DIR%\%MAIN_JAR%" (
    echo ERROR: %INPUT_DIR%\%MAIN_JAR% not found.
    echo        Run "mvn -B -DskipTests package" first.
    exit /b 1
)

if not exist "%INPUT_DIR%\osprey\win-x64\Osprey.exe" (
    if /I "%ALLOW_NO_OSPREY%"=="1" (
        echo WARNING: %INPUT_DIR%\osprey\win-x64\Osprey.exe not found; building
        echo          WITHOUT a bundled Osprey because ALLOW_NO_OSPREY=1.
    ) else (
        echo ERROR: %INPUT_DIR%\osprey\win-x64\Osprey.exe not found.
        echo        This script bundles Osprey; stage it first
        echo        ^(scripts\build_osprey.bat win-x64^), or set ALLOW_NO_OSPREY=1
        echo        to build a Carafe MSI without it.
        exit /b 1
    )
)

if not exist "%DEST_DIR%" mkdir "%DEST_DIR%"

echo Building %APP_NAME% %APP_VERSION% MSI from "%INPUT_DIR%" ...
jpackage ^
    --type msi ^
    --name "%APP_NAME%" ^
    --app-version "%APP_VERSION%" ^
    --vendor "%VENDOR%" ^
    --icon "%ICON%" ^
    --input "%INPUT_DIR%" ^
    --main-jar "%MAIN_JAR%" ^
    --main-class "%MAIN_CLASS%" ^
    --java-options "--enable-native-access=ALL-UNNAMED" ^
    --dest "%DEST_DIR%" ^
    --win-menu ^
    --win-shortcut ^
    --win-dir-chooser ^
    --win-per-user-install

if errorlevel 1 (
    echo ERROR: jpackage failed.
    exit /b 1
)

echo.
echo MSI written to %DEST_DIR%\
dir /b "%DEST_DIR%\*.msi"
endlocal
