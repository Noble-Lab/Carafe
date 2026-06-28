@echo off
REM Build a self-contained Osprey executable for bundling with the Carafe
REM Windows distribution. Osprey is the MacCoss-lab DIA search tool in the
REM ProteoWizard (pwiz) tree; Carafe drives it as an external search engine.
REM
REM For the win-x64 RID this runs:
REM   dotnet publish -c Release -f net8.0 -r win-x64 --self-contained ...
REM and stages Osprey.exe under <output-root>\win-x64\ so Carafe's
REM resolveOspreyBinary() can locate it next to the jar (or under
REM %USERPROFILE%\.carafe\osprey\win-x64\) at runtime with no separate .NET
REM runtime install.
REM
REM Usage:
REM   scripts\build_osprey.bat [rid]
REM
REM Environment overrides:
REM   OSPREY_CSPROJ  Path to Osprey.csproj
REM   OUTPUT_ROOT    Where to stage per-RID output (default: <repo>\target\osprey)

setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "REPO_ROOT=%%~fI"

if "%OSPREY_CSPROJ%"=="" set "OSPREY_CSPROJ=%USERPROFILE%\GitHub-Repo\ProteoWizard\pwiz\pwiz_tools\Osprey\Osprey\Osprey.csproj"
if "%OUTPUT_ROOT%"=="" set "OUTPUT_ROOT=%REPO_ROOT%\target\osprey"
set "TARGET_FRAMEWORK=net8.0"

set "RID=%~1"
if "%RID%"=="" set "RID=win-x64"

where dotnet >nul 2>nul
if errorlevel 1 (
    echo ERROR: 'dotnet' SDK not found on PATH. Install the .NET 8 SDK first. 1>&2
    exit /b 1
)

if not exist "%OSPREY_CSPROJ%" (
    echo ERROR: Osprey.csproj not found at: %OSPREY_CSPROJ% 1>&2
    echo        Set OSPREY_CSPROJ to the correct path. 1>&2
    exit /b 1
)

set "DEST=%OUTPUT_ROOT%\%RID%"
echo Osprey csproj : %OSPREY_CSPROJ%
echo Output dir         : %DEST%
echo Target framework   : %TARGET_FRAMEWORK%
echo RID                : %RID%
echo.

if exist "%DEST%" rmdir /s /q "%DEST%"
mkdir "%DEST%"

REM Do NOT use PublishSingleFile: Osprey's blib writer uses System.Data.SQLite,
REM whose native SQLite.Interop.dll is located via the managed assembly's own directory.
REM Under single-file publish Assembly.Location is empty, so the SQLiteConnection ctor
REM throws ArgumentNullException (Parameter 'path1') when it opens the output .blib.
REM A folder publish ships SQLite.Interop.dll as a real sibling file and works.
dotnet publish "%OSPREY_CSPROJ%" ^
    -c Release ^
    -f %TARGET_FRAMEWORK% ^
    -r %RID% ^
    --self-contained true ^
    -p:PublishSingleFile=false ^
    -p:Platform=x64 ^
    -o "%DEST%"

if errorlevel 1 (
    echo ERROR: dotnet publish failed. 1>&2
    exit /b 1
)

echo.
echo Staged Osprey build under %DEST%.
echo Copy %RID%\ next to the Carafe jar as 'osprey\%RID%\' so resolveOspreyBinary()
echo finds it, or let the jpackage step include it via --input.

endlocal
