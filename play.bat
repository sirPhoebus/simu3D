@echo off
setlocal
rem Explorer starts a UNC batch in C:\Windows. Map the script folder first.
pushd "%~dp0"
set "HERE=%CD%"
rem wsl.exe maps the Windows cwd into Linux. A pushd drive that points
rem back at \\wsl$ cannot be translated, so leave it before calling wsl.
cd /d "%SystemRoot%"
rem Launch the WSL build, keep the Windows pointer inside the simu3d window, and hide it.
rem Alt-Tab shows the pointer and lets it out. Closing the game does too.
rem Esc stops look. The pointer stays hidden while the window is in front.
set "DISTRO=Ubuntu-22.04"
set "APP=/home/psc/repo/simu3D/target/release/simu3d"
set "STOP=%TEMP%\simu3d-clip.stop"
set "LOG=%TEMP%\simu3d-clip.log"
del "%STOP%" 2>nul
del "%LOG%" 2>nul

wsl.exe -d %DISTRO% -- test -x %APP%
if errorlevel 1 (
    echo simu3d release binary is missing.
    echo In WSL: cd /home/psc/repo/simu3D ^&^& cargo build --release
    popd
    exit /b 1
)

echo Pointer stays inside simu3d and stays hidden while that window is in front. Alt-Tab shows it again.
start "" /MIN powershell.exe -NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File "%HERE%\clip-cursor.ps1" -StopFile "%STOP%" -LogFile "%LOG%"
wsl.exe -d %DISTRO% -e bash -lc "cd /home/psc/repo/simu3D && exec ./target/release/simu3d"
set "RC=%ERRORLEVEL%"
echo.>"%STOP%"
popd
endlocal & exit /b %RC%
