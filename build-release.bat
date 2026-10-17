@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b 1
set "PATH=C:\Users\pscho\.cargo\bin;%PATH%"
cd /d C:\Users\pscho\simu3D
cargo build --release
echo CARGO_EXIT=%ERRORLEVEL%
exit /b %ERRORLEVEL%
