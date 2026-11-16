@echo off
rem Full build: configure, clean, build (the all-in-one entry point).
rem Run from anywhere: all paths are resolved relative to this script.
setlocal
set "ROOT=%~dp0"
set "ROOTU=%ROOT:\=/%"

rem Locate Visual Studio via vswhere, falling back to the development install
rem (adjust the fallback if your Visual Studio lives elsewhere).
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
set "VCVARS="
if exist "%VSWHERE%" for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VCVARS=%%i\VC\Auxiliary\Build\vcvars64.bat"
if not defined VCVARS set "VCVARS=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
call "%VCVARS%" >nul

rem Git for Windows provides the bash used to run the CMake/ninja wrapper.
"%ProgramFiles%\Git\bin\bash.exe" -c "cd '%ROOTU%' && ./configure-msvc.sh && ./make-msvc.sh clean && ./make-msvc.sh" > "%ROOT%build-gpu.log" 2>&1
exit /b %ERRORLEVEL%
