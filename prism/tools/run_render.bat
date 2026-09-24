@echo off
rem GSWM: MuJoCo -> UE5 replay render (offscreen)
rem Usage: _gswm_run.bat [OutDir]
set OUT=%1
if "%OUT%"=="" set OUT=D:/gswm_l1

if exist "%OUT%" rmdir /s /q "%OUT%"

start "GSWMRender" "g:\theozhu_THEOZHU-PC1_4001\RaidGame\Binaries\Win64\RaidGameEditor-Cmd.exe" ^
  "g:\theozhu_THEOZHU-PC1_4001\RaidGame\RaidGame.uproject" ^
  /Engine/Maps/Templates/Template_Default ^
  -game -RenderOffScreen -ResX=320 -ResY=240 -NoSplash -unattended ^
  -GSWMReplay -GSWMIn=D:/gswm_poses -GSWMOut=%OUT%

echo started out=%OUT%
exit /b 0
