@echo off
rem GSWM: 使用关卡自带光照/后期处理渲染（画面由编辑器里的关卡决定）
rem Usage: run_render_levellook.bat [MapPath] [OutDir] [MaxEpisodes]
set MAP=%1
if "%MAP%"=="" set MAP=/Engine/Maps/Templates/THEOZHU
set OUT=%2
if "%OUT%"=="" set OUT=D:/gswm_l1
set MAXEP=%3
if "%MAXEP%"=="" set MAXEP=1300

if exist "%OUT%" rmdir /s /q "%OUT%"

start "GSWMRender" "g:\theozhu_THEOZHU-PC1_4001\RaidGame\Binaries\Win64\RaidGameEditor-Cmd.exe" ^
  "g:\theozhu_THEOZHU-PC1_4001\RaidGame\RaidGame.uproject" ^
  %MAP% ^
  -game -RenderOffScreen -ResX=320 -ResY=240 -NoSplash -unattended ^
  -GSWMReplay -GSWMLevelLook -GSWMIn=D:/gswm_poses -GSWMOut=%OUT% -GSWMMaxEp=%MAXEP%

echo started map=%MAP% out=%OUT% maxEp=%MAXEP%
exit /b 0
