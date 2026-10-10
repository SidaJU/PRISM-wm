@echo off
chcp 65001 >nul
setlocal

REM ============================================================
REM  imagine 推理：参数全部来自 prism\config.yaml（imagine 段）
REM  每次运行自动新建结果目录: prism\runs\imagine_<时间戳>\
REM      curve.png / compare.png / steps.csv / curve.json / observe\
REM  结束后自动打开该目录里的曲线图与每步数据。
REM ============================================================

set PY=C:\Users\theozhu\miniconda3\envs\llava\python.exe
set ROOT=%~dp0..\..

echo.
echo ============================================================
echo   imagine 推理（参数来自 prism\config.yaml）
echo ============================================================

"%PY%" "%ROOT%\prism\tools\imagine_demo.py"
if errorlevel 1 (
    echo [X] 推理失败。检查 config：权重是否存在、sample_index 是否有效。
    pause
    exit /b 1
)

REM 读取本次结果目录（脚本写入 latest_imagine.txt）
set RUNFILE=%ROOT%\prism\runs\latest_imagine.txt
set RUNDIR=
if exist "%RUNFILE%" set /p RUNDIR=<"%RUNFILE%"

echo.
echo ============================================================
echo   打开本次结果
echo ============================================================

if defined RUNDIR (
    if exist "%ROOT%\%RUNDIR%\curve.png" start "" "%ROOT%\%RUNDIR%\curve.png"
    if exist "%ROOT%\%RUNDIR%\steps.csv" start "" "%ROOT%\%RUNDIR%\steps.csv"
    echo.
    echo 本次结果目录: %ROOT%\%RUNDIR%
) else (
    echo [warn] 没找到 latest_imagine.txt，打开 runs 目录
    start "" "%ROOT%\prism\runs"
)

echo.
echo 说明:
echo   curve.png     误差曲线（每步 MSE，与基线对比）
echo   compare.png   真实 vs 想象（三列对比图）
echo   steps.csv     每步数据（Excel 可开，含末端真实/预测坐标）
echo   observe\     本次用的 observe 帧
echo.
pause
