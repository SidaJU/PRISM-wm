@echo off
chcp 65001 >nul
setlocal

REM ============================================================
REM  RSSM 训练：参数来自 prism\config.yaml（train 段）
REM  每次运行自动新建目录: prism\runs\train_<时间戳>\
REM      rssm_epoch*.pt / loss_log.json / loss_curves.png
REM  结束后自动打开 loss 曲线确认收敛。
REM ============================================================

set PY=C:\Users\theozhu\miniconda3\envs\llava\python.exe
set ROOT=%~dp0..\..

echo.
echo ============================================================
echo   1/3  训练 RSSM（参数来自 prism\config.yaml）
echo ============================================================

"%PY%" "%ROOT%\prism\train\train_rssm.py"
if errorlevel 1 (
    echo [X] 训练失败。
    pause
    exit /b 1
)

REM 读取本次训练目录（脚本写入 latest_train.txt）
set RUNFILE=%ROOT%\prism\runs\latest_train.txt
set RUNDIR=
if exist "%RUNFILE%" set /p RUNDIR=<"%RUNFILE%"

if not defined RUNDIR (
    echo [X] 没找到 latest_train.txt，无法定位本次输出。
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   2/3  生成 loss 曲线（4 项分解 + 收敛判断）
echo ============================================================

"%PY%" "%ROOT%\prism\tools\plot_loss.py" "%ROOT%\%RUNDIR%\loss_log.json" --out "%ROOT%\%RUNDIR%\loss_curves.png"
if errorlevel 1 (
    echo [X] 画图失败。
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   3/3  打开结果
echo ============================================================

start "" "%ROOT%\%RUNDIR%\loss_curves.png"

echo.
echo 完成！
echo   本次目录: %ROOT%\%RUNDIR%
echo     权重      : rssm_epoch*.pt
echo     loss 数值 : loss_log.json
echo     曲线图    : loss_curves.png
echo.
echo 判断收敛: recon / emb / proprio 应大幅下降后走平；kl 应稳定在 free_bits 附近（不塌到 0）。
echo.
pause
