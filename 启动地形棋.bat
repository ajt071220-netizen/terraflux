@echo off
chcp 65001 >nul
cd /d "%~dp0"
title 涌陆 Terraflux

where node >nul 2>nul
if errorlevel 1 (
  echo.
  echo  [错误] 没有找到 Node.js
  echo  请先到 https://nodejs.org 下载安装 LTS 版本（一路下一步即可），
  echo  安装完成后重新双击本文件。
  echo.
  pause
  exit /b 1
)

if not exist lib\three.module.js (
  echo.
  echo  [缺少文件] 需要先下载 three.js 库文件（只有这一步需要联网）。
  echo  请打开 lib 文件夹，按照里面的"下载说明.txt"下载两个文件。
  echo  下好之后再双击本文件。
  echo.
  pause
  exit /b 1
)

echo  正在启动涌陆（Terraflux）服务……
echo  浏览器稍后会自动打开。若页面显示"无法连接"，以本窗口里
echo  显示的地址为准，手动复制到浏览器打开。
echo.
echo  ※ 游戏运行期间请勿关闭本窗口。
echo.

start "" cmd /c "timeout /t 2 >nul & start http://localhost:5173"
node server.js
pause
