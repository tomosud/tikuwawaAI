@echo off
echo ====================================
echo  🐕 チワワ vs チクワ判定器 🥢
echo ====================================
echo.
echo Web UIを起動しています...
echo ブラウザが自動的に開きます
echo.
echo 終了するには Ctrl+C を押してください
echo ====================================
echo.

cd /d "%~dp0"

REM 仮想環境の有効化
if exist "venv\Scripts\activate.bat" (
    echo 仮想環境を有効化中...
    call venv\Scripts\activate.bat
) else (
    echo ⚠️ 仮想環境が見つかりません
    echo セットアップを実行してください: python setup_and_test.py
    pause
    exit /b 1
)

REM 依存関係のチェック
python -c "import gradio" 2>nul
if errorlevel 1 (
    echo 📦 依存関係をインストール中...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ❌ 依存関係のインストールに失敗しました
        pause
        exit /b 1
    )
)

REM Web UI起動
echo 🚀 Web UI を起動中...
python web_ui.py

echo.
echo Web UI が終了しました
pause