@echo off
chcp 65001 >nul
echo ====================================
echo  チワワ vs チクワ判定器 - セットアップ
echo ====================================
echo.

cd /d "%~dp0"

echo 1. Python バージョンチェック...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Pythonがインストールされていません
    echo    Python 3.8以上をインストールしてください
    echo    https://www.python.org/downloads/
    pause
    exit /b 1
)

python --version
echo ✅ Python が見つかりました
echo.

echo 2. 仮想環境チェック...
if exist "venv" (
    echo ✅ 仮想環境が既に存在します
) else (
    echo 📦 仮想環境を作成中...
    python -m venv venv
    if errorlevel 1 (
        echo ❌ 仮想環境の作成に失敗しました
        pause
        exit /b 1
    )
    echo ✅ 仮想環境を作成しました
)
echo.

echo 3. 仮想環境を有効化中...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ❌ 仮想環境の有効化に失敗しました
    pause
    exit /b 1
)
echo ✅ 仮想環境を有効化しました
echo.

echo 4. pip をアップグレード中...
python -m pip install --upgrade pip
echo.

echo 5. 依存関係をインストール中...
echo    これには数分かかる場合があります...
pip install -r requirements.txt
if errorlevel 1 (
    echo ❌ 依存関係のインストールに失敗しました
    echo    インターネット接続を確認してください
    pause
    exit /b 1
)
echo ✅ 依存関係をインストールしました
echo.

echo 6. 学習済みモデルチェック...
if exist "models\chihuawa_chikuwa_classifier.pt" (
    echo ✅ 学習済みモデルが見つかりました
) else (
    echo ⚠️ 学習済みモデルが見つかりません
    echo    モデルを学習するには以下を実行してください:
    echo    python train_model.py
)
echo.

echo 7. データセットチェック...
set /a chihuahua_count=0
set /a chikuwa_count=0

if exist "dataset\chihuahua\*.jpg" (
    for %%f in (dataset\chihuahua\*.jpg) do set /a chihuahua_count+=1
)

if exist "dataset\chikuwa\*.jpg" (
    for %%f in (dataset\chikuwa\*.jpg) do set /a chikuwa_count+=1
)

echo    チワワ画像: %chihuahua_count% 枚
echo    チクワ画像: %chikuwa_count% 枚

if %chihuahua_count% gtr 0 if %chikuwa_count% gtr 0 (
    echo ✅ データセットが見つかりました
) else (
    echo ⚠️ データセットが不完全です
    echo    画像収集を行うには以下を実行してください:
    echo    python download_images.py
)
echo.

echo 8. 簡単テスト実行...
python -c "import torch, torchvision, gradio; print('✅ 主要ライブラリのインポート成功')"
if errorlevel 1 (
    echo ❌ ライブラリのテストに失敗しました
    pause
    exit /b 1
)
echo.

echo ====================================
echo 🎉 セットアップ完了！
echo ====================================
echo.
echo 次の手順:
echo 1. Web UI起動: start_web_ui.bat をダブルクリック
echo 2. コマンドライン: python model_inference.py [画像ファイル]
echo 3. 学習実行: python train_model.py
echo 4. テスト実行: python setup_and_test.py
echo.
echo Web UIを今すぐ起動しますか？ (Y/N)
set /p choice="選択: "
if /i "%choice%"=="y" (
    echo.
    echo 🚀 Web UI を起動中...
    start_web_ui.bat
) else (
    echo.
    echo セットアップが完了しました
    echo start_web_ui.bat をダブルクリックしてWeb UIを起動できます
)

echo.
pause