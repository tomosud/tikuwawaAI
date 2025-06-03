# Chihuahua vs Chikuwa Classifier

チワワ（犬種）とチクワ（食品）を画像から判別するAI分類ツール

## 概要

このプロジェクトは、画像内の対象が「チワワ（犬種）」か「チクワ（食品）」かをAIで判別する画像分類ツールです。
Pythonを用い、Windows環境で動作可能な形に構築します。

## 環境要件

- Python 3.10+
- Windows 11
- 仮想環境：`C:\work\script\tikuwawaAI\venv`

## セットアップ

### 1. 仮想環境の作成と有効化

```bash
# 仮想環境作成
python -m venv venv

# 仮想環境有効化 (Windows)
venv\Scripts\activate
```

### 2. 依存関係のインストール

```bash
pip install -r requirements.txt
```

## フェーズ構成

### Phase 1: 画像収集（データ収集）✅ **現在実装中**

**目的：** 「チワワ」「チクワ」の画像をネットから自動でダウンロード

**技術選定：**
- `duckduckgo_search`（API制限が少ない検索エンジン）
- `requests`（画像ダウンロード）
- `Pillow`（画像確認・処理）
- `pathlib`（フォルダ管理）
- `uuid`（ファイル名の一意性）

**ディレクトリ構造：**
```
project_root/
├── dataset/
│   ├── chihuahua/
│   └── chikuwa/
└── download_images.py
```

**機能：**
- 検索ワードに基づいてDuckDuckGoから画像URLを取得
- ファイルサイズ・拡張子チェック
- 重複や壊れた画像の除外
- 各カテゴリごとにディレクトリに保存

### Phase 2: データ前処理

**目的：** モデル学習用に画像をクリーンに整形

**技術選定：**
- `opencv-python` / `PIL`（画像リサイズ）
- 画像サイズ統一：224x224px
- 拡張子：`.jpg` 推奨

### Phase 3: AIモデルの学習

**目的：** 「チワワ」と「チクワ」を分類する2クラス画像分類モデルを作成

**技術選定：**
- フレームワーク：PyTorch
- 学習方法：転移学習（ResNet18ベース）
- 学習環境：ローカルCPU/GPU

### Phase 4: 推論・ツール作成

**目的：** 任意の画像ファイルに対して「チワワ」or「チクワ」を判定するツールを作成

**技術選定：**
- GUI：Gradio
- CLI版も併設予定

## 使用方法

### Phase 1: 画像収集の実行

```bash
python download_images.py
```

## ライセンス

MIT License

## 注意点（Windows環境）

- `torch` はGPUなしでもOKですが、`torchvision` バージョンに注意
- `duckduckgo_search` は `duckduckgo-search` でインストール
- 画像収集においてCAPTCHAが出る場合はAPI回数を抑えるかスリープを挟む