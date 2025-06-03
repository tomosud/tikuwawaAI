# Chihuahua vs Chikuwa Classifier - 実装計画書

## 概要

このプロジェクトは、画像内の対象が「チワワ（犬種）」か「チクワ（食品）」かをAIで判別する画像分類ツールを作成することを目的とする。  
Pythonを用い、Windows環境で動作可能な形に構築する。

---

## フェーズ構成

### Phase 1: 画像収集（データ収集）
#### 目的：
- 「チワワ」「チクワ」の画像をネットから自動でダウンロード

#### 技術選定：
- 言語：Python 3.10+
- 検索API候補：
  - `duckduckgo_search`（API制限が少ない）
  - `Google Images Download`（制限厳しいが豊富）
- ライブラリ：
  - `requests`（画像ダウンロード）
  - `Pillow`（画像確認・処理）
  - `os` / `pathlib`（フォルダ管理）
  - `uuid`（ファイル名の一意性）

#### 構成：
```
project_root/
├── dataset/
│   ├── chihuahua/
│   └── chikuwa/
└── download_images.py
```

#### スクリプト仕様（download_images.py）：
- 検索ワードに基づいて画像URLを取得
- ファイルサイズ・拡張子チェック
- ディレクトリに保存
- 重複や壊れた画像は除外

---

### Phase 2: データ前処理
#### 目的：
- モデル学習用に画像をクリーンに整形

#### 技術選定：
- `opencv-python` or `PIL`（画像リサイズ）
- 画像サイズ統一：224x224px
- 拡張子：`.jpg` 推奨
- 自動フィルタ：
  - サイズが小さい画像除外
  - 破損画像検出

---

### Phase 3: AIモデルの学習
#### 目的：
- 「チワワ」と「チクワ」を分類する2クラス画像分類モデルを作成

#### 技術選定：
- フレームワーク：PyTorch（軽量かつWindows向けに動作安定）
- 学習方法：転移学習（例：ResNet18ベース）
- 学習環境：ローカルCPUまたはNVIDIA GPU（あれば）

#### データセット仕様：
- フォルダ構造に従って自動ラベリング（`ImageFolder`）
- 学習・検証・テストに分割（例：8:1:1）

#### 学習の流れ（概要）：
1. ResNet18をImageNet重み付きで読み込み
2. 出力層を2クラスに変更
3. 画像フォルダからデータセット作成
4. 学習・検証ループ構築
5. 精度 > 90% で保存（`.pt`ファイル）

---

### Phase 4: 推論・ツール作成
#### 目的：
- 任意の画像ファイルに対して「チワワ」or「チクワ」を判定するツールを作成

#### 技術選定：
- GUI：Gradio または Streamlit（Windowsでも簡単に動作）
- CLI版も併設可能（`predict.py`）

#### 実装案：
- `model.pt` をロード
- 画像を読み込み・リサイズ・テンソル変換
- 推論結果（クラス名＋信頼度）を表示
- GUIでは画像プレビュー付き

---

## オプション機能（将来的に）
- 面白画像ランキング（紛らわしいチクワ・チワワ）
- 判定後にTwitter投稿
- モバイル対応Webアプリ化

---

## ライブラリ一覧（pip install 用）

```bash
pip install duckduckgo-search pillow opencv-python requests torchvision torch gradio
```

---

## 注意点（Windows）
- `torch` はGPUなしでもOKだが、`torchvision` バージョンに注意
- `duckduckgo_search` は `duckduckgo-search` でpip installすること
- 画像収集においてCAPTCHAが出る場合はAPI回数を抑えるかスリープを挟む

---
