# Chihuahua vs Chikuwa Classifier

チワワ（犬種）とチクワ（食品）を画像から判別するAI分類ツール

## 概要

このプロジェクトは、画像内の対象が「チワワ（犬種）」か「チクワ（食品）」かをAIで判別する画像分類ツールです。
Pythonを用い、Windows環境で動作可能な形に構築します。

## 環境要件

- Python 3.11
- Windows 11
- 推奨RAM: 4GB以上

## 🚀 クイックスタート（Gitクローン後）

このリポジトリには学習済みモデルとデータセットが含まれています。クローン後すぐに使用できます。

### 1. リポジトリのクローン

```bash
git clone <this-repository-url>
cd tikuwawaAI
```

### 2. 自動セットアップとテスト

```bash
# 環境セットアップとモデル動作確認を一括実行
python setup_and_test.py
```

### 3. 即座に推論実行

```bash
# チワワ画像の分類
python model_inference.py dataset/chihuahua/chihuahua_d2537840_000.jpg

# チクワ画像の分類（詳細表示）
python model_inference.py dataset/chikuwa/chikuwa_fb74c371_000.jpg --detailed
```

## 手動セットアップ（オプション）

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

### Phase 1: 画像収集（データ収集） ✅ **完了**

**目的：** 「チワワ」「チクワ」の画像をネットから自動でダウンロード

**実装完了状況：**
- ✅ **データセット**: チワワ25枚、チクワ25枚（計50枚）
- ✅ **品質管理**: パッケージ画像等を除外し、判定に適した画像のみ選別
- ✅ **Git管理**: フェアユース範囲で画像をリポジトリに含めて管理
- ✅ **エラーハンドリング**: レート制限回避とロバストな収集処理

**使用技術：**
- `duckduckgo_search`（API制限回避実装済み）
- `requests`（画像ダウンロード）
- `Pillow`（画像検証・処理）
- `pathlib`（フォルダ管理）
- `uuid`（ファイル名の一意性）

**最終ディレクトリ構造：**
```
project_root/
├── dataset/
│   ├── chihuahua/          # チワワ画像 25枚
│   └── chikuwa/            # チクワ画像 25枚（50枚から選別）
├── download_images.py      # 詳細版収集スクリプト
└── simple_download.py      # 実際に成功したシンプル版
```

**実装済み機能：**
- 複数検索クエリによる多様な画像収集
- 適切なファイルサイズ・形式チェック（1KB-5MB）
- レート制限回避（0.8秒間隔）
- UUID付きファイル名で重複防止
- エラーハンドリングとログ出力
- 人的品質管理による最終選別

### Phase 2: データ前処理 ✅ **完了**

**目的：** モデル学習用に画像をクリーンに整形

**実装完了状況：**
- ✅ **画像リサイズ**: 224x224pxに統一（学習時は256→224でRandomCrop）
- ✅ **正規化処理**: ImageNet標準値で正規化 `mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]`
- ✅ **データ分割**: 学習40枚・検証5枚・テスト5枚（8:1:1比率）
- ✅ **データ拡張**: RandomCrop、HorizontalFlip、Rotation、ColorJitter実装
- ✅ **PyTorch DataLoader**: バッチ処理、シャッフル、num_workers対応
- ✅ **画像検証**: 破損画像の自動検出・除外
- ✅ **可視化機能**: クラス分布グラフ、サンプル画像表示、拡張効果確認

**使用技術：**
- `PyTorch` + `torchvision`（データローダー・変換）
- `PIL`（画像処理）
- `matplotlib`（可視化）
- `numpy`（数値計算）

**出力ファイル：**
```
processed_data/
├── data_split.json           # データ分割情報・統計
└── class_distribution.png    # クラス分布グラフ
```

**実装済み機能：**
- カスタムDatasetクラス（ChihuahuaChikuwaDataset）
- 段階的データ変換（学習時は拡張あり、検証・テスト時は拡張なし）
- 詳細な統計情報とクラスバランス分析
- エラーハンドリングと画像整合性チェック
- 可視化ツールによる品質確認

### Phase 3: AIモデルの学習 ✅ **完了**

**目的：** 「チワワ」と「チクワ」を分類する2クラス画像分類モデルを作成

**実装完了結果：**
- ✅ **モデル**: ResNet18転移学習
- ✅ **達成精度**: **100%**（目標90%を大幅に上回る）
- ✅ **検証精度**: 100.00%
- ✅ **テスト精度**: 100.00%
- ✅ **学習済みモデル**: [`models/chihuawa_chikuwa_classifier.pt`](models/chihuawa_chikuwa_classifier.pt)
- ✅ **学習履歴**: [`models/training_history.png`](models/training_history.png)

**使用技術：**
- PyTorch + ResNet18（転移学習）
- 25エポック学習
- Adam オプティマイザ
- CrossEntropy 損失関数

### Phase 4: 推論・ツール作成 ✅ **完了**

**目的：** 任意の画像ファイルに対して「チワワ」or「チクワ」を判定するツールを作成

**実装完了内容：**
- ✅ **推論スクリプト**: [`model_inference.py`](model_inference.py) - コマンドライン実行
- ✅ **自動セットアップ**: [`setup_and_test.py`](setup_and_test.py) - 環境構築と動作確認
- ✅ **信頼度表示**: 確率とパーセンテージ表示
- ✅ **バッチ処理**: 複数画像の一括処理対応
- ✅ **エラーハンドリング**: 不正画像・モデル不備の検出

## 使用方法

### Phase 1: 画像収集（完了済み）

データセットは既に収集・選別済みです。追加で画像収集する場合：

```bash
# シンプル版（推奨）
python simple_download.py

# 詳細版
python download_images.py
```

### データセット確認

```bash
# データセット構造確認
ls dataset/chihuahua/ | wc -l  # チワワ画像数
ls dataset/chikuwa/ | wc -l    # チクワ画像数
```

## 🎉 プロジェクト完了状況

### ✅ 全フェーズ完了

- **Phase 1: 画像収集** - チワワ25枚、チクワ25枚の高品質データセット構築
- **Phase 2: データ前処理** - PyTorchDataLoader対応、拡張、分割、可視化
- **Phase 3: AIモデル学習** - **100%精度**の分類器完成
- **Phase 4: 推論ツール** - コマンドライン推論システム完成
- **Git管理** - 学習済みモデルとデータセットを含む完全なリポジトリ

### 🚀 すぐに使用可能

このリポジトリをクローンするだけで、以下が即座に利用できます：
- **学習済みモデル**: 100%精度のチワワ vs チクワ分類器
- **推論システム**: コマンドライン画像分類ツール
- **完全なデータセット**: 厳選された50枚の学習用画像
- **前処理・学習・推論の全コード**: 再現可能な実装

### 📊 データセット詳細

- **総画像数**: 50枚（バランス良く各25枚）
- **画像品質**: パッケージ画像等を除外済み
- **ファイル形式**: JPEG
- **著作権**: 研究・教育目的のフェアユース範囲

## ライセンス

MIT License

## 注意点（Windows環境）

- `torch` はGPUなしでもOKですが、`torchvision` バージョンに注意
- `duckduckgo_search` は `duckduckgo-search` でインストール
- 画像収集においてCAPTCHAが出る場合はAPI回数を抑えるかスリープを挟む

## 著作権について

このプロジェクトに含まれる画像は、研究・教育目的のフェアユース（公正利用）の範囲内で使用されています。商用利用は行いません。