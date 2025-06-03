# 🐕 チワワ vs チクワ AI判定器 🥢

ブラウザ内でリアルタイムにチワワとチクワを判定するWebアプリケーションです。
ONNX.jsを使用してクライアントサイドでAI推論を実行します。

## 🚀 特徴

- **ブラウザ内AI推論**: サーバー不要でプライベート
- **リアルタイム判定**: 画像アップロード後即座に結果表示
- **静的サイト**: GitHub Pagesで簡単にホスティング可能
- **レスポンシブデザイン**: モバイル対応
- **ドラッグ&ドロップ対応**: 直感的な操作

## 🌐 使用方法

### ローカルでの実行

1. このディレクトリをWebサーバーで配信:
   ```bash
   # Python 3の場合
   python -m http.server 8000
   
   # Node.jsの場合
   npx serve .
   ```

2. ブラウザで `http://localhost:8000` にアクセス

### GitHub Pagesでの配信

1. GitHubリポジトリの Settings > Pages で配信設定
2. Source を "Deploy from a branch" に設定
3. Branch を `main` (または適切なブランチ) に設定
4. フォルダを `/web_app` に設定

## 📁 ファイル構成

```
web_app/
├── index.html                          # メインHTML
├── styles.css                          # スタイルシート
├── app.js                             # JavaScript (ONNX.js使用)
├── chihuawa_chikuwa_classifier.onnx   # ONNXモデル
└── README.md                          # このファイル
```

## 🎯 判定について

- **チワワ**: 犬の画像として判定
- **チクワ**: 食品の画像として判定
- 信頼度が高いほど確信度が高い結果
- ResNet18ベースのディープラーニングモデルを使用

## 🔧 技術仕様

- **フロントエンド**: HTML5, CSS3, Vanilla JavaScript
- **AI推論**: ONNX Runtime Web
- **モデル**: PyTorch → ONNX変換済みResNet18
- **画像処理**: Canvas API
- **対応ブラウザ**: Chrome, Firefox, Safari, Edge (モダンブラウザ)

## 🖼️ 対応画像形式

- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- WebP (.webp)
- その他のCanvas APIで対応している形式

## 📱 レスポンシブ対応

- デスクトップ: フル機能
- タブレット: 最適化されたレイアウト
- スマートフォン: タッチ操作対応

## 🛠️ 開発情報

### モデル情報
- **アーキテクチャ**: ResNet18
- **入力サイズ**: 224×224×3 (RGB)
- **出力**: 2クラス (chihuahua, chikuwa)
- **前処理**: ImageNet標準化

### パフォーマンス
- **モデルサイズ**: 約45MB
- **初回読み込み**: 2-5秒 (ネットワーク速度依存)
- **推論時間**: 100-500ms (デバイス性能依存)

## 🔒 プライバシー

- すべての処理がブラウザ内で完結
- 画像データはサーバーに送信されません
- 完全にオフラインで動作可能 (初回読み込み後)

## 📄 ライセンス

このプロジェクトは元のプロジェクトと同じライセンスの下で配布されています。