#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chihuahua vs Chikuwa Classifier - セットアップとテストスクリプト

Gitクローン後に実行して、環境のセットアップとモデルの動作確認を行います。
"""

import subprocess
import sys
import os
from pathlib import Path
import torch

def check_python_version():
    """Python バージョンの確認"""
    print("🐍 Python バージョンチェック...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} (OK)")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} (3.8以上が必要)")
        return False

def install_requirements():
    """依存関係のインストール"""
    print("\n📦 依存関係のインストール...")
    requirements_file = Path("requirements.txt")
    
    if not requirements_file.exists():
        print("❌ requirements.txt が見つかりません")
        return False
    
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, capture_output=True, text=True)
        print("✅ 依存関係のインストール完了")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ インストールエラー: {e}")
        print(f"エラー出力: {e.stderr}")
        return False

def check_pytorch():
    """PyTorch の動作確認"""
    print("\n🔥 PyTorch 動作確認...")
    try:
        import torch
        import torchvision
        print(f"✅ PyTorch {torch.__version__}")
        print(f"✅ TorchVision {torchvision.__version__}")
        
        # CUDA確認
        if torch.cuda.is_available():
            print(f"🚀 CUDA利用可能: {torch.cuda.get_device_name(0)}")
        else:
            print("💻 CPU実行モード")
        return True
    except ImportError as e:
        print(f"❌ PyTorchインポートエラー: {e}")
        return False

def check_dataset():
    """データセットの確認"""
    print("\n📸 データセット確認...")
    dataset_path = Path("dataset")
    
    if not dataset_path.exists():
        print("❌ datasetフォルダが見つかりません")
        return False
    
    chihuahua_path = dataset_path / "chihuahua"
    chikuwa_path = dataset_path / "chikuwa"
    
    chihuahua_count = len(list(chihuahua_path.glob("*.jpg"))) if chihuahua_path.exists() else 0
    chikuwa_count = len(list(chikuwa_path.glob("*.jpg"))) if chikuwa_path.exists() else 0
    
    print(f"📷 チワワ画像: {chihuahua_count}枚")
    print(f"🥢 チクワ画像: {chikuwa_count}枚")
    
    if chihuahua_count > 0 and chikuwa_count > 0:
        print("✅ データセット確認完了")
        return True
    else:
        print("❌ データセットが不完全です")
        return False

def check_model():
    """学習済みモデルの確認"""
    print("\n🤖 学習済みモデル確認...")
    model_path = Path("models/chihuawa_chikuwa_classifier.pt")
    
    if not model_path.exists():
        print("❌ 学習済みモデルが見つかりません")
        print("💡 モデルを学習するには: python train_model.py")
        return False
    
    try:
        # モデルサイズ確認
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"📦 モデルサイズ: {size_mb:.2f} MB")
        
        # モデル読み込みテスト
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(model_path, map_location=device)
        print("✅ モデル読み込み成功")
        return True
        
    except Exception as e:
        print(f"❌ モデル読み込みエラー: {e}")
        return False

def test_inference():
    """推論テスト"""
    print("\n🔍 推論テスト...")
    
    # テスト用画像を探す
    test_images = []
    dataset_path = Path("dataset")
    
    for category in ["chihuahua", "chikuwa"]:
        category_path = dataset_path / category
        if category_path.exists():
            images = list(category_path.glob("*.jpg"))
            if images:
                test_images.append((category, images[0]))
    
    if not test_images:
        print("❌ テスト用画像が見つかりません")
        return False
    
    try:
        from model_inference import ChihuahuaChikuwaClassifier
        
        classifier = ChihuahuaChikuwaClassifier()
        
        for expected_class, image_path in test_images:
            prediction, probabilities = classifier.predict(image_path, return_probabilities=True)
            confidence = probabilities[prediction]
            
            print(f"📸 {image_path.name}")
            print(f"   期待値: {expected_class}")
            print(f"   予測値: {prediction}")
            print(f"   信頼度: {confidence:.4f} ({confidence*100:.2f}%)")
            
            if prediction == expected_class:
                print("   ✅ 正解")
            else:
                print("   ❌ 不正解")
        
        print("✅ 推論テスト完了")
        return True
        
    except Exception as e:
        print(f"❌ 推論テストエラー: {e}")
        return False

def print_usage_guide():
    """使用方法ガイドの表示"""
    print("\n" + "="*60)
    print("🎉 セットアップ完了！使用方法ガイド")
    print("="*60)
    print()
    print("📷 単一画像の分類:")
    print("   python model_inference.py dataset/chihuahua/chihuahua_001.jpg")
    print("   python model_inference.py dataset/chikuwa/chikuwa_001.jpg --detailed")
    print()
    print("🔬 データ前処理の実行:")
    print("   python data_preprocessing.py")
    print()
    print("🏋️ モデルの再学習:")
    print("   python train_model.py")
    print()
    print("🧪 各フェーズのテスト:")
    print("   python test_phase2.py  # データ前処理テスト")
    print("   python test_phase3.py  # モデル学習テスト")
    print()
    print("📊 データ分析と可視化:")
    print("   python data_utils.py")
    print()

def main():
    """メイン実行"""
    print("🚀 Chihuahua vs Chikuwa Classifier - セットアップとテスト")
    print("="*60)
    
    all_ok = True
    
    # 各チェックを実行
    checks = [
        ("Python バージョン", check_python_version),
        ("依存関係インストール", install_requirements),
        ("PyTorch 動作確認", check_pytorch),
        ("データセット", check_dataset),
        ("学習済みモデル", check_model),
        ("推論テスト", test_inference)
    ]
    
    for check_name, check_func in checks:
        try:
            if not check_func():
                all_ok = False
                break
        except Exception as e:
            print(f"❌ {check_name}でエラー: {e}")
            all_ok = False
            break
    
    print("\n" + "="*60)
    if all_ok:
        print("🎉 すべてのチェックが完了しました！")
        print_usage_guide()
    else:
        print("❌ セットアップに問題があります。上記のエラーを確認してください。")
        print("\n💡 トラブルシューティング:")
        print("1. Python 3.8以上がインストールされているか確認")
        print("2. pip install -r requirements.txt を手動実行")
        print("3. モデルが存在しない場合は python train_model.py を実行")

if __name__ == "__main__":
    main()