#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
フェーズ1のテスト用スクリプト
ライブラリのインポートと基本機能をテストします
"""

import sys
from pathlib import Path

def test_imports():
    """必要なライブラリのインポートテスト"""
    print("=== ライブラリインポートテスト ===")
    
    try:
        import requests
        print("✅ requests - OK")
    except ImportError as e:
        print(f"❌ requests - {e}")
        return False
    
    try:
        from PIL import Image
        print("✅ Pillow - OK")
    except ImportError as e:
        print(f"❌ Pillow - {e}")
        return False
    
    try:
        from duckduckgo_search import DDGS
        print("✅ duckduckgo-search - OK")
    except ImportError as e:
        print(f"❌ duckduckgo-search - {e}")
        return False
    
    return True

def test_directories():
    """ディレクトリ構造のテスト"""
    print("\n=== ディレクトリ構造テスト ===")
    
    dataset_dir = Path("dataset")
    chihuahua_dir = dataset_dir / "chihuahua"
    chikuwa_dir = dataset_dir / "chikuwa"
    
    if dataset_dir.exists():
        print("✅ dataset/ - OK")
    else:
        print("❌ dataset/ - フォルダが存在しません")
        return False
    
    if chihuahua_dir.exists():
        print("✅ dataset/chihuahua/ - OK")
    else:
        print("❌ dataset/chihuahua/ - フォルダが存在しません")
        return False
    
    if chikuwa_dir.exists():
        print("✅ dataset/chikuwa/ - OK")
    else:
        print("❌ dataset/chikuwa/ - フォルダが存在しません")
        return False
    
    return True

def test_download_script():
    """download_images.pyの存在確認"""
    print("\n=== スクリプトファイルテスト ===")
    
    script_path = Path("download_images.py")
    if script_path.exists():
        print("✅ download_images.py - OK")
        return True
    else:
        print("❌ download_images.py - スクリプトが存在しません")
        return False

def main():
    """メインテスト"""
    print("🔍 フェーズ1セットアップテスト開始\n")
    
    # 各テストを実行
    import_ok = test_imports()
    dirs_ok = test_directories()
    script_ok = test_download_script()
    
    print("\n" + "="*50)
    print("📊 テスト結果サマリー")
    print("="*50)
    
    if import_ok and dirs_ok and script_ok:
        print("🎉 すべてのテストが成功しました！")
        print("✅ フェーズ1の実行準備が完了しています。")
        print("\n次のコマンドで画像収集を開始できます:")
        print("python download_images.py")
        return 0
    else:
        print("⚠️  一部のテストが失敗しました。")
        print("上記のエラーを確認して修正してください。")
        return 1

if __name__ == "__main__":
    sys.exit(main())