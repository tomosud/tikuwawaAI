"""
Phase 2のテストスクリプト
データ前処理の動作確認
"""

import os
import sys
from pathlib import Path
import torch
from data_preprocessing import DataPreprocessor, test_dataloader


def main():
    print("=== Phase 2: データ前処理テスト ===")
    
    # 1. 環境情報を表示
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # 2. データセットの存在確認
    dataset_dir = Path("dataset")
    print(f"\nデータセットディレクトリ: {dataset_dir.absolute()}")
    
    chihuahua_dir = dataset_dir / "chihuahua"
    chikuwa_dir = dataset_dir / "chikuwa"
    
    if not chihuahua_dir.exists():
        print(f"エラー: {chihuahua_dir} が存在しません")
        return False
    
    if not chikuwa_dir.exists():
        print(f"エラー: {chikuwa_dir} が存在しません")
        return False
    
    # 画像数を確認
    chihuahua_images = list(chihuahua_dir.glob("*.jpg")) + list(chihuahua_dir.glob("*.jpeg"))
    chikuwa_images = list(chikuwa_dir.glob("*.jpg")) + list(chikuwa_dir.glob("*.jpeg"))
    
    print(f"チワワ画像: {len(chihuahua_images)}枚")
    print(f"チクワ画像: {len(chikuwa_images)}枚")
    
    if len(chihuahua_images) == 0 or len(chikuwa_images) == 0:
        print("エラー: 画像が不足しています")
        return False
    
    # 3. データ前処理の実行
    try:
        print("\n前処理を開始します...")
        preprocessor = DataPreprocessor()
        
        # 小さなバッチサイズでテスト
        dataloaders, stats = preprocessor.process_all(batch_size=4)
        
        # 4. DataLoaderのテスト
        print("\n=== DataLoaderテスト ===")
        
        for phase in ['train', 'val', 'test']:
            if phase in dataloaders and len(dataloaders[phase]) > 0:
                print(f"\n{phase.upper()} DataLoader:")
                print(f"  バッチ数: {len(dataloaders[phase])}")
                
                # 最初のバッチを取得
                try:
                    images, labels = next(iter(dataloaders[phase]))
                    print(f"  画像テンソル形状: {images.shape}")
                    print(f"  ラベル形状: {labels.shape}")
                    print(f"  画像値の範囲: [{images.min():.3f}, {images.max():.3f}]")
                    
                    # ラベルの分布を確認
                    unique_labels = labels.unique()
                    print(f"  このバッチのラベル: {labels.tolist()}")
                    print(f"  ユニークラベル: {unique_labels.tolist()}")
                    
                except Exception as e:
                    print(f"  バッチ取得エラー: {e}")
        
        # 5. 保存されたファイルの確認
        processed_dir = Path("processed_data")
        if processed_dir.exists():
            print(f"\n処理済みデータディレクトリ: {processed_dir.absolute()}")
            files = list(processed_dir.glob("*"))
            print(f"保存されたファイル: {[f.name for f in files]}")
        
        print("\n=== テスト完了 ===")
        print("Phase 2のデータ前処理が正常に動作しています！")
        return True
        
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return False


def quick_test():
    """簡単なテスト"""
    print("=== 簡単なテスト ===")
    
    # データ前処理器の初期化
    preprocessor = DataPreprocessor()
    
    # 画像パスとラベルの取得
    try:
        image_paths, labels = preprocessor.get_image_paths_and_labels()
        print(f"取得した画像数: {len(image_paths)}")
        print(f"ラベル数: {len(labels)}")
        
        if len(image_paths) > 0:
            print(f"最初の画像: {image_paths[0]}")
            print(f"最初のラベル: {labels[0]}")
        
        # クラス分布
        from collections import Counter
        label_counts = Counter(labels)
        print(f"クラス分布: {dict(label_counts)}")
        
        return True
        
    except Exception as e:
        print(f"エラー: {e}")
        return False


if __name__ == "__main__":
    print("どちらのテストを実行しますか？")
    print("1. 簡単なテスト (quick)")
    print("2. 完全なテスト (full)")
    
    choice = input("選択してください (1/2) または Enter で完全なテスト: ").strip()
    
    if choice == "1" or choice.lower() == "quick":
        success = quick_test()
    else:
        success = main()
    
    if success:
        print("\n✅ テスト成功！")
    else:
        print("\n❌ テスト失敗")
        sys.exit(1)