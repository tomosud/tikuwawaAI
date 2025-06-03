"""
Phase 3: AIモデル学習のテスト実行
"""

import os
import sys
import torch
from train_model import ChihuawaChikuwaClassifier

def test_phase3():
    """Phase 3のテスト実行"""
    print("=== Phase 3: AIモデル学習のテスト ===")
    
    # 前提条件をチェック
    print("\n1. 前提条件チェック...")
    
    # データ分割ファイルの存在確認
    data_split_path = 'processed_data/data_split.json'
    if not os.path.exists(data_split_path):
        print(f"❌ エラー: {data_split_path} が見つかりません")
        return False
    print(f"✅ データ分割ファイル: {data_split_path}")
    
    # PyTorchの動作確認
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"✅ PyTorch動作確認: デバイス = {device}")
    except Exception as e:
        print(f"❌ PyTorchエラー: {e}")
        return False
    
    # 2. 分類器の初期化テスト
    print("\n2. 分類器初期化テスト...")
    try:
        classifier = ChihuawaChikuwaClassifier()
        print("✅ 分類器初期化成功")
    except Exception as e:
        print(f"❌ 分類器初期化エラー: {e}")
        return False
    
    # 3. モデル作成テスト
    print("\n3. モデル作成テスト...")
    try:
        model = classifier.create_model()
        model_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"✅ ResNet18モデル作成成功")
        print(f"   総パラメータ数: {model_params:,}")
        print(f"   学習可能パラメータ数: {trainable_params:,}")
    except Exception as e:
        print(f"❌ モデル作成エラー: {e}")
        return False
    
    # 4. データロードテスト
    print("\n4. データロードテスト...")
    try:
        train_loader, val_loader, test_loader = classifier.load_data()
        print(f"✅ データロード成功")
        print(f"   学習バッチ数: {len(train_loader)}")
        print(f"   検証バッチ数: {len(val_loader)}")
        print(f"   テストバッチ数: {len(test_loader)}")
        
        # サンプルバッチの確認
        for images, labels in train_loader:
            print(f"   サンプルバッチサイズ: {images.shape}")
            print(f"   ラベルサンプル: {labels.tolist()}")
            break
    except Exception as e:
        print(f"❌ データロードエラー: {e}")
        return False
    
    # 5. 簡単な学習テスト（1エポックのみ）
    print("\n5. 簡単な学習テスト（1エポック）...")
    try:
        # 非常に短い学習を実行
        test_acc = classifier.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=1,
            learning_rate=0.001
        )
        print(f"✅ 学習テスト成功: 検証精度 = {test_acc:.2f}%")
    except Exception as e:
        print(f"❌ 学習テストエラー: {e}")
        return False
    
    # 6. 評価テスト
    print("\n6. 評価テスト...")
    try:
        test_acc, predictions, true_labels = classifier.evaluate(test_loader)
        print(f"✅ 評価テスト成功: テスト精度 = {test_acc:.2f}%")
        print(f"   予測値: {predictions}")
        print(f"   正解値: {true_labels}")
    except Exception as e:
        print(f"❌ 評価テストエラー: {e}")
        return False
    
    print("\n=== Phase 3テスト完了 ===")
    print("✅ すべてのテストが正常に完了しました！")
    print("\n次のステップ:")
    print("1. 本格的な学習を実行: python train_model.py")
    print("2. 学習結果の確認: models/ディレクトリ")
    
    return True

if __name__ == "__main__":
    success = test_phase3()
    if not success:
        sys.exit(1)