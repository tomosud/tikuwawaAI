import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import json
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import pickle


class ChihuawaChikuwaDataset(Dataset):
    """チワワ vs チクワ分類用のカスタムデータセット"""
    
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        # 画像パスを正規化（バックスラッシュをスラッシュに変換）
        image_path = self.paths[idx].replace('\\', '/')
        
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"エラー: 画像を読み込めませんでした {image_path}: {e}")
            # デフォルト画像を作成
            image = Image.new('RGB', (224, 224), color=(128, 128, 128))
        
        if self.transform:
            image = self.transform(image)
        
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label


class ChihuawaChikuwaClassifier:
    """チワワ vs チクワ分類器のメインクラス"""
    
    def __init__(self, num_classes=2, device=None):
        self.num_classes = num_classes
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        print(f"使用デバイス: {self.device}")
    
    def create_model(self):
        """ResNet18ベースの転移学習モデルを作成"""
        # 事前学習済みResNet18をロード
        self.model = models.resnet18(pretrained=True)
        
        # 特徴抽出層を凍結（転移学習）
        for param in self.model.parameters():
            param.requires_grad = False
        
        # 最終分類層のみ学習可能に設定
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, self.num_classes)
        
        # モデルをデバイスに移動
        self.model = self.model.to(self.device)
        
        print(f"ResNet18ベースモデルを作成しました (出力クラス数: {self.num_classes})")
        return self.model
    
    def create_data_transforms(self):
        """データ変換を定義"""
        # ImageNet正規化パラメータ
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        # 学習用変換（データ拡張あり）
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            normalize
        ])
        
        # 検証・テスト用変換（データ拡張なし）
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize
        ])
        
        return train_transform, val_transform
    
    def load_data(self, data_split_path='processed_data/data_split.json'):
        """データ分割ファイルからデータをロード"""
        with open(data_split_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        split_data = data['split_data']
        
        # データ変換を作成
        train_transform, val_transform = self.create_data_transforms()
        
        # データセットを作成
        train_dataset = ChihuawaChikuwaDataset(
            split_data['train']['paths'],
            split_data['train']['labels'],
            transform=train_transform
        )
        
        val_dataset = ChihuawaChikuwaDataset(
            split_data['val']['paths'],
            split_data['val']['labels'],
            transform=val_transform
        )
        
        test_dataset = ChihuawaChikuwaDataset(
            split_data['test']['paths'],
            split_data['test']['labels'],
            transform=val_transform
        )
        
        # DataLoaderを作成
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)
        
        print(f"データ読み込み完了:")
        print(f"  学習データ: {len(train_dataset)}枚")
        print(f"  検証データ: {len(val_dataset)}枚")
        print(f"  テストデータ: {len(test_dataset)}枚")
        
        return train_loader, val_loader, test_loader
    
    def train_epoch(self, train_loader, criterion, optimizer):
        """1エポックの学習を実行"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # 勾配をリセット
            optimizer.zero_grad()
            
            # 順伝播
            outputs = self.model(images)
            loss = criterion(outputs, labels)
            
            # 逆伝播
            loss.backward()
            optimizer.step()
            
            # 統計情報を更新
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, val_loader, criterion):
        """1エポックの検証を実行"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self, train_loader, val_loader, num_epochs=20, learning_rate=0.001):
        """メインの学習ループ"""
        print(f"\n学習を開始します...")
        print(f"エポック数: {num_epochs}")
        print(f"学習率: {learning_rate}")
        print("-" * 50)
        
        # 損失関数と最適化器を定義
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.fc.parameters(), lr=learning_rate)
        
        best_val_acc = 0.0
        best_model_state = None
        
        for epoch in range(num_epochs):
            # 学習
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer)
            
            # 検証
            val_loss, val_acc = self.validate_epoch(val_loader, criterion)
            
            # 履歴を記録
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # 最良モデルを保存
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = self.model.state_dict().copy()
            
            # 学習状況を表示
            print(f'Epoch [{epoch+1:2d}/{num_epochs}] '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # 最良モデルをロード
        if best_model_state:
            self.model.load_state_dict(best_model_state)
        
        print(f"\n学習完了! 最高検証精度: {best_val_acc:.2f}%")
        return best_val_acc
    
    def evaluate(self, test_loader, class_names=['chihuahua', 'chikuwa']):
        """テストデータで最終評価"""
        self.model.eval()
        correct = 0
        total = 0
        class_correct = list(0. for i in range(len(class_names)))
        class_total = list(0. for i in range(len(class_names)))
        
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # クラス別精度を計算
                c = (predicted == labels).squeeze()
                for i in range(labels.size(0)):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
                
                predictions.extend(predicted.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        overall_acc = 100 * correct / total
        
        print(f"\n=== テストデータでの最終評価 ===")
        print(f"全体精度: {overall_acc:.2f}%")
        print("\nクラス別精度:")
        
        for i in range(len(class_names)):
            if class_total[i] > 0:
                acc = 100 * class_correct[i] / class_total[i]
                print(f"  {class_names[i]}: {acc:.2f}% ({int(class_correct[i])}/{int(class_total[i])})")
        
        return overall_acc, predictions, true_labels
    
    def plot_training_history(self, save_path='models/training_history.png'):
        """学習履歴をプロット"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # 損失のプロット
        ax1.plot(self.history['train_loss'], label='Train Loss', color='blue')
        ax1.plot(self.history['val_loss'], label='Val Loss', color='orange')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # 精度のプロット
        ax2.plot(self.history['train_acc'], label='Train Accuracy', color='blue')
        ax2.plot(self.history['val_acc'], label='Val Accuracy', color='orange')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        # ディレクトリを作成
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"学習履歴グラフを保存しました: {save_path}")
    
    def save_model(self, save_path='models/chihuawa_chikuwa_classifier.pt'):
        """モデルを保存"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # モデルの状態とメタデータを保存
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'num_classes': self.num_classes,
            'history': self.history,
            'timestamp': datetime.now().isoformat()
        }
        
        torch.save(save_dict, save_path)
        print(f"モデルを保存しました: {save_path}")
    
    def load_model(self, load_path='models/chihuawa_chikuwa_classifier.pt'):
        """モデルを読み込み"""
        checkpoint = torch.load(load_path, map_location=self.device)
        
        if self.model is None:
            self.create_model()
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.history = checkpoint.get('history', self.history)
        
        print(f"モデルを読み込みました: {load_path}")


def main():
    """メイン関数"""
    print("=== チワワ vs チクワ分類器 Phase 3: AIモデル学習 ===")
    
    # 分類器を初期化
    classifier = ChihuawaChikuwaClassifier()
    
    # モデルを作成
    model = classifier.create_model()
    
    # データを読み込み
    train_loader, val_loader, test_loader = classifier.load_data()
    
    # 学習を実行
    best_val_acc = classifier.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=25,
        learning_rate=0.001
    )
    
    # テストデータで最終評価
    test_acc, predictions, true_labels = classifier.evaluate(test_loader)
    
    # 学習履歴をプロット
    classifier.plot_training_history()
    
    # モデルを保存
    classifier.save_model()
    
    # 結果サマリーを表示
    print(f"\n=== 学習結果サマリー ===")
    print(f"最高検証精度: {best_val_acc:.2f}%")
    print(f"最終テスト精度: {test_acc:.2f}%")
    
    if test_acc >= 90.0:
        print("🎉 目標精度90%を達成しました！")
    else:
        print("📊 さらなる改善が必要です")
    
    print(f"\nモデルファイル: models/chihuawa_chikuwa_classifier.pt")
    print(f"学習履歴グラフ: models/training_history.png")


if __name__ == "__main__":
    main()