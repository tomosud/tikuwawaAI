"""
Phase 2: データ前処理
チワワ vs チクワ画像分類のための前処理スクリプト
"""

import os
import shutil
from pathlib import Path
import random
from typing import Tuple, List, Dict
import json

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np


class ChihuahuaChikuwaDataset(Dataset):
    """チワワ vs チクワのカスタムデータセット"""
    
    def __init__(self, image_paths: List[str], labels: List[int], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # 画像の読み込み
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"画像読み込みエラー: {image_path}, エラー: {e}")
            # エラーの場合は黒い画像を返す
            image = Image.new('RGB', (224, 224), color='black')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


class DataPreprocessor:
    """データ前処理クラス"""
    
    def __init__(self, dataset_dir: str = "dataset", output_dir: str = "processed_data"):
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        self.class_names = ['chihuahua', 'chikuwa']
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        
        # 出力ディレクトリの作成
        self.output_dir.mkdir(exist_ok=True)
        
    def get_image_paths_and_labels(self) -> Tuple[List[str], List[int]]:
        """画像パスとラベルのリストを取得"""
        image_paths = []
        labels = []
        
        for class_name in self.class_names:
            class_dir = self.dataset_dir / class_name
            if not class_dir.exists():
                print(f"警告: {class_dir} が存在しません")
                continue
                
            # 画像ファイルを取得
            image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg")) + list(class_dir.glob("*.png"))
            
            print(f"{class_name}: {len(image_files)}枚の画像を発見")
            
            for image_file in image_files:
                image_paths.append(str(image_file))
                labels.append(self.class_to_idx[class_name])
        
        return image_paths, labels
    
    def split_data(self, image_paths: List[str], labels: List[int], 
                   train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1) -> Dict:
        """データを学習・検証・テストに分割"""
        # データをシャッフル
        combined = list(zip(image_paths, labels))
        random.shuffle(combined)
        image_paths, labels = zip(*combined)
        
        total_samples = len(image_paths)
        train_size = int(total_samples * train_ratio)
        val_size = int(total_samples * val_ratio)
        
        # 分割
        train_paths = list(image_paths[:train_size])
        train_labels = list(labels[:train_size])
        
        val_paths = list(image_paths[train_size:train_size + val_size])
        val_labels = list(labels[train_size:train_size + val_size])
        
        test_paths = list(image_paths[train_size + val_size:])
        test_labels = list(labels[train_size + val_size:])
        
        split_info = {
            'train': {'paths': train_paths, 'labels': train_labels},
            'val': {'paths': val_paths, 'labels': val_labels},
            'test': {'paths': test_paths, 'labels': test_labels}
        }
        
        print(f"データ分割完了:")
        print(f"  学習データ: {len(train_paths)}枚")
        print(f"  検証データ: {len(val_paths)}枚") 
        print(f"  テストデータ: {len(test_paths)}枚")
        
        return split_info
    
    def get_transforms(self, phase: str = 'train') -> transforms.Compose:
        """データ変換の定義"""
        if phase == 'train':
            # 学習時：データ拡張を含む
            return transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            # 検証・テスト時：データ拡張なし
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def create_dataloaders(self, split_data: Dict, batch_size: int = 16, num_workers: int = 2) -> Dict[str, DataLoader]:
        """DataLoaderを作成"""
        dataloaders = {}
        
        for phase in ['train', 'val', 'test']:
            # データセットの作成
            transform = self.get_transforms(phase)
            dataset = ChihuahuaChikuwaDataset(
                split_data[phase]['paths'],
                split_data[phase]['labels'],
                transform=transform
            )
            
            # DataLoaderの作成
            shuffle = (phase == 'train')
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=True if torch.cuda.is_available() else False
            )
            
            dataloaders[phase] = dataloader
        
        return dataloaders
    
    def save_split_info(self, split_data: Dict, filename: str = "data_split.json"):
        """データ分割情報を保存"""
        save_path = self.output_dir / filename
        
        # データ統計を追加
        stats = {
            'class_names': self.class_names,
            'class_to_idx': self.class_to_idx,
            'total_samples': sum(len(split_data[phase]['paths']) for phase in split_data),
            'split_sizes': {phase: len(split_data[phase]['paths']) for phase in split_data},
            'class_distribution': {}
        }
        
        # クラス別データ数を計算
        for phase in split_data:
            phase_labels = split_data[phase]['labels']
            class_counts = {class_name: phase_labels.count(idx) 
                          for class_name, idx in self.class_to_idx.items()}
            stats['class_distribution'][phase] = class_counts
        
        # 分割データと統計情報を保存
        save_data = {
            'split_data': split_data,
            'statistics': stats
        }
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        
        print(f"データ分割情報を保存しました: {save_path}")
        return stats
    
    def validate_images(self, image_paths: List[str]) -> List[str]:
        """画像ファイルの有効性を検証"""
        valid_paths = []
        invalid_count = 0
        
        for path in image_paths:
            try:
                with Image.open(path) as img:
                    img.verify()  # 画像の整合性チェック
                valid_paths.append(path)
            except Exception as e:
                print(f"無効な画像: {path}, エラー: {e}")
                invalid_count += 1
        
        print(f"画像検証完了: {len(valid_paths)}枚有効, {invalid_count}枚無効")
        return valid_paths
    
    def process_all(self, batch_size: int = 16, validate_images: bool = True):
        """全体的な前処理を実行"""
        print("=== Phase 2: データ前処理開始 ===")
        
        # 1. 画像パスとラベルを取得
        image_paths, labels = self.get_image_paths_and_labels()
        
        # 2. 画像の有効性を検証（オプション）
        if validate_images:
            print("\n画像ファイルの検証中...")
            valid_paths = self.validate_images(image_paths)
            valid_labels = [labels[i] for i, path in enumerate(image_paths) if path in valid_paths]
            image_paths, labels = valid_paths, valid_labels
        
        if len(image_paths) == 0:
            raise ValueError("有効な画像が見つかりませんでした")
        
        # 3. データ分割
        print("\nデータ分割中...")
        split_data = self.split_data(image_paths, labels)
        
        # 4. データ分割情報を保存
        print("\nデータ分割情報を保存中...")
        stats = self.save_split_info(split_data)
        
        # 5. DataLoaderを作成
        print("\nDataLoader作成中...")
        dataloaders = self.create_dataloaders(split_data, batch_size=batch_size)
        
        # 6. 統計情報を表示
        print("\n=== データセット統計 ===")
        print(f"総サンプル数: {stats['total_samples']}")
        print(f"クラス: {stats['class_names']}")
        
        for phase in ['train', 'val', 'test']:
            print(f"\n{phase.upper()}:")
            print(f"  総数: {stats['split_sizes'][phase]}")
            for class_name, count in stats['class_distribution'][phase].items():
                print(f"  {class_name}: {count}枚")
        
        print("\n=== Phase 2: データ前処理完了 ===")
        
        return dataloaders, stats


def test_dataloader(dataloader: DataLoader, phase: str = "train"):
    """DataLoaderのテスト"""
    print(f"\n=== {phase.upper()} DataLoaderテスト ===")
    
    # 最初のバッチを取得
    for batch_idx, (images, labels) in enumerate(dataloader):
        print(f"バッチ {batch_idx + 1}:")
        print(f"  画像テンソルサイズ: {images.shape}")
        print(f"  ラベル: {labels}")
        print(f"  バッチサイズ: {len(labels)}")
        
        # 画像テンソルの統計
        print(f"  画像値の範囲: [{images.min():.3f}, {images.max():.3f}]")
        print(f"  画像の平均: {images.mean():.3f}")
        print(f"  画像の標準偏差: {images.std():.3f}")
        
        if batch_idx >= 2:  # 最初の3バッチのみテスト
            break


if __name__ == "__main__":
    # 乱数シードを設定（再現性のため）
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    
    # データ前処理器を作成
    preprocessor = DataPreprocessor()
    
    try:
        # 前処理を実行
        dataloaders, stats = preprocessor.process_all(batch_size=8)
        
        # DataLoaderのテスト
        for phase in ['train', 'val', 'test']:
            if phase in dataloaders:
                test_dataloader(dataloaders[phase], phase)
        
        print("\n前処理が正常に完了しました！")
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        raise