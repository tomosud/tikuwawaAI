"""
データ前処理のユーティリティ関数
可視化や分析のための補助機能
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Tuple
import random

import torch
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np


class DataVisualizer:
    """データ可視化クラス"""
    
    def __init__(self, processed_data_dir: str = "processed_data"):
        self.processed_data_dir = Path(processed_data_dir)
        
    def load_split_info(self, filename: str = "data_split.json") -> Dict:
        """保存されたデータ分割情報を読み込み"""
        filepath = self.processed_data_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"データ分割情報が見つかりません: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data
    
    def plot_class_distribution(self, split_info: Dict, save_path: str = None):
        """クラス分布のグラフを作成"""
        try:
            import matplotlib.pyplot as plt
            
            stats = split_info['statistics']
            phases = ['train', 'val', 'test']
            class_names = stats['class_names']
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle('Dataset Class Distribution', fontsize=16)
            
            for i, phase in enumerate(phases):
                if phase in stats['class_distribution']:
                    class_counts = stats['class_distribution'][phase]
                    
                    # 棒グラフ
                    axes[i].bar(class_counts.keys(), class_counts.values())
                    axes[i].set_title(f'{phase.upper()} ({sum(class_counts.values())} samples)')
                    axes[i].set_ylabel('Number of samples')
                    
                    # 値をバーの上に表示
                    for j, (class_name, count) in enumerate(class_counts.items()):
                        axes[i].text(j, count + 0.1, str(count), ha='center')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"クラス分布グラフを保存: {save_path}")
            
            plt.show()
            
        except ImportError:
            print("matplotlib が利用できません。グラフの表示をスキップします。")
            
    def visualize_sample_images(self, split_info: Dict, num_samples: int = 8, phase: str = 'train'):
        """サンプル画像を可視化"""
        try:
            import matplotlib.pyplot as plt
            
            split_data = split_info['split_data'][phase]
            image_paths = split_data['paths']
            labels = split_data['labels']
            class_names = split_info['statistics']['class_names']
            
            # ランダムにサンプルを選択
            indices = random.sample(range(len(image_paths)), min(num_samples, len(image_paths)))
            
            # グリッドのサイズを計算
            cols = 4
            rows = (len(indices) + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
            if rows == 1:
                axes = [axes]
            if cols == 1:
                axes = [[ax] for ax in axes]
            
            fig.suptitle(f'Sample Images - {phase.upper()}', fontsize=16)
            
            for i, idx in enumerate(indices):
                row = i // cols
                col = i % cols
                
                # 画像を読み込み
                image_path = image_paths[idx]
                label = labels[idx]
                class_name = class_names[label]
                
                try:
                    image = Image.open(image_path).convert('RGB')
                    axes[row][col].imshow(image)
                    axes[row][col].set_title(f'{class_name}\n{Path(image_path).name}', fontsize=10)
                    axes[row][col].axis('off')
                except Exception as e:
                    axes[row][col].text(0.5, 0.5, f'Error loading\n{Path(image_path).name}', 
                                       ha='center', va='center', transform=axes[row][col].transAxes)
                    axes[row][col].set_title(f'{class_name} (Error)', fontsize=10)
                    axes[row][col].axis('off')
            
            # 余った軸を非表示
            for i in range(len(indices), rows * cols):
                row = i // cols
                col = i % cols
                axes[row][col].axis('off')
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("matplotlib が利用できません。画像の表示をスキップします。")
        except Exception as e:
            print(f"画像可視化エラー: {e}")
    
    def show_transforms_effect(self, image_path: str, num_samples: int = 6):
        """データ拡張の効果を可視化"""
        try:
            import matplotlib.pyplot as plt
            from data_preprocessing import DataPreprocessor
            
            # 元画像を読み込み
            original_image = Image.open(image_path).convert('RGB')
            
            # データ前処理器からtransformを取得
            preprocessor = DataPreprocessor()
            train_transform = preprocessor.get_transforms('train')
            val_transform = preprocessor.get_transforms('val')
            
            # 図を作成
            fig, axes = plt.subplots(2, num_samples//2 + 1, figsize=(15, 6))
            fig.suptitle(f'Data Augmentation Effects\nSource: {Path(image_path).name}', fontsize=14)
            
            # 元画像を表示
            axes[0][0].imshow(original_image)
            axes[0][0].set_title('Original', fontsize=10)
            axes[0][0].axis('off')
            
            # 検証用transform（拡張なし）
            val_tensor = val_transform(original_image)
            val_image = self.tensor_to_pil(val_tensor)
            axes[1][0].imshow(val_image)
            axes[1][0].set_title('Val Transform\n(Resize + Normalize)', fontsize=10)
            axes[1][0].axis('off')
            
            # 学習用transform（拡張あり）を複数回適用
            for i in range(1, num_samples//2 + 1):
                if i < axes.shape[1]:
                    train_tensor = train_transform(original_image)
                    train_image = self.tensor_to_pil(train_tensor)
                    
                    axes[0][i].imshow(train_image)
                    axes[0][i].set_title(f'Train Transform #{i}', fontsize=10)
                    axes[0][i].axis('off')
                    
                    # 別のサンプル
                    if i < axes.shape[1]:
                        train_tensor2 = train_transform(original_image)
                        train_image2 = self.tensor_to_pil(train_tensor2)
                        
                        axes[1][i].imshow(train_image2)
                        axes[1][i].set_title(f'Train Transform #{i+num_samples//2}', fontsize=10)
                        axes[1][i].axis('off')
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("matplotlib が利用できません。変換効果の表示をスキップします。")
        except Exception as e:
            print(f"変換効果可視化エラー: {e}")
    
    def tensor_to_pil(self, tensor):
        """PyTorchテンソルをPIL画像に変換"""
        # 正規化を逆変換
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        tensor = tensor * std + mean
        tensor = torch.clamp(tensor, 0, 1)
        
        # PIL画像に変換
        to_pil = transforms.ToPILImage()
        return to_pil(tensor)
    
    def analyze_dataset_statistics(self, split_info: Dict):
        """データセット統計の詳細分析"""
        print("=== データセット統計分析 ===")
        
        stats = split_info['statistics']
        
        print(f"総サンプル数: {stats['total_samples']}")
        print(f"クラス数: {len(stats['class_names'])}")
        print(f"クラス名: {stats['class_names']}")
        
        print("\n--- 分割サイズ ---")
        total = stats['total_samples']
        for phase, size in stats['split_sizes'].items():
            percentage = (size / total) * 100
            print(f"{phase:>5}: {size:>3}枚 ({percentage:>5.1f}%)")
        
        print("\n--- クラス分布 ---")
        for phase in ['train', 'val', 'test']:
            if phase in stats['class_distribution']:
                print(f"\n{phase.upper()}:")
                class_dist = stats['class_distribution'][phase]
                phase_total = sum(class_dist.values())
                
                for class_name, count in class_dist.items():
                    percentage = (count / phase_total) * 100 if phase_total > 0 else 0
                    print(f"  {class_name:>10}: {count:>2}枚 ({percentage:>5.1f}%)")
        
        # バランス度の計算
        print("\n--- クラスバランス ---")
        for phase in ['train', 'val', 'test']:
            if phase in stats['class_distribution']:
                class_dist = stats['class_distribution'][phase]
                counts = list(class_dist.values())
                if len(counts) > 1:
                    balance_ratio = min(counts) / max(counts) if max(counts) > 0 else 0
                    print(f"{phase:>5}: バランス度 {balance_ratio:.3f} (1.0が完全バランス)")


def create_sample_visualization():
    """サンプル可視化を実行"""
    try:
        visualizer = DataVisualizer()
        
        # データ分割情報の読み込み
        split_info = visualizer.load_split_info()
        
        # 統計分析
        visualizer.analyze_dataset_statistics(split_info)
        
        # クラス分布グラフ
        print("\nクラス分布グラフを作成中...")
        save_path = visualizer.processed_data_dir / "class_distribution.png"
        visualizer.plot_class_distribution(split_info, str(save_path))
        
        # サンプル画像表示
        print("\nサンプル画像を表示中...")
        visualizer.visualize_sample_images(split_info, num_samples=8, phase='train')
        
        # データ拡張効果の表示
        split_data = split_info['split_data']['train']
        if len(split_data['paths']) > 0:
            sample_image = split_data['paths'][0]
            print(f"\nデータ拡張効果を表示中: {sample_image}")
            visualizer.show_transforms_effect(sample_image)
        
        return True
        
    except Exception as e:
        print(f"可視化エラー: {e}")
        return False


if __name__ == "__main__":
    print("データ可視化を実行しますか？ (y/n)")
    response = input().strip().lower()
    
    if response in ['y', 'yes', '']:
        success = create_sample_visualization()
        if success:
            print("\n✅ 可視化完了！")
        else:
            print("\n❌ 可視化に失敗しました")
    else:
        print("可視化をスキップしました")