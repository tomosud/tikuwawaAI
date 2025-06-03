#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chihuahua vs Chikuwa Classifier - 推論用スクリプト

学習済みモデルを使用してチワワとチクワの判定を行います。
Gitクローン後にすぐに使用できるように設計されています。
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import argparse
import json
from pathlib import Path
import sys

class ChihuahuaChikuwaClassifier:
    """チワワ vs チクワ分類器"""
    
    def __init__(self, model_path="models/chihuawa_chikuwa_classifier.pt"):
        """
        分類器の初期化
        
        Args:
            model_path (str): 学習済みモデルのパス
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = ["chihuahua", "chikuwa"]
        
        # モデルの構築
        self.model = self._build_model()
        
        # 学習済みモデルの読み込み
        self._load_model(model_path)
        
        # 画像前処理の定義
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _build_model(self):
        """ResNet18ベースのモデル構築"""
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 2)  # 2クラス分類
        return model.to(self.device)
    
    def _load_model(self, model_path):
        """学習済みモデルの読み込み"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model.eval()
            print(f"✅ モデルを読み込みました: {model_path}")
            
        except FileNotFoundError:
            print(f"❌ モデルファイルが見つかりません: {model_path}")
            print("学習を実行してモデルを作成してください: python train_model.py")
            sys.exit(1)
        except Exception as e:
            print(f"❌ モデル読み込みエラー: {e}")
            sys.exit(1)
    
    def predict(self, image_path, return_probabilities=False):
        """
        画像の分類予測
        
        Args:
            image_path (str): 予測する画像のパス
            return_probabilities (bool): 確率も返すかどうか
            
        Returns:
            str or tuple: クラス名、または(クラス名, 確率辞書)
        """
        try:
            # 画像の読み込みと前処理
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # 予測実行
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_idx = torch.argmax(probabilities, dim=1).item()
                predicted_class = self.class_names[predicted_idx]
                confidence = probabilities[0][predicted_idx].item()
            
            if return_probabilities:
                prob_dict = {
                    class_name: probabilities[0][i].item() 
                    for i, class_name in enumerate(self.class_names)
                }
                return predicted_class, prob_dict
            else:
                return predicted_class
                
        except Exception as e:
            print(f"❌ 予測エラー: {e}")
            return None
    
    def batch_predict(self, image_paths):
        """
        複数画像の一括予測
        
        Args:
            image_paths (list): 画像パスのリスト
            
        Returns:
            list: 予測結果のリスト
        """
        results = []
        for image_path in image_paths:
            prediction, probabilities = self.predict(image_path, return_probabilities=True)
            results.append({
                'image_path': str(image_path),
                'prediction': prediction,
                'probabilities': probabilities
            })
        return results

def main():
    """メイン関数 - コマンドライン実行用"""
    parser = argparse.ArgumentParser(description='チワワ vs チクワ分類器')
    parser.add_argument('image_path', help='分類する画像のパス')
    parser.add_argument('--model', default='models/chihuawa_chikuwa_classifier.pt',
                        help='学習済みモデルのパス')
    parser.add_argument('--detailed', action='store_true',
                        help='詳細な確率も表示')
    
    args = parser.parse_args()
    
    # 分類器の初期化
    classifier = ChihuahuaChikuwaClassifier(args.model)
    
    # 予測実行
    if args.detailed:
        prediction, probabilities = classifier.predict(args.image_path, return_probabilities=True)
        print(f"\n📷 画像: {args.image_path}")
        print(f"🎯 予測結果: {prediction}")
        print(f"📊 確率:")
        for class_name, prob in probabilities.items():
            print(f"  {class_name}: {prob:.4f} ({prob*100:.2f}%)")
    else:
        prediction = classifier.predict(args.image_path)
        print(f"予測結果: {prediction}")

if __name__ == "__main__":
    # 使用例の表示
    if len(sys.argv) == 1:
        print("=" * 50)
        print("🐕 Chihuahua vs Chikuwa Classifier 🥢")
        print("=" * 50)
        print()
        print("使用方法:")
        print("  python model_inference.py <画像パス>")
        print()
        print("例:")
        print("  python model_inference.py dataset/chihuahua/chihuahua_001.jpg")
        print("  python model_inference.py dataset/chikuwa/chikuwa_001.jpg --detailed")
        print()
        print("オプション:")
        print("  --detailed  : 詳細な確率も表示")
        print("  --model     : 使用するモデルファイルを指定")
        print()
    else:
        main()