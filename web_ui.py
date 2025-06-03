#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chihuahua vs Chikuwa Classifier - Web UI

クリップボードから画像を取得して、チワワかチクワかを判定するWebアプリケーション
"""

import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageGrab
import io
import base64
import numpy as np
from pathlib import Path
import sys
import traceback

class ChihuahuaChikuwaWebClassifier:
    """チワワ vs チクワ分類器 (Web UI版)"""
    
    def __init__(self, model_path="models/chihuawa_chikuwa_classifier.pt"):
        """分類器の初期化"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = ["chihuahua", "chikuwa"]
        self.class_names_jp = ["チワワ", "チクワ"]
        
        # モデルの構築と読み込み
        self.model = self._build_and_load_model(model_path)
        
        # 画像前処理の定義
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _build_and_load_model(self, model_path):
        """モデル構築と読み込み"""
        try:
            # モデル構築
            model = models.resnet18(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, 2)  # 2クラス分類
            model = model.to(self.device)
            
            # 学習済みモデルの読み込み
            if not Path(model_path).exists():
                raise FileNotFoundError(f"モデルファイルが見つかりません: {model_path}")
            
            checkpoint = torch.load(model_path, map_location=self.device)
            
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model.eval()
            print(f"✅ モデルを読み込みました: {model_path}")
            return model
            
        except Exception as e:
            print(f"❌ モデル読み込みエラー: {e}")
            raise e
    
    def predict_image(self, image):
        """画像の分類予測"""
        try:
            if image is None:
                return "❌ 画像がありません", 0.0, "画像をアップロードまたはクリップボードから取得してください"
            
            # PIL画像に変換
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            elif not isinstance(image, Image.Image):
                return "❌ 不正な画像形式", 0.0, "対応していない画像形式です"
            
            # RGB変換
            image = image.convert('RGB')
            
            # 前処理
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # 予測実行
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_idx = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_idx].item()
                
                # 日本語クラス名
                predicted_class_jp = self.class_names_jp[predicted_idx]
                predicted_class_en = self.class_names[predicted_idx]
                
                # 結果メッセージ（HTMLで巨大フォント）
                result_text = f'<div style="font-size: 80px; font-weight: 900; text-align: center; color: #ff1744; text-shadow: 4px 4px 8px rgba(0,0,0,0.5); padding: 30px; background: linear-gradient(135deg, #ffebee, #ffcdd2); border-radius: 20px; border: 5px solid #e91e63; margin: 20px 0;">🎯 判定結果: {predicted_class_jp}</div>'
                confidence_text = f"信頼度: {confidence:.2%}"
                
                # 詳細情報
                detail_text = f"""
📊 詳細結果:
• チワワ: {probabilities[0][0].item():.2%}
• チクワ: {probabilities[0][1].item():.2%}

🔍 予測: {predicted_class_jp} ({predicted_class_en})
📈 信頼度: {confidence:.4f} ({confidence:.2%})
"""
                
                return result_text, confidence, detail_text.strip()
                
        except Exception as e:
            error_msg = f"❌ 予測エラー: {str(e)}"
            print(error_msg)
            print(traceback.format_exc())
            return error_msg, 0.0, "エラーが発生しました。画像形式を確認してください。"
    
    def get_clipboard_image(self):
        """クリップボードから画像を取得"""
        try:
            # クリップボードから画像を取得
            clipboard_image = ImageGrab.grabclipboard()
            
            if clipboard_image is None:
                return None, "❌ クリップボードに画像がありません"
            
            # PIL画像として返す
            if isinstance(clipboard_image, Image.Image):
                return clipboard_image, "✅ クリップボードから画像を取得しました"
            else:
                return None, "❌ クリップボードのデータが画像ではありません"
                
        except Exception as e:
            error_msg = f"❌ クリップボード取得エラー: {str(e)}"
            print(error_msg)
            return None, error_msg

def create_web_ui():
    """GradioによるWeb UIを作成"""
    
    # 分類器の初期化
    try:
        classifier = ChihuahuaChikuwaWebClassifier()
    except Exception as e:
        print(f"❌ 分類器の初期化に失敗: {e}")
        print("学習済みモデルが存在しない可能性があります。")
        print("python train_model.py を実行してモデルを作成してください。")
        sys.exit(1)
    
    def classify_uploaded_image(image):
        """アップロード画像の分類"""
        result, confidence, detail = classifier.predict_image(image)
        return result, detail
    
    def classify_clipboard_image():
        """クリップボード画像の分類"""
        clipboard_img, status_msg = classifier.get_clipboard_image()
        
        if clipboard_img is None:
            return None, status_msg, "クリップボードから画像を取得できませんでした"
        
        result, confidence, detail = classifier.predict_image(clipboard_img)
        return clipboard_img, result, detail
    
    # Gradio インターフェース作成
    with gr.Blocks(
        title="🐕 チワワ vs チクワ 判定器 🥢",
        theme=gr.themes.Soft(),
        css="""
        .main-header { text-align: center; margin: 20px 0; }
        .result-box {
            font-size: 72px !important;
            font-weight: 900 !important;
            text-align: center !important;
            color: #ff1744 !important;
            text-shadow: 4px 4px 8px rgba(0,0,0,0.5) !important;
            padding: 30px !important;
            background: linear-gradient(135deg, #ffebee, #ffcdd2) !important;
            border-radius: 20px !important;
            border: 5px solid #e91e63 !important;
            margin: 20px 0 !important;
            min-height: 120px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            text-transform: uppercase !important;
            letter-spacing: 3px !important;
        }
        .result-box textarea {
            font-size: 72px !important;
            font-weight: 900 !important;
            text-align: center !important;
            color: #ff1744 !important;
            background: transparent !important;
            border: none !important;
            text-shadow: 4px 4px 8px rgba(0,0,0,0.5) !important;
        }
        .detail-box { font-family: monospace; white-space: pre-line; }
        """
    ) as demo:
        
        gr.HTML("""
        <div class="main-header">
            <h1>🐕 チワワ vs チクワ AI判定器 🥢</h1>
            <p>画像をアップロードするか、クリップボードから取得してチワワとチクワを判定します</p>
        </div>
        """)
        
        with gr.Tab("📷 画像アップロード"):
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(
                        label="判定したい画像をドラッグ&ドロップまたはクリックして選択",
                        type="pil"
                    )
                    upload_btn = gr.Button("🔍 判定実行", variant="primary", size="lg")
                
                with gr.Column():
                    upload_result = gr.HTML(
                        label="判定結果"
                    )
                    upload_detail = gr.Textbox(
                        label="詳細情報",
                        interactive=False,
                        lines=6,
                        elem_classes=["detail-box"]
                    )
        
        with gr.Tab("📋 クリップボード"):
            with gr.Row():
                clipboard_btn = gr.Button(
                    "📋 クリップボードから画像を取得して判定",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Row():
                with gr.Column():
                    clipboard_image = gr.Image(
                        label="クリップボードから取得した画像",
                        interactive=False
                    )
                
                with gr.Column():
                    clipboard_result = gr.HTML(
                        label="判定結果"
                    )
                    clipboard_detail = gr.Textbox(
                        label="詳細情報",
                        interactive=False,
                        lines=6,
                        elem_classes=["detail-box"]
                    )
        
        with gr.Tab("ℹ️ 使用方法"):
            gr.Markdown("""
            ## 📖 使用方法
            
            ### 📷 画像アップロード
            1. 「画像アップロード」タブで画像ファイルを選択
            2. 「判定実行」ボタンをクリック
            3. 結果を確認
            
            ### 📋 クリップボード
            1. 他のアプリケーションで画像をコピー (Ctrl+C)
            2. 「クリップボード」タブで「クリップボードから画像を取得して判定」ボタンをクリック
            3. 自動的に画像が表示され、判定結果が表示されます
            
            ## 🎯 判定について
            - **チワワ**: 犬の画像として判定
            - **チクワ**: 食品の画像として判定
            - 信頼度が高いほど確信度が高い結果です
            
            ## 💡 ヒント
            - 明確にチワワまたはチクワが写っている画像で最も正確な結果が得られます
            - 画像は自動的に224x224ピクセルにリサイズされます
            - JPG、PNG、BMP等の一般的な画像形式に対応しています
            """)
        
        # イベントハンドラー
        upload_btn.click(
            fn=classify_uploaded_image,
            inputs=[input_image],
            outputs=[upload_result, upload_detail]
        )
        
        input_image.change(
            fn=classify_uploaded_image,
            inputs=[input_image],
            outputs=[upload_result, upload_detail]
        )
        
        clipboard_btn.click(
            fn=classify_clipboard_image,
            outputs=[clipboard_image, clipboard_result, clipboard_detail]
        )
    
    return demo

def main():
    """メイン関数"""
    print("🚀 チワワ vs チクワ判定器 Web UI を起動しています...")
    
    # Web UI作成
    demo = create_web_ui()
    
    # サーバー起動
    print("🌐 Webブラウザが自動的に開きます...")
    print("📱 ローカルアクセス: http://localhost:7860")
    print("🛑 終了するには Ctrl+C を押してください")
    
    demo.launch(
        server_name="localhost",
        server_port=None,  # 自動的に空いているポートを探す
        share=False,
        inbrowser=True,  # 自動的にブラウザを開く
        show_error=True
    )

if __name__ == "__main__":
    main()