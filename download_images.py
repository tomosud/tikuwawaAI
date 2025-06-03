#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chihuahua vs Chikuwa Classifier - Phase 1: 画像収集スクリプト

このスクリプトは DuckDuckGo を使用して「チワワ」と「チクワ」の画像を
自動でダウンロードし、dataset フォルダに保存します。
"""

import os
import requests
import uuid
from pathlib import Path
from PIL import Image
from duckduckgo_search import DDGS
import time
import logging

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('download_log.txt', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ImageDownloader:
    def __init__(self, base_dir="dataset", max_images=100):
        """
        画像ダウンローダーの初期化
        
        Args:
            base_dir (str): 画像保存先のベースディレクトリ
            max_images (int): 各カテゴリの最大ダウンロード画像数
        """
        self.base_dir = Path(base_dir)
        self.max_images = max_images
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # ディレクトリ作成
        self.base_dir.mkdir(exist_ok=True)
        
    def search_images(self, query, category):
        """
        DuckDuckGo で画像を検索してURLリストを取得
        
        Args:
            query (str): 検索クエリ
            category (str): カテゴリ名（保存先フォルダ名）
            
        Returns:
            list: 画像URLのリスト
        """
        logger.info(f"'{query}' の画像を検索中...")
        
        try:
            with DDGS() as ddgs:
                # 画像検索を実行
                results = list(ddgs.images(
                    keywords=query,
                    region="jp-jp",
                    safesearch="moderate",
                    size="medium",
                    max_results=self.max_images * 2  # 多めに取得してフィルタリング
                ))
                
            urls = [result['image'] for result in results]
            logger.info(f"'{query}' で {len(urls)} 個の画像URLを取得")
            return urls
            
        except Exception as e:
            logger.error(f"画像検索でエラーが発生: {e}")
            return []
    
    def download_image(self, url, save_path):
        """
        画像をダウンロードして保存
        
        Args:
            url (str): 画像URL
            save_path (Path): 保存先パス
            
        Returns:
            bool: ダウンロード成功時True
        """
        try:
            # 画像をダウンロード
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            # ファイルサイズチェック（最小5KB、最大10MB）
            content_length = len(response.content)
            if content_length < 5000 or content_length > 10_000_000:
                return False
                
            # 画像として有効かチェック
            try:
                from io import BytesIO
                img_bytes = BytesIO(response.content)
                with Image.open(img_bytes) as img:
                    # 最小サイズチェック（30x30ピクセル）
                    if img.width < 30 or img.height < 30:
                        return False
                    
                    # RGB形式に変換して保存
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    img.save(save_path, 'JPEG', quality=85)
                    return True
                    
            except Exception as e:
                logger.debug(f"画像処理エラー: {e}")
                # PILで処理できない場合は直接保存を試行
                try:
                    with open(save_path, 'wb') as f:
                        f.write(response.content)
                    return True
                except Exception:
                    return False
                
        except Exception as e:
            logger.debug(f"ダウンロードエラー {url}: {e}")
            return False
    
    def download_category(self, query, category):
        """
        特定カテゴリの画像をダウンロード
        
        Args:
            query (str): 検索クエリ
            category (str): カテゴリ名
        """
        # 保存先ディレクトリ作成
        category_path = self.base_dir / category
        category_path.mkdir(exist_ok=True)
        
        # 既存ファイル数を確認
        existing_files = list(category_path.glob("*.jpg"))
        if len(existing_files) >= self.max_images:
            logger.info(f"{category} は既に {len(existing_files)} 個の画像があります")
            return
        
        # 画像URLを検索
        urls = self.search_images(query, category)
        if not urls:
            logger.warning(f"{category} の画像URLが取得できませんでした")
            return
        
        # ダウンロード実行
        success_count = len(existing_files)
        logger.info(f"{category} の画像ダウンロード開始...")
        
        for i, url in enumerate(urls):
            if success_count >= self.max_images:
                break
                
            # ファイル名生成（UUID + 連番）
            filename = f"{category}_{uuid.uuid4().hex[:8]}_{i:03d}.jpg"
            save_path = category_path / filename
            
            # 既に存在する場合はスキップ
            if save_path.exists():
                continue
            
            # ダウンロード実行
            if self.download_image(url, save_path):
                success_count += 1
                logger.info(f"{category}: {success_count}/{self.max_images} - {filename}")
            
            # リクエスト間隔を空ける（CAPTCHA対策）
            time.sleep(0.5)
        
        logger.info(f"{category} 完了: {success_count} 個の画像をダウンロード")

def main():
    """メイン処理"""
    logger.info("=== Chihuahua vs Chikuwa 画像収集開始 ===")
    
    # ダウンローダー初期化
    downloader = ImageDownloader(max_images=50)  # 各カテゴリ50枚ずつ
    
    # 検索クエリ設定
    search_queries = {
        "chihuahua": ["チワワ 犬", "chihuahua dog", "チワワ かわいい"],
        "chikuwa": ["ちくわ 食べ物", "chikuwa food", "竹輪"]
    }
    
    # 各カテゴリをダウンロード
    for category, queries in search_queries.items():
        logger.info(f"\n--- {category.upper()} カテゴリ開始 ---")
        
        for query in queries:
            logger.info(f"検索クエリ: '{query}'")
            downloader.download_category(query, category)
            time.sleep(2)  # クエリ間の間隔
    
    logger.info("\n=== 画像収集完了 ===")
    
    # 結果サマリー
    chihuahua_count = len(list(Path("dataset/chihuahua").glob("*.jpg")))
    chikuwa_count = len(list(Path("dataset/chikuwa").glob("*.jpg")))
    
    print(f"\n【収集結果】")
    print(f"チワワ画像: {chihuahua_count} 枚")
    print(f"チクワ画像: {chikuwa_count} 枚")
    print(f"合計: {chihuahua_count + chikuwa_count} 枚")
    
    if chihuahua_count > 0 and chikuwa_count > 0:
        print("\n✅ フェーズ1（画像収集）が正常に完了しました！")
        print("次はフェーズ2（データ前処理）に進むことができます。")
    else:
        print("\n⚠️  一部のカテゴリで画像収集に失敗しました。")
        print("ネットワーク接続やAPIの制限を確認してください。")

if __name__ == "__main__":
    main()