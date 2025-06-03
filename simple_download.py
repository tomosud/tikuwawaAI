#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
シンプルな画像ダウンロードスクリプト（チワワ・チクワ）
"""

import os
import requests
import uuid
from pathlib import Path
from duckduckgo_search import DDGS
import time
import logging

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def download_images(query, category, max_images=25):
    """指定されたクエリで画像をダウンロード"""
    
    # ディレクトリ作成
    base_dir = Path("dataset")
    category_dir = base_dir / category
    category_dir.mkdir(parents=True, exist_ok=True)
    
    # 既存ファイルをチェック
    existing_files = list(category_dir.glob("*.jpg"))
    if len(existing_files) >= max_images:
        logger.info(f"{category} は既に {len(existing_files)} 個の画像があります")
        return len(existing_files)
    
    logger.info(f"{category} カテゴリの画像収集開始: '{query}'")
    
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })
    
    try:
        # DuckDuckGoで画像検索
        with DDGS() as ddgs:
            results = list(ddgs.images(
                keywords=query,
                region="jp-jp",
                safesearch="moderate",
                size="medium",
                max_results=max_images * 2  # 余分に取得
            ))
            
        urls = [result['image'] for result in results]
        logger.info(f"'{query}' で {len(urls)} 個のURLを取得")
        
        success_count = len(existing_files)
        
        for i, url in enumerate(urls):
            if success_count >= max_images:
                break
                
            try:
                logger.info(f"ダウンロード中 ({success_count+1}/{max_images}): {url[:50]}...")
                
                response = session.get(url, timeout=15)
                response.raise_for_status()
                
                # 基本的なサイズチェック
                content_length = len(response.content)
                if content_length < 1000 or content_length > 5_000_000:  # 1KB-5MB
                    logger.warning(f"ファイルサイズ不適切: {content_length} bytes")
                    continue
                
                # ファイル名生成
                filename = f"{category}_{uuid.uuid4().hex[:8]}_{success_count:03d}.jpg"
                save_path = category_dir / filename
                
                # ファイル保存
                with open(save_path, 'wb') as f:
                    f.write(response.content)
                
                success_count += 1
                logger.info(f"保存成功: {filename} ({content_length} bytes)")
                
            except Exception as e:
                logger.warning(f"ダウンロードエラー: {e}")
            
            time.sleep(0.8)  # レート制限対策
        
        logger.info(f"{category} 完了: {success_count} 個の画像")
        return success_count
        
    except Exception as e:
        logger.error(f"{category} 検索エラー: {e}")
        return len(existing_files)

def main():
    """メイン処理"""
    logger.info("=== シンプル画像収集開始 ===")
    
    # 検索設定
    categories = [
        ("チワワ 犬", "chihuahua"),
        ("ちくわ 食べ物", "chikuwa")
    ]
    
    results = {}
    
    for query, category in categories:
        results[category] = download_images(query, category, max_images=25)
        time.sleep(3)  # カテゴリ間の待機
    
    # 結果表示
    logger.info("\n=== 収集結果 ===")
    total = 0
    for category, count in results.items():
        logger.info(f"{category}: {count} 枚")
        total += count
    
    logger.info(f"合計: {total} 枚")
    
    if results["chihuahua"] > 0 and results["chikuwa"] > 0:
        logger.info("✅ 画像収集が正常に完了しました！")
    else:
        logger.warning("⚠️ 一部のカテゴリで収集に失敗しました")

if __name__ == "__main__":
    main()