#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
画像ダウンロードのテスト用スクリプト
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
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_download():
    """テスト用ダウンロード（5枚のみ）"""
    base_dir = Path("test_dataset")
    base_dir.mkdir(exist_ok=True)
    
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })
    
    try:
        with DDGS() as ddgs:
            results = list(ddgs.images(
                keywords="chihuahua dog",
                region="jp-jp",
                safesearch="moderate",
                size="medium",
                max_results=10
            ))
            
        urls = [result['image'] for result in results[:5]]  # 最初の5個のみ
        logger.info(f"取得したURL数: {len(urls)}")
        
        success_count = 0
        for i, url in enumerate(urls):
            try:
                logger.info(f"ダウンロード中: {url}")
                response = session.get(url, timeout=15)
                response.raise_for_status()
                
                # 簡単なサイズチェックのみ
                content_length = len(response.content)
                if content_length < 1000:  # 1KB以上
                    logger.warning(f"ファイルサイズが小さすぎます: {content_length} bytes")
                    continue
                
                # ファイルに保存
                filename = f"test_chihuahua_{i:03d}.jpg"
                save_path = base_dir / filename
                
                with open(save_path, 'wb') as f:
                    f.write(response.content)
                
                success_count += 1
                logger.info(f"保存成功: {filename} ({content_length} bytes)")
                
            except Exception as e:
                logger.error(f"ダウンロードエラー: {e}")
            
            time.sleep(1)  # 間隔を空ける
        
        logger.info(f"テスト完了: {success_count}/{len(urls)} 個の画像をダウンロード")
        return success_count > 0
        
    except Exception as e:
        logger.error(f"検索エラー: {e}")
        return False

if __name__ == "__main__":
    test_download()