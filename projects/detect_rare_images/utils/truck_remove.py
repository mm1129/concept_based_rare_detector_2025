#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import re

def rename_truck_files(source_dir, target_dir):
    """
    ファイル名から "_数字_truck" パターンを削除して新しいディレクトリにコピーする
    
    Args:
        source_dir: 元のファイルがあるディレクトリ
        target_dir: 修正したファイルを保存するディレクトリ
    """
    # ターゲットディレクトリの作成
    os.makedirs(target_dir, exist_ok=True)
    
    # ファイル名のパターン: _数字_truck.jpg
    pattern = re.compile(r'_\d+_truck')
    
    # ディレクトリ内のすべてのファイルを処理
    for filename in os.listdir(source_dir):
        if filename.endswith('.jpg'):
            # パターンに一致する部分を削除して新しいファイル名を作成
            new_filename = pattern.sub('', filename)
            
            # 元のファイルパスと新しいファイルパスを構築
            source_path = os.path.join(source_dir, filename)
            target_path = os.path.join(target_dir, new_filename)
            
            # ファイルをコピー
            shutil.copy2(source_path, target_path)
            print(f"コピー: {filename} -> {new_filename}")

if __name__ == "__main__":
    # 現在のディレクトリを取得
    current_dir = os.getcwd()
    # 新しいディレクトリ名
    new_dir = os.path.join(current_dir, "cleaned_images")
    
    # ファイルのリネームとコピー
    rename_truck_files(current_dir, new_dir)
    
    print(f"\n処理完了: {new_dir} にファイルが保存されました")