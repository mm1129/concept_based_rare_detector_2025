#!/usr/bin/env python
# filepath: /home/maitsujimoto/autoware-ml/tools/detection3d/create_nuscenes_subset.py

"""
NuScenesのinfoファイルからサブセットを作成するスクリプト
- ランダム選択
- レア検出画像の選択
をサポート
"""

import os
import pickle
import argparse
import random
import numpy as np
from collections import defaultdict
from os import path as osp
from mmengine import print_log

def create_subset_infos(input_path, 
                        output_path, 
                        random_ratio=None, 
                        rare_ratio=None,
                        outlier_detection_path=None):  # 追加: アウトライア検出結果のパス
    """NuScenes infoファイルからサブセットを作成する

    Args:
        input_path (str): 元のinfoファイルのパス
        output_path (str): 出力するinfoファイルのパス
        random_ratio (float, optional): ランダムに選択するサンプルの割合
        rare_ratio (float, optional): レアケースから選択するサンプルの割合
        outlier_detection_path (str, optional): アウトライア検出結果のJSONファイルパス
    """
    print_log(f'Loading info from {input_path}')
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
    
    # 入力データの構造を確認
    if 'metainfo' in data and 'data_list' in data:
        # MMDetection3D v1.1+形式
        infos = data['data_list']
        metadata = data['metainfo']
        new_format = True
    else:
        # 従来形式
        infos = data.get('infos', data if isinstance(data, list) else [])
        metadata = data.get('metadata', {})
        new_format = False
    
    total_samples = len(infos)
    print_log(f'Total samples: {total_samples}')
    
    # メタデータの準備
    metadata = metadata.copy() if metadata else {}
    metadata.update({
        'random_ratio': random_ratio,
        'rare_ratio': rare_ratio,
        'original_samples': total_samples,
        'outlier_detection_used': outlier_detection_path is not None
    })
    
    selected_infos = []
    
    # レアケースの抽出
    rare_infos = []
    if rare_ratio is not None and rare_ratio > 0:
        rare_count = int(total_samples * rare_ratio)
        print_log(f'Selecting {rare_count} rare case samples')
        
        # アウトライア検出結果が提供されている場合はそれを使用
        if outlier_detection_path and os.path.exists(outlier_detection_path):
            import json
            with open(outlier_detection_path, 'r') as f:
                outlier_results = json.load(f)
            
            print_log(f'Loaded {len(outlier_results)} outlier detection results')
            
            # サンプルトークンでフィルタリングするためにマッピングを作成
            outlier_tokens = set(item['sample_token'] for item in outlier_results)
            print_log(f'Found {len(outlier_tokens)} unique outlier tokens')
            
            # トークンに基づいてレアケースを収集
            token_to_info = {}
            for info in infos:
                # 新旧データ形式に応じてトークンを取得
                if new_format:
                    token = info.get('sample_idx', info.get('token', ''))
                else:
                    token = info.get('token', '')
                
                if token:
                    token_to_info[token] = info
            
            for token in outlier_tokens:
                if token in token_to_info:
                    rare_infos.append(token_to_info[token])
            
            print_log(f'Found {len(rare_infos)} samples matching outlier tokens')
        else:
            # 従来の方法: クラスの出現頻度でレア判定
            class_counts = defaultdict(int)
            for info in infos:
                if new_format:
                    gt_names = [ann.get('category_name') for ann in info.get('instances', [])]
                else:
                    gt_names = info.get('gt_names', [])
                
                for name in gt_names:
                    class_counts[name] += 1
            
            # クラスを出現頻度でソート
            sorted_classes = sorted(class_counts.items(), key=lambda x: x[1])
            
            # 出現頻度の少ない上位3クラスを「レア」と定義
            rare_classes = [cls for cls, _ in sorted_classes[:3]]
            print_log(f'Rare classes (based on frequency): {rare_classes}')
            
            # レアクラスを含むサンプルを収集
            for info in infos:
                if new_format:
                    gt_names = [ann.get('category_name') for ann in info.get('instances', [])]
                else:
                    gt_names = info.get('gt_names', [])
                
                if any(name in rare_classes for name in gt_names):
                    rare_infos.append(info)
            
            print_log(f'Found {len(rare_infos)} samples containing rare classes')
        
        # 必要数を超える場合はランダムに選択
        if len(rare_infos) > rare_count:
            rare_infos = random.sample(rare_infos, rare_count)
        
        selected_infos.extend(rare_infos)
    
    # ランダムサンプルの追加
    if random_ratio is not None and random_ratio > 0:
        random_count = int(total_samples * random_ratio)
        print_log(f'Selecting {random_count} random samples')
        
        # レアケースとして選択されていないサンプルからランダムに選択
        remaining_infos = [info for info in infos if info not in rare_infos]
        random_infos = random.sample(remaining_infos, min(random_count, len(remaining_infos)))
        
        selected_infos.extend(random_infos)
    
    if not selected_infos:
        selected_infos = infos  # 選択基準がない場合は全サンプルを使用
    
    # 結果を保存 (元のデータ形式に合わせる)
    if new_format:
        result = {
            'metainfo': metadata,
            'data_list': selected_infos
        }
    elif isinstance(data, list):
        result = selected_infos
    else:
        result = {
            'infos': selected_infos,
            'metadata': metadata
        }
    
    print_log(f'Selected {len(selected_infos)} samples ({len(selected_infos)/total_samples:.2%} of original)')
    print_log(f'Saving to {output_path}')
    
    os.makedirs(osp.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(result, f)
    
    print_log('Done!')

def main():
    parser = argparse.ArgumentParser(description='Create NuScenes subset infos')
    parser.add_argument('--input', required=False, default="./data/nuscenes/nuscenes_infos_test.pkl", help='Input info pkl file')
    parser.add_argument('--output', required=False, help='Output info pkl file (if not specified, auto-generated from input and outlier-detection)')
    parser.add_argument('--random-ratio', type=float, default=0.1, 
                        help='Ratio of random samples to select (e.g., 0.2 for 20%)')
    parser.add_argument('--rare-ratio', type=float, default=0.1,
                        help='Ratio of rare case samples to select (e.g., 0.1 for 10%)')
    parser.add_argument('--outlier-detection', type=str, default="outlier_detection_results.json", #new_concept/
                        help='Path to outlier detection results JSON file')
    args = parser.parse_args()
    
    # outputが指定されていない場合は自動生成
    if args.output is None:
        # Get base names
        input_basename = os.path.basename(args.input)
        input_name = os.path.splitext(input_basename)[0]
        outlier_basename = os.path.basename(args.outlier_detection) if args.outlier_detection else "no_outlier"
        outlier_name = os.path.splitext(outlier_basename)[0]
        
        # Generate descriptive parts
        random_part = f"_random{args.random_ratio}" if args.random_ratio is not None else ""
        rare_part = f"_rare{args.rare_ratio}" if args.rare_ratio is not None else ""
        
        # Combine all parts
        output_dir = os.path.dirname(args.input)
        args.output = os.path.join(output_dir, f"{outlier_name}{random_part}{rare_part}.pkl")
        print_log(f"Auto-generated output path: {args.output}")
    create_subset_infos(
        args.input, 
        args.output,
        args.random_ratio,
        args.rare_ratio,
        args.outlier_detection
    )

if __name__ == '__main__':
    main()