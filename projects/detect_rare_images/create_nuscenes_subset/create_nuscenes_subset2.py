#!/usr/bin/env python

"""
データセットのinfoファイルからサブセットを作成するスクリプト
- ランダム選択
- レア検出画像の選択
をサポート
"""

import os
import pickle
import argparse
import random
from collections import defaultdict
from os import path as osp
from mmengine import print_log

def create_subset_infos(input_path, 
                        output_path, 
                        random_ratio=None, 
                        rare_ratio=None,
                        outlier_detection_path=None,
                        dataset_type='nuscenes'):
    """データセットのinfoファイルからサブセットを作成する

    Args:
        input_path (str): 元のinfoファイルのパス
        output_path (str): 出力するinfoファイルのパス
        random_ratio (float, optional): ランダムに選択するサンプルの割合
        rare_ratio (float, optional): レアケースから選択するサンプルの割合
        outlier_detection_path (str, optional): アウトライア検出結果のJSONファイルパス
        dataset_type (str, optional): データセットタイプ('nuscenes', 'kitti', 'waymo', 'lyft'など)
    """
    print_log(f'Loading info from {input_path}')
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
    
    # データセットタイプに応じて構造を取得
    if isinstance(data, list):
        # KITTIなどのリスト形式
        infos = data
        data_dict = {'infos': infos}
    else:
        # NuScenesなどの辞書形式
        infos = data.get('infos', data)  # 'infos'キーがなければデータ全体を使用
        data_dict = data
    
    total_samples = len(infos)
    print_log(f'Total samples: {total_samples}')
    
    # メタデータの準備
    metadata = data_dict.get('metadata', {}) if isinstance(data_dict, dict) else {}
    metadata.update({
        'random_ratio': random_ratio,
        'rare_ratio': rare_ratio,
        'original_samples': total_samples,
        'outlier_detection_used': outlier_detection_path is not None,
        'dataset_type': dataset_type
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
            if dataset_type == 'nuscenes':
                # 変更: outlier_tokensをoutlier_image_pathsに変更
                outlier_image_paths = set()
                
                # outlier_resultsから画像パスを取得
                for item in outlier_results:
                    # 従来のトークン方式もサポート（後方互換性のため）
                    if 'sample_token' in item:
                        outlier_image_paths.add(item.get('sample_token', ''))
                    
                    # 画像パスの収集（検出された画像パス）
                    if 'img_path' in item:
                        outlier_image_paths.add(item['img_path'])
                        
                    # # ネストされた構造内の画像パスも収集
                    # if 'images' in item:
                    #     for camera in item['images']:
                    #         if 'img_path' in item['images'][camera]:
                    #             outlier_image_paths.add(item['images'][camera]['img_path'])
                
                print_log(f'Found {len(outlier_image_paths)} unique outlier image paths')
                
                # 各infoオブジェクトをチェックして、画像パスが一致するものを収集
                for info in infos:
                    should_include = False
                    
                    # infoオブジェクト内の画像パスをチェック
                    if 'images' in info:
                        for camera in info['images']:
                            if 'img_path' in info['images'][camera]:
                                img_path = info['images'][camera]['img_path']
                                if img_path in outlier_image_paths:
                                    should_include = True
                                    break
                    
                    # 一致する画像パスが見つかった場合はサンプルを含める
                    if should_include:
                        rare_infos.append(info)
                
                print_log(f'Found {len(rare_infos)} samples with matching outlier image paths')
            elif dataset_type == 'kitti':
                outlier_tokens = set(item.get('image_idx', '') for item in outlier_results)
                token_field = 'image_idx'
            elif dataset_type in ['waymo', 'lyft']:
                outlier_tokens = set(item.get('token', '') for item in outlier_results)
                token_field = 'token'
            else:
                outlier_tokens = set()
                token_field = 'token'  # デフォルト
            
            print_log(f'Found {len(outlier_tokens)} unique outlier tokens')
            
            # トークンに基づいてレアケースを収集
            token_to_info = {info.get(token_field, ''): info for info in infos}
            for token in outlier_tokens:
                if token in token_to_info:
                    rare_infos.append(token_to_info[token])
            
            print_log(f'Found {len(rare_infos)} samples matching outlier tokens')
        else:
            # 従来の方法: クラスの出現頻度でレア判定
            class_counts = defaultdict(int)
            
            # データセットタイプに応じたクラス情報の取得
            if dataset_type == 'nuscenes':
                class_field = 'gt_names'
            elif dataset_type == 'kitti':
                class_field = 'annos.name' if 'annos' in infos[0] else 'gt_names'
            elif dataset_type in ['waymo', 'lyft']:
                class_field = 'gt_names' if 'gt_names' in infos[0] else 'annos.name'
            else:
                class_field = 'gt_names'  # デフォルト
            
            for info in infos:
                # ネストされたフィールドをドット記法で取得
                if '.' in class_field:
                    parts = class_field.split('.')
                    value = info
                    for part in parts:
                        if isinstance(value, dict) and part in value:
                            value = value[part]
                        else:
                            value = []
                            break
                    if isinstance(value, list):
                        for name in value:
                            class_counts[name] += 1
                elif class_field in info:
                    for name in info[class_field]:
                        class_counts[name] += 1
            
            # クラスを出現頻度でソート
            sorted_classes = sorted(class_counts.items(), key=lambda x: x[1])
            
            # 出現頻度の少ない上位3クラス（または全クラスの30%）を「レア」と定義
            num_rare_classes = min(3, max(1, int(len(sorted_classes) * 0.3)))
            rare_classes = [cls for cls, _ in sorted_classes[:num_rare_classes]]
            print_log(f'Rare classes (based on frequency): {rare_classes}')
            
            # レアクラスを含むサンプルを収集
            for info in infos:
                has_rare_class = False
                
                # ネストされたフィールドをドット記法で取得・評価
                if '.' in class_field:
                    parts = class_field.split('.')
                    value = info
                    for part in parts:
                        if isinstance(value, dict) and part in value:
                            value = value[part]
                        else:
                            value = []
                            break
                    if isinstance(value, list) and any(name in rare_classes for name in value):
                        has_rare_class = True
                elif class_field in info and any(name in rare_classes for name in info[class_field]):
                    has_rare_class = True
                
                if has_rare_class:
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
    
    # 結果を保存
    if isinstance(data, list):
        result = selected_infos
    else:
        data_dict['infos'] = selected_infos
        data_dict['metadata'] = metadata
        result = data_dict
    
    print_log(f'Selected {len(selected_infos)} samples ({len(selected_infos)/total_samples:.2%} of original)')
    print_log(f'Saving to {output_path}')
    
    os.makedirs(osp.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(result, f)
    
    print_log('Done!')

def main():
    parser = argparse.ArgumentParser(description='Create dataset subset infos')
    parser.add_argument('--input', required=False, default="./data/nuscenes/nuscenes_infos_test.pkl", help='Input info pkl file')
    parser.add_argument('--output', required=False, help='Output info pkl file (if not specified, auto-generated from input and params)')
    parser.add_argument('--random-ratio', type=float, default=0.1, 
                        help='Ratio of random samples to select (e.g., 0.2 for 20%)')
    parser.add_argument('--rare-ratio', type=float, default=0.1,
                        help='Ratio of rare case samples to select (e.g., 0.1 for 10%)')
    parser.add_argument('--outlier-detection', type=str, default="outlier_detection_results.json",
                        help='Path to outlier detection results JSON file')
    parser.add_argument('--dataset-type', type=str, default='nuscenes',
                        help='Dataset type (nuscenes, kitti, waymo, lyft)')
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
        args.outlier_detection,
        args.dataset_type
    )

if __name__ == '__main__':
    main()
