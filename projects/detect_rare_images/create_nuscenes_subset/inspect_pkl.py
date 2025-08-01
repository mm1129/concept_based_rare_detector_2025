#!/usr/bin/env python

import pickle
import sys
import pprint
import argparse
import json
import os
import shutil
from os import path as osp

def parse_args():
    """コマンドライン引数を解析する

    Returns:
        argparse.Namespace: 解析された引数
    """
    parser = argparse.ArgumentParser(description='PKLファイルの内容を検査して表示するツール')
    parser.add_argument('file_path', nargs='?', default='./data/nuscenes/nuscenes_infos_train_interim_results_construction_vehicle_bicycle_pedestrian_with_bicycle_with_outliers_with_final_outliers_random0.1_rare0.1.pkl',
                        help='検査するPickleファイルのパス')
    parser.add_argument('-o', '--output', help='サンプルデータをJSON形式で出力するファイルパス')
    parser.add_argument('-n', '--num-items', type=int, default=2,
                        help='リストから表示する項目数（デフォルト: 2）')
    parser.add_argument('-d', '--depth', type=int, default=3,
                        help='表示するネスト構造の最大深さ（デフォルト: 3）')
    parser.add_argument('--target-dir', default='./selected_0426', help='選択されたCAM_FRONTの画像をコピーする先のディレクトリ')
    parser.add_argument('--camera-type', default='CAM_FRONT', help='対象とするカメラタイプ（デフォルト: CAM_FRONT）')
    return parser.parse_args()

def inspect_pkl(file_path, output=None, num_items=2, depth=3, target_dir=None, camera_type='CAM_FRONT'):
    """PKLファイルの内容を表示し、必要に応じて画像をコピーする"""
    print(f"Inspecting: {file_path}")
    
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    # データのタイプとサイズを表示
    print(f"Data type: {type(data)}")
    if isinstance(data, list):
        print(f"List length: {len(data)}")
        if len(data) > 0:
            print(f"\nFirst {min(num_items, len(data))} items structure:")
            for i in range(min(num_items, len(data))):
                print(f"\nItem {i}:")
                pprint.pprint(data[i], depth=depth, compact=True)
    elif isinstance(data, dict):
        print(f"Dict keys: {list(data.keys())}")
        if 'infos' in data:
            print(f"Number of infos: {len(data['infos'])}")
            if len(data['infos']) > 0:
                print(f"\nFirst {min(num_items, len(data['infos']))} info structures:")
                for i in range(min(num_items, len(data['infos']))):
                    print(f"\nInfo {i}:")
                    pprint.pprint(data['infos'][i], depth=depth, compact=True)
        elif 'data_list' in data:  # MMDetection3D v1.1+形式のサポートを追加
            print(f"Number of data_list items: {len(data['data_list'])}")
            if len(data['data_list']) > 0:
                print(f"\nFirst {min(num_items, len(data['data_list']))} data_list structures:")
                for i in range(min(num_items, len(data['data_list']))):
                    print(f"\nData item {i}:")
                    pprint.pprint(data['data_list'][i], depth=depth, compact=True)
        
        if 'metadata' in data:
            print("\nMetadata:")
            pprint.pprint(data['metadata'])
        elif 'metainfo' in data:  # MMDetection3D v1.1+形式のサポートを追加
            print("\nMetainfo:")
            pprint.pprint(data['metainfo'])
    
    # サブセット情報の分析
    original_file_path = file_path.replace('_random', '').replace('_rare', '')
    original_file_path = original_file_path.split('_outlier')[0] 
    if not original_file_path.endswith('.pkl'):
        original_file_path += '.pkl'
    
    # ファイル名からサブセットのパラメータ情報を抽出
    import re
    random_ratio_match = re.search(r'_random([0-9.]+)', file_path)
    rare_ratio_match = re.search(r'_rare([0-9.]+)', file_path)
    
    random_ratio = float(random_ratio_match.group(1)) if random_ratio_match else None
    rare_ratio = 0.1 #float(rare_ratio_match.group(1)) if rare_ratio_match else None
    
    # 元のデータセットが存在し、かつパラメータが抽出できた場合
    if random_ratio is not None or rare_ratio is not None:
        try:
            print("\nSubset analysis:")
            if os.path.exists(original_file_path):
                with open(original_file_path, 'rb') as f:
                    original_data = pickle.load(f)
                
                # 元データとサブセットデータのサンプル数を取得
                if isinstance(original_data, list):
                    original_samples = len(original_data)
                elif isinstance(original_data, dict):
                    if 'infos' in original_data:
                        original_samples = len(original_data['infos'])
                    elif 'data_list' in original_data:
                        original_samples = len(original_data['data_list'])
                    else:
                        original_samples = None
                else:
                    original_samples = None
                
                if isinstance(data, list):
                    subset_samples = len(data)
                elif isinstance(data, dict):
                    if 'infos' in data:
                        subset_samples = len(data['infos'])
                    elif 'data_list' in data:
                        subset_samples = len(data['data_list'])
                    else:
                        subset_samples = None
                else:
                    subset_samples = None
                
                if original_samples is not None and subset_samples is not None:
                    print(f"- Original dataset: {original_samples} samples")
                    print(f"- Subset dataset: {subset_samples} samples")
                    print(f"- Subset ratio: {subset_samples/original_samples:.2%}")
                    
                    if random_ratio is not None:
                        expected_random = int(original_samples * random_ratio)
                        print(f"- Expected random samples: {expected_random} ({random_ratio:.2%} of original)")
                    
                    if rare_ratio is not None:
                        expected_rare = int(original_samples * rare_ratio)
                        print(f"- Expected rare samples: {expected_rare} ({rare_ratio:.2%} of original)")
                    
                    if random_ratio is not None and rare_ratio is not None:
                        expected_total = min(original_samples, expected_random + expected_rare)
                        print(f"- Expected total subset: {expected_total} samples")
                        print(f"- Actual vs Expected: {subset_samples}/{expected_total} ({subset_samples/expected_total:.2%})")
            else:
                print(f"- Original dataset not found at {original_file_path}")
                if random_ratio is not None:
                    print(f"- Random ratio parameter: {random_ratio:.2%}")
                if rare_ratio is not None:
                    print(f"- Rare ratio parameter: {rare_ratio:.2%}")
        except Exception as e:
            print(f"Error during subset analysis: {e}")
    
    # サマリー情報の作成
    print("\nSummary information:")
    
    # もし辞書タイプなら
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, list):
                print(f"- {key}: {len(value)} items")
            else:
                print(f"- {key}: {type(value)}")
    
    # JSONファイルへの出力
    if output:
        try:
            sample_data = {}
            if isinstance(data, list) and len(data) > 0:
                sample_data['list_sample'] = data[:min(num_items, len(data))]
            elif isinstance(data, dict):
                sample_data = {k: v for k, v in data.items() if not isinstance(v, list)}
                if 'infos' in data and len(data['infos']) > 0:
                    sample_data['infos_sample'] = data['infos'][:min(num_items, len(data['infos']))]
            
            with open(output, 'w') as f:
                json.dump(sample_data, f, indent=4)
            print(f"\nSample data exported to {output}")
        except Exception as e:
            print(f"Error exporting to JSON: {e}")
    
    # 画像ファイルの収集とコピー機能を追加
    if target_dir:
        os.makedirs(target_dir, exist_ok=True)
        selected_images = []

        def collect_image_paths(data_item):
            """データ項目から画像パスを抽出"""
            if isinstance(data_item, dict):
                # MMDetection3D v1.1+形式
                if 'images' in data_item and camera_type in data_item['images']:
                    return data_item['images'][camera_type].get('img_path', '')
                # 古い形式
                elif 'cams' in data_item and camera_type in data_item['cams']:
                    return data_item['cams'][camera_type].get('data_path', '')
            return None

        # データ構造に応じて画像パスを収集
        if isinstance(data, list):
            for item in data:
                img_path = collect_image_paths(item)
                if img_path:
                    selected_images.append(img_path)
        elif isinstance(data, dict):
            if 'data_list' in data:
                for item in data['data_list']:
                    img_path = collect_image_paths(item)
                    if img_path:
                        selected_images.append(img_path)
            elif 'infos' in data:
                for item in data['infos']:
                    img_path = collect_image_paths(item)
                    if img_path:
                        selected_images.append(img_path)

        print(f"\nFound {len(selected_images)} images for {camera_type}")
        
        # 画像のコピー
        copied_count = 0
        for img_path in selected_images:
            # データセット内の画像パスを推測
            src_path = osp.join('data_nuscenes/samples/CAM_FRONT', osp.basename(img_path))
            
            # 実際のファイルを探してコピー
            print(f"Copying {img_path} to {target_dir}")
            print(f"Checking {src_path}")
            if osp.exists(src_path):
                dst_path = osp.join(target_dir, osp.basename(img_path))
                shutil.copy2(src_path, dst_path)
                copied_count += 1
        print(f"Copied {copied_count} images to {target_dir}")

    print("\nDone!")

if __name__ == "__main__":
    args = parse_args()
    if args.output is None:
        basename = os.path.basename(args.file_path)
        args.output = basename.replace('.pkl', '.json')
    inspect_pkl(args.file_path, args.output, args.num_items, args.depth, args.target_dir, args.camera_type)