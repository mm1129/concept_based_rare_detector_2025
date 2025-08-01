#!/usr/bin/env python

"""
指定されたイメージフォルダ内の画像が含まれるサンプルのみからなる
データセットのサブセットを作成するスクリプト
"""

import os
import pickle
import argparse
import glob
import random
import numpy as np
from os import path as osp
from mmengine import print_log

def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)

def create_filtered_subset(input_path, 
                          output_path, 
                          image_folder_path,
                          camera_type='CAM_FRONT',
                          random_ratio=0.0,
                          rare_ratio=1.0,
                          seed=42):
    """データセットの中から、指定されたイメージフォルダ内の画像を含むサンプルのみを抽出する

    Args:
        input_path (str): 元のinfoファイルのパス
        output_path (str): 出力するinfoファイルのパス
        image_folder_path (str): 画像ファイルが含まれるフォルダのパス
        camera_type (str, optional): 使用するカメラタイプ（デフォルト: 'CAM_FRONT'）
        random_ratio (float, optional): 無作為選択するサンプルの比率（デフォルト: 0.0）
        rare_ratio (float, optional): レアケースから選択するサンプルの比率（デフォルト: 1.0）
        seed (int, optional): 乱数シードの値（デフォルト: 42）
    """
    set_seed(seed)  # シード値を設定
    
    print_log(f'Loading info from {input_path}')
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
    
    # 入力データの構造を確認
    if 'metainfo' in data and 'data_list' in data:
        # MMDetection3D v1.1+形式
        data_list = data['data_list']
        metainfo = data['metainfo']
        new_format = True
    else:
        # 従来形式
        data_list = data.get('infos', data if isinstance(data, list) else [])
        metainfo = data.get('metadata', {})
        new_format = False
    
    total_samples = len(data_list)
    print_log(f'Total samples in original dataset: {total_samples}')
    
    # 画像フォルダ内のファイルリストを取得
    image_files = set()
    for ext in ['jpg', 'png', 'jpeg']:
        image_files.update(
            [osp.basename(f) for f in glob.glob(osp.join(image_folder_path, f'*.{ext}'))]
        )
    
    print_log(f'Found {len(image_files)} images in {image_folder_path}')
    
    # 画像パスに基づいてサンプルをフィルタリング
    rare_candidates = []  # レア画像の候補を保存するリスト
    already_selected_indices = set()  # 選択されたサンプルのインデックスを記録
    
    # レア画像からの選択数を計算
    num_rare_to_select = int(total_samples * rare_ratio)
    print_log(f'Will select up to {num_rare_to_select} rare samples')
    
    # まず、すべてのレア候補を収集
    for i, sample in enumerate(data_list):
        # データ構造に応じて画像情報にアクセス
        if new_format:
            # MMDetection3D v1.1+形式
            if 'images' in sample and camera_type in sample['images']:
                img_path = sample['images'][camera_type].get('img_path', '')
                if osp.basename(img_path) in image_files:
                    rare_candidates.append((i, sample))
        else:
            # 古い形式（各データセットによって構造が異なる可能性あり）
            cams = sample.get('cams', {})
            if cams and camera_type in cams:
                img_path = cams[camera_type].get('data_path', '')
                if osp.basename(img_path) in image_files:
                    rare_candidates.append((i, sample))
            # NuScenes特有の構造の場合
            elif 'images' in sample and camera_type in sample['images']:
                img_path = sample['images'][camera_type].get('img_path', '')
                if osp.basename(img_path) in image_files:
                    rare_candidates.append((i, sample))
    
    print_log(f'Found {len(rare_candidates)} rare candidates')
    
    # レア候補からランダムに選択
    filtered_data_list = []
    if rare_candidates:
        # 選択数を調整
        num_rare_to_select = min(num_rare_to_select, len(rare_candidates))
        
        # ランダムに選択
        selected_indices = random.sample(range(len(rare_candidates)), num_rare_to_select)
        for idx in selected_indices:
            i, sample = rare_candidates[idx]
            filtered_data_list.append(sample)
            already_selected_indices.add(i)
    
    print_log(f'Selected {len(filtered_data_list)} samples containing images from the folder')
    selected_data_list = filtered_data_list
    
    # 無作為選択するサンプルの数を計算
    num_random_to_select = int(total_samples * random_ratio)
    print_log(f'Will select {num_random_to_select} random samples')
    
    # 未選択のサンプルインデックスを取得
    remaining_indices = [i for i in range(total_samples) if i not in already_selected_indices]
    
    # 無作為選択するサンプル数の調整
    num_random_to_select = min(num_random_to_select, len(remaining_indices))
    
    # 無作為選択
    if num_random_to_select > 0:
        random_indices = random.sample(remaining_indices, num_random_to_select)
        for i in random_indices:
            selected_data_list.append(data_list[i])
    
    print_log(f'Added {num_random_to_select} random samples')
    print_log(f'Total selected samples: {len(selected_data_list)}')
    
    # 元のデータ形式を維持して結果を保存
    if new_format:
        result = {
            'metainfo': metainfo,
            'data_list': selected_data_list
        }
    elif isinstance(data, list):
        result = selected_data_list
    else:
        # 辞書形式の場合は元の構造を維持
        result = data.copy()
        if 'infos' in result:
            result['infos'] = selected_data_list
        else:
            result = selected_data_list
    
    # 出力ディレクトリが存在しない場合は作成
    os.makedirs(osp.dirname(output_path), exist_ok=True)
    
    # 結果を保存
    print_log(f'Saving filtered dataset to {output_path}')
    with open(output_path, 'wb') as f:
        pickle.dump(result, f)
    
    print_log('Done!')
    
    return len(selected_data_list)

# JSONファイルの処理フローを解説するコメントを追加
"""
outlier_detection_results.jsonの処理フロー:
1. JSONファイルを読み込み、各エントリから `sample_token` を抽出
   例: "sample_token": "n008-2018-08-31-11-19-57-0400__CAM_FRONT__1535729241912404"
2. 重複を排除し、ユニークな `sample_token` のセットを作成
3. データセット内の各サンプルについて:
   a. サンプルから sample_token または画像パスを抽出
   b. 抽出した sample_token が JSONから取得したトークンセットに含まれているか確認
   c. 含まれていれば、そのサンプルをレアサンプルとして選択

注意点:
- 同じ画像に対して複数のオブジェクト(person, carなど)が検出されていても、
  sample_token が同じなら1つのサンプルとして扱われます
- データセット内のサンプル数 × rare_ratio が選択できるレアサンプルの上限です
- 指定された比率分のレアサンプルが見つからない場合は、見つかった分だけ選択されます
"""

def create_subset_infos(input_path, output_path, random_ratio=0.1, rare_ratio=0.1, outlier_detection_path=None, camera_type='CAM_FRONT', seed=42):
    """データセットから、検出された特殊（レア）ケースと無作為選択したサンプルを組み合わせたサブセットを作成する

    Args:
        input_path (str): 元のinfoファイルのパス
        output_path (str): 出力するinfoファイルのパス
        random_ratio (float): 無作為選択するサンプルの比率
        rare_ratio (float): レアケースから選択するサンプルの比率
        outlier_detection_path (str): 外れ値検出結果のJSONファイルパス
        camera_type (str, optional): 使用するカメラタイプ（デフォルト: 'CAM_FRONT'）
        seed (int, optional): 乱数シードの値（デフォルト: 42）
    """
    set_seed(seed)  # シード値を設定
    
    import json
    import random

    print_log(f'Loading info from {input_path}')
    with open(input_path, 'rb') as f:
        data = pickle.load(f)

    # 入力データの構造を確認
    if 'metainfo' in data and 'data_list' in data:
        # MMDetection3D v1.1+形式
        data_list = data['data_list']
        metainfo = data['metainfo']
        new_format = True
    else:
        # 従来形式
        data_list = data.get('infos', data if isinstance(data, list) else [])
        metainfo = data.get('metadata', {})
        new_format = False

    total_samples = len(data_list)
    print_log(f'Total samples in original dataset: {total_samples}')

    # レア画像情報の読み込み
    if outlier_detection_path and osp.exists(outlier_detection_path):
        try:
            with open(outlier_detection_path, 'r') as f:
                rare_images_data = json.load(f)
                
            # データがリスト形式でない場合はリストに変換
            if not isinstance(rare_images_data, list):
                rare_images_data = [rare_images_data]
                
            print_log(f'Loaded {len(rare_images_data)} rare image entries from {outlier_detection_path}')
            
            # sample_tokenを抽出（重複を除く）
            rare_tokens = set()
            for item in rare_images_data:
                if 'sample_token' in item:
                    # JSONファイルから直接sample_tokenを取得
                    rare_tokens.add(item['sample_token'])
                    
            print_log(f'Found {len(rare_tokens)} unique sample tokens in outlier detection results')
        except json.JSONDecodeError:
            print_log(f'Error: Could not parse JSON file {outlier_detection_path}')
            rare_tokens = set()
        except Exception as e:
            print_log(f'Error processing outlier detection file: {e}')
            rare_tokens = set()
    else:
        rare_tokens = set()
        print_log(f'No outlier detection file found at {outlier_detection_path}')

    # レア画像からの選択数を計算
    num_rare_to_select = int(total_samples * rare_ratio)
    
    # レアトークンからnum_rare_to_select個をランダムに選択
    if len(rare_tokens) > num_rare_to_select:
        selected_rare_tokens = set(random.sample(list(rare_tokens), num_rare_to_select))
        print_log(f'Randomly selected {len(selected_rare_tokens)} rare tokens from {len(rare_tokens)} available tokens')
    else:
        selected_rare_tokens = rare_tokens
        print_log(f'Using all {len(selected_rare_tokens)} available rare tokens')
    
    image_files = list(selected_rare_tokens)
    
    # レアサンプルの候補を収集
    rare_candidates = []
    for i, sample in enumerate(data_list):
        # データ構造に応じて画像情報にアクセス
        if new_format:
            # MMDetection3D v1.1+形式
            if 'images' in sample and camera_type in sample['images']:
                img_path = sample['images'][camera_type].get('img_path', '')
                basename_no_ext, _ = osp.splitext(osp.basename(img_path))
                if basename_no_ext in selected_rare_tokens or osp.basename(img_path) in image_files:
                    rare_candidates.append((i, sample))
        else:
            # 古い形式（各データセットによって構造が異なる可能性あり）
            cams = sample.get('cams', {})
            if cams and camera_type in cams:
                img_path = cams[camera_type].get('data_path', '')
                if osp.basename(img_path) in image_files:
                    rare_candidates.append((i, sample))
            # NuScenes特有の構造の場合
            elif 'images' in sample and camera_type in sample['images']:
                img_path = sample['images'][camera_type].get('img_path', '')
                if osp.basename(img_path) in image_files:
                    rare_candidates.append((i, sample))
    
    # レア候補からランダムに選択
    filtered_data_list = []
    already_selected_indices = set()
    
    if rare_candidates:
        # 選択数を調整
        num_rare_to_select = min(num_rare_to_select, len(rare_candidates))
        
        # ランダムに選択
        if num_rare_to_select < len(rare_candidates):
            selected_indices = random.sample(range(len(rare_candidates)), num_rare_to_select)
            for idx in selected_indices:
                i, sample = rare_candidates[idx]
                filtered_data_list.append(sample)
                already_selected_indices.add(i)
        else:
            # すべての候補を選択
            for i, sample in rare_candidates:
                filtered_data_list.append(sample)
                already_selected_indices.add(i)
    
    print_log(f'Selected {len(filtered_data_list)} samples identified as rare by sample_token')
    selected_data_list = filtered_data_list

    # 無作為選択するサンプルの数を計算
    num_random_to_select = int(total_samples * random_ratio)
    print_log(f'Will select {num_random_to_select} random samples')

    # 未選択のサンプルインデックスを取得
    remaining_indices = [i for i in range(total_samples) if i not in already_selected_indices]

    # 無作為選択するサンプル数の調整
    num_random_to_select = min(num_random_to_select, len(remaining_indices))

    # 無作為選択
    if num_random_to_select > 0:
        random_indices = random.sample(remaining_indices, num_random_to_select)
        for i in random_indices:
            selected_data_list.append(data_list[i])

    print_log(f'Added {num_random_to_select} random samples')
    print_log(f'Total selected samples: {len(selected_data_list)}')

    # 元のデータ形式を維持して結果を保存
    if new_format:
        result = {
            'metainfo': metainfo,
            'data_list': selected_data_list
        }
    elif isinstance(data, list):
        result = selected_data_list
    else:
        # 辞書形式の場合は元の構造を維持
        result = data.copy()
        if 'infos' in result:
            result['infos'] = selected_data_list
        else:
            result = selected_data_list

    # 出力ディレクトリが存在しない場合は作成
    os.makedirs(osp.dirname(output_path), exist_ok=True)

    # 結果を保存
    print_log(f'Saving filtered dataset to {output_path}')
    with open(output_path, 'wb') as f:
        pickle.dump(result, f)

    print_log('Done!')

    return len(selected_data_list)

def main():
    parser = argparse.ArgumentParser(description='Create NuScenes subset infos')
    parser.add_argument('--input', required=False, default="./data/nuscenes/nuscenes_infos_train.pkl", help='Input info pkl file')
    parser.add_argument('--output', required=False, help='Output info pkl file (if not specified, auto-generated from input and outlier-detection)')
    parser.add_argument('--random-ratio', type=float, default=0.1,
                        help='Ratio of random samples to select (e.g., 0.2 for 20%)')
    parser.add_argument('--rare-ratio', type=float, default=0.1,
                        help='Ratio of rare case samples to select (e.g., 0.1 for 10%)')
    parser.add_argument('--outlier-detection', type=str, default="outlier_detection_results.json",
                        help='Path to outlier detection results JSON file')
    # 元の引数も維持（元の機能も使えるように）
    parser.add_argument('--image-folder', required=False, default="./detected_images_0416",
                        help='Folder containing detection images to filter by')
    parser.add_argument('--camera-type', type=str, default='CAM_FRONT',
                        help='Camera type to use for filtering (default: CAM_FRONT)')
    parser.add_argument('--use-folder', action='store_true',
                        help='Use the folder structure instead of the sample_token for filtering')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    args = parser.parse_args()

    # 入力ファイル名に "train" が含まれているか確認
    if "train" in args.input:
        # "train" を "val" に置換して対応する検証データセットのファイル名を生成
        val_input = args.input.replace("train", "val")
        
        # 検証データセットの出力ファイル名も同様に生成
        if args.output is None:
            # Get base names
            input_basename = os.path.basename(args.input)
            input_name = os.path.splitext(input_basename)[0]
            
            if not args.use_folder:
                outlier_basename = os.path.basename(args.outlier_detection) if args.outlier_detection else "no_outlier"
                outlier_name = os.path.splitext(outlier_basename)[0]
                random_part = f"_random{args.random_ratio}" if args.random_ratio is not None else ""
                rare_part = f"_rare{args.rare_ratio}" if args.rare_ratio is not None else ""
                output_dir = os.path.dirname(args.input)
                args.output = os.path.join(output_dir, f"{input_name}_{outlier_name}{random_part}{rare_part}.pkl")
                val_output = args.output.replace("train", "val")
            else:
                image_folder_basename = os.path.basename(args.image_folder.rstrip('/'))
                random_part = f"_random{args.random_ratio}" if args.random_ratio > 0 else ""
                rare_part = f"_rare{args.rare_ratio}" if args.rare_ratio > 0 else ""
                output_dir = os.path.dirname(args.input)
                args.output = os.path.join(output_dir, f"{input_name}_{image_folder_basename}{random_part}{rare_part}.pkl")
                val_output = args.output.replace("train", "val")
            
            print_log(f"Auto-generated output path for train: {args.output}")
            print_log(f"Auto-generated output path for val: {val_output}")
        else:
            val_output = args.output.replace("train", "val")
            print_log(f"Output path for val dataset: {val_output}")
        
        # trainデータセットの処理
        print_log("Processing train dataset...")
        if not args.use_folder:
            create_subset_infos(
                args.input,
                args.output,
                args.random_ratio,
                args.rare_ratio,
                args.outlier_detection,
                args.camera_type,
                args.seed
            )
        else:
            create_filtered_subset(
                args.input,
                args.output,
                args.image_folder,
                args.camera_type,
                args.random_ratio,
                args.rare_ratio,
                args.seed
            )
        
        # valデータセットの処理
        print_log("Processing val dataset...")
        if not args.use_folder:
            create_subset_infos(
                val_input,
                val_output,
                args.random_ratio,
                args.rare_ratio,
                args.outlier_detection,
                args.camera_type,
                args.seed
            )
        else:
            create_filtered_subset(
                val_input,
                val_output,
                args.image_folder,
                args.camera_type,
                args.random_ratio,
                args.rare_ratio,
                args.seed
            )
    else:
        # "train" が含まれていない場合は、通常の処理
        if args.output is None:
            input_basename = os.path.basename(args.input)
            input_name = os.path.splitext(input_basename)[0]
            if not args.use_folder:
                outlier_basename = os.path.basename(args.outlier_detection) if args.outlier_detection else "no_outlier"
                outlier_name = os.path.splitext(outlier_basename)[0]
                random_part = f"_random{args.random_ratio}" if args.random_ratio is not None else ""
                rare_part = f"_rare{args.rare_ratio}" if args.rare_ratio is not None else ""
                output_dir = os.path.dirname(args.input)
                args.output = os.path.join(output_dir, f"{input_name}_{outlier_name}{random_part}{rare_part}.pkl")
            else:
                image_folder_basename = os.path.basename(args.image_folder.rstrip('/'))
                random_part = f"_random{args.random_ratio}" if args.random_ratio > 0 else ""
                rare_part = f"_rare{args.rare_ratio}" if args.rare_ratio > 0 else ""
                output_dir = os.path.dirname(args.input)
                args.output = os.path.join(output_dir, f"{input_name}_{image_folder_basename}{random_part}{rare_part}.pkl")
            print_log(f"Auto-generated output path: {args.output}")

        if not args.use_folder:
            create_subset_infos(
                args.input,
                args.output,
                args.random_ratio,
                args.rare_ratio,
                args.outlier_detection,
                args.camera_type,
                args.seed
            )
        else:
            create_filtered_subset(
                args.input,
                args.output,
                args.image_folder,
                args.camera_type,
                args.random_ratio,
                args.rare_ratio,
                args.seed
            )

if __name__ == '__main__':
    main()
