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
import shutil  # 画像ファイルのコピーに使用
import json

def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)

def create_filtered_subset(input_path, 
                          output_path, 
                          image_folder_path,
                          second_folder_path=None,  # 追加: 2つ目の画像フォルダのパス
                          camera_type='CAM_FRONT',
                          random_ratio=0.0,
                          rare_ratio=1.0,
                          seed=42,
                          visualize_folder=None):  # 追加: 可視化用フォルダのパス
    """データセットの中から、指定されたイメージフォルダ内の画像を含むサンプルのみを抽出する

    Args:
        input_path (str): 元のinfoファイルのパス
        output_path (str): 出力するinfoファイルのパス
        image_folder_path (str): 画像ファイルが含まれるフォルダのパス
        second_folder_path (str, optional): 2つ目の画像フォルダのパス（デフォルト: None）
        camera_type (str, optional): 使用するカメラタイプ（デフォルト: 'CAM_FRONT'）
        random_ratio (float, optional): 無作為選択するサンプルの比率（デフォルト: 0.0）
        rare_ratio (float, optional): レアケースから選択するサンプルの比率（デフォルト: 1.0）
        seed (int, optional): 乱数シードの値（デフォルト: 42）
        visualize_folder (str, optional): 選択された画像をコピーする可視化用フォルダのパス（デフォルト: None）
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
    image_file_paths = {}  # 画像ファイル名とそのフルパスのマッピング
    for ext in ['jpg', 'png', 'jpeg']:
        for file_path in glob.glob(osp.join(image_folder_path, f'*.{ext}')):
            file_name = osp.basename(file_path)
            image_files.add(file_name)
            image_file_paths[file_name] = file_path
    
    print_log(f'Found {len(image_files)} images in {image_folder_path}')
    
    # 第2フォルダの画像リストを取得（指定された場合）
    second_image_files = set()
    if second_folder_path:
        for ext in ['jpg', 'png', 'jpeg']:
            for file_path in glob.glob(osp.join(second_folder_path, f'*.{ext}')):
                file_name = osp.basename(file_path)
                second_image_files.add(file_name)
                # 重複しない場合のみマッピングに追加
                if file_name not in image_file_paths:
                    image_file_paths[file_name] = file_path
        
        print_log(f'Found {len(second_image_files)} images in {second_folder_path}')
        
        # 第1フォルダと第2フォルダで重複する画像を削除
        duplicate_count = len(second_image_files.intersection(image_files))
        if duplicate_count > 0:
            print_log(f'Found {duplicate_count} duplicate images between folders, removing duplicates from second folder')
            second_image_files = second_image_files - image_files
            print_log(f'Remaining {len(second_image_files)} unique images in second folder')
    
    # 画像パスに基づいてサンプルをフィルタリング
    filtered_data_list = []
    already_selected_indices = set()  # 選択されたサンプルのインデックスを記録
    selected_image_files = []  # 選択された画像ファイル名を記録
    
    # レア画像からの選択数を計算
    num_rare_to_select = int(total_samples * rare_ratio)
    print_log(f'Will select up to {num_rare_to_select} rare samples')
    
    # 第1フォルダからの選択
    for i, sample in enumerate(data_list):
        if len(filtered_data_list) >= num_rare_to_select:  # 選択数の上限に達したらbreak
            break
            
        # データ構造に応じて画像情報にアクセス
        if new_format:
            # MMDetection3D v1.1+形式
            if 'images' in sample and camera_type in sample['images']:
                img_path = sample['images'][camera_type].get('img_path', '')
                img_name = osp.basename(img_path)
                if img_name in image_files:
                    filtered_data_list.append(sample)
                    already_selected_indices.add(i)  # 選択されたインデックスを記録
                    selected_image_files.append(img_name)  # 選択された画像ファイル名を記録
        else:
            # 古い形式（各データセットによって構造が異なる可能性あり）
            cams = sample.get('cams', {})
            if cams and camera_type in cams:
                img_path = cams[camera_type].get('data_path', '')
                img_name = osp.basename(img_path)
                if img_name in image_files:
                    filtered_data_list.append(sample)
                    already_selected_indices.add(i)  # 選択されたインデックスを記録
                    selected_image_files.append(img_name)  # 選択された画像ファイル名を記録
            # NuScenes特有の構造の場合
            elif 'images' in sample and camera_type in sample['images']:
                img_path = sample['images'][camera_type].get('img_path', '')
                img_name = osp.basename(img_path)
                if img_name in image_files:
                    filtered_data_list.append(sample)
                    already_selected_indices.add(i)  # 選択されたインデックスを記録
                    selected_image_files.append(img_name)  # 選択された画像ファイル名を記録
    
    print_log(f'Selected {len(filtered_data_list)} samples from first image folder')
    
    # 第2フォルダからの選択（もしまだ目標数に達していない場合）
    second_folder_selected_images = []  # 第2フォルダから選択された画像ファイル名
    if second_folder_path and len(filtered_data_list) < num_rare_to_select:
        remaining_to_select = num_rare_to_select - len(filtered_data_list)
        print_log(f'Need {remaining_to_select} more samples, selecting from second image folder')
        
        # 第2フォルダの画像を含むサンプルのインデックスを収集
        second_folder_indices = []
        second_folder_img_names = {}  # インデックスと画像ファイル名のマッピング
        for i, sample in enumerate(data_list):
            if i in already_selected_indices:
                continue  # すでに選択されたサンプルはスキップ
                
            # データ構造に応じて画像情報にアクセス
            if new_format:
                # MMDetection3D v1.1+形式
                if 'images' in sample and camera_type in sample['images']:
                    img_path = sample['images'][camera_type].get('img_path', '')
                    img_name = osp.basename(img_path)
                    if img_name in second_image_files:
                        second_folder_indices.append(i)
                        second_folder_img_names[i] = img_name
            else:
                # 古い形式（各データセットによって構造が異なる可能性あり）
                cams = sample.get('cams', {})
                if cams and camera_type in cams:
                    img_path = cams[camera_type].get('data_path', '')
                    img_name = osp.basename(img_path)
                    if img_name in second_image_files:
                        second_folder_indices.append(i)
                        second_folder_img_names[i] = img_name
                # NuScenes特有の構造の場合
                elif 'images' in sample and camera_type in sample['images']:
                    img_path = sample['images'][camera_type].get('img_path', '')
                    img_name = osp.basename(img_path)
                    if img_name in second_image_files:
                        second_folder_indices.append(i)
                        second_folder_img_names[i] = img_name
        
        # 必要数をランダムに選択
        num_to_select = min(remaining_to_select, len(second_folder_indices))
        if num_to_select > 0:
            selected_indices = random.sample(second_folder_indices, num_to_select)
            
            for i in selected_indices:
                filtered_data_list.append(data_list[i])
                already_selected_indices.add(i)
                if i in second_folder_img_names:
                    second_folder_selected_images.append(second_folder_img_names[i])
        
        print_log(f'Added {num_to_select} randomly selected samples from second image folder')
        
        # 選択されたサンプル数が目標に達しなかった場合の警告
        if len(filtered_data_list) < num_rare_to_select:
            print_log(f'Warning: Could only select {len(filtered_data_list)} samples, which is less than the target {num_rare_to_select}')
    
    selected_data_list = filtered_data_list
    print_log(f'Total rare samples selected: {len(selected_data_list)}')
    
    # 無作為選択するサンプルの数を計算
    num_random_to_select = int(total_samples * random_ratio)
    print_log(f'Will select {num_random_to_select} random samples')
    
    # 未選択のサンプルインデックスを取得
    remaining_indices = [i for i in range(total_samples) if i not in already_selected_indices]
    
    # 無作為選択するサンプル数の調整
    num_random_to_select = min(num_random_to_select, len(remaining_indices))
    
    # 無作為選択
    random_selected_images = []  # 無作為選択された画像ファイル名
    if num_random_to_select > 0:
        random_indices = random.sample(remaining_indices, num_random_to_select)
        for i in random_indices:
            selected_data_list.append(data_list[i])
            # 画像ファイル名を記録
            sample = data_list[i]
            if new_format and 'images' in sample and camera_type in sample['images']:
                img_path = sample['images'][camera_type].get('img_path', '')
                random_selected_images.append(osp.basename(img_path))
            elif not new_format:
                if 'cams' in sample and camera_type in sample['cams']:
                    img_path = sample['cams'][camera_type].get('data_path', '')
                    random_selected_images.append(osp.basename(img_path))
                elif 'images' in sample and camera_type in sample['images']:
                    img_path = sample['images'][camera_type].get('img_path', '')
                    random_selected_images.append(osp.basename(img_path))
    
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
    
    # 可視化用フォルダが指定されている場合、選択された画像をコピー
    if visualize_folder:
        # 可視化用フォルダを作成
        os.makedirs(visualize_folder, exist_ok=True)
        
        # レア画像用のサブフォルダを作成
        rare_folder = osp.join(visualize_folder, 'rare_images')
        os.makedirs(rare_folder, exist_ok=True)
        
        # 第1フォルダから選択された画像をコピー
        print_log(f'Copying {len(selected_image_files)} images from first folder to {rare_folder}')
        for img_name in selected_image_files:
            if img_name in image_file_paths:
                src_path = image_file_paths[img_name]
                dst_path = osp.join(rare_folder, img_name)
                shutil.copy2(src_path, dst_path)
        
        # 第2フォルダから選択された画像をコピー
        if second_folder_path and second_folder_selected_images:
            second_rare_folder = osp.join(visualize_folder, 'second_rare_images')
            os.makedirs(second_rare_folder, exist_ok=True)
            print_log(f'Copying {len(second_folder_selected_images)} images from second folder to {second_rare_folder}')
            for img_name in second_folder_selected_images:
                if img_name in image_file_paths:
                    src_path = image_file_paths[img_name]
                    dst_path = osp.join(second_rare_folder, img_name)
                    shutil.copy2(src_path, dst_path)
        
        # 無作為選択された画像をコピー
        if random_selected_images:
            random_folder = osp.join(visualize_folder, 'random_images')
            os.makedirs(random_folder, exist_ok=True)
            print_log(f'Copying {len(random_selected_images)} randomly selected images to {random_folder}')
            
            # データセット内の画像パスを取得するための関数
            def get_dataset_image_path(img_name):
                # データセット内の画像パスを推測
                dataset_dir = osp.dirname(osp.dirname(input_path))
                possible_img_dirs = [
                    osp.join(dataset_dir, 'samples', 'CAM_FRONT'),
                    osp.join(dataset_dir, 'images'),
                    osp.join(dataset_dir, 'img'),
                ]
                
                for img_dir in possible_img_dirs:
                    for ext in ['jpg', 'png', 'jpeg']:
                        img_path = osp.join(img_dir, f"{osp.splitext(img_name)[0]}.{ext}")
                        if osp.exists(img_path):
                            return img_path
                
                return None
            
            # 無作為選択された画像をコピー
            copied_count = 0
            for img_name in random_selected_images:
                # まず、既知の画像パスから探す
                if img_name in image_file_paths:
                    src_path = image_file_paths[img_name]
                else:
                    # データセット内から画像パスを推測して探す
                    src_path = get_dataset_image_path(img_name)
                
                if src_path and osp.exists(src_path):
                    dst_path = osp.join(random_folder, img_name)
                    shutil.copy2(src_path, dst_path)
                    copied_count += 1
            
            print_log(f'Successfully copied {copied_count} out of {len(random_selected_images)} random images')
    
    return len(selected_data_list)

def create_subset_infos(
    input_path, 
    output_path, 
    random_ratio=0.0, 
    rare_ratio=1.0, 
    outlier_detection_path=None,
    camera_type='CAM_FRONT',
    seed=42
):
    """アウトライア検出結果に基づいてデータセットのサブセットを作成する
    
    Args:
        input_path (str): 元のinfoファイルのパス
        output_path (str): 出力するinfoファイルのパス
        random_ratio (float): 無作為選択するサンプルの比率
        rare_ratio (float): レアケースから選択するサンプルの比率
        outlier_detection_path (str): アウトライア検出結果のJSONファイルパス
        camera_type (str): 使用するカメラタイプ
        seed (int): 乱数シード
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
    
    # アウトライア検出結果を読み込む
    rare_image_files = set()
    if outlier_detection_path and os.path.exists(outlier_detection_path):
        print_log(f'Loading outlier detection results from {outlier_detection_path}')
        try:
            with open(outlier_detection_path, 'r') as f:
                outlier_results = json.load(f)
            
            # 結果の形式に応じて処理
            if isinstance(outlier_results, list):
                # リスト形式の場合
                for item in outlier_results:
                    if isinstance(item, dict) and 'original_path' in item:
                        img_path = item['original_path']
                        img_name = osp.basename(img_path)
                        rare_image_files.add(img_name)
            elif isinstance(outlier_results, dict):
                # 辞書形式の場合
                for key, value in outlier_results.items():
                    if 'original_path' in value:
                        img_path = value['original_path']
                        img_name = osp.basename(img_path)
                        rare_image_files.add(img_name)
            
            print_log(f'Found {len(rare_image_files)} rare images in outlier detection results')
        except Exception as e:
            print_log(f'Error loading outlier detection results: {e}')
            rare_image_files = set()
    else:
        print_log('No outlier detection results provided or file not found')
    
    # レア画像からの選択数を計算
    num_rare_to_select = int(total_samples * rare_ratio)
    print_log(f'Will select up to {num_rare_to_select} rare samples')
    
    # レア画像を含むサンプルを選択
    filtered_data_list = []
    already_selected_indices = set()
    
    for i, sample in enumerate(data_list):
        if len(filtered_data_list) >= num_rare_to_select:
            break
            
        # データ構造に応じて画像情報にアクセス
        img_name = None
        if new_format:
            if 'images' in sample and camera_type in sample['images']:
                img_path = sample['images'][camera_type].get('img_path', '')
                img_name = osp.basename(img_path)
        else:
            cams = sample.get('cams', {})
            if cams and camera_type in cams:
                img_path = cams[camera_type].get('data_path', '')
                img_name = osp.basename(img_path)
            elif 'images' in sample and camera_type in sample['images']:
                img_path = sample['images'][camera_type].get('img_path', '')
                img_name = osp.basename(img_path)
        
        if img_name and img_name in rare_image_files:
            filtered_data_list.append(sample)
            already_selected_indices.add(i)
    
    print_log(f'Selected {len(filtered_data_list)} rare samples')
    
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
            filtered_data_list.append(data_list[i])
    
    print_log(f'Added {num_random_to_select} random samples')
    print_log(f'Total selected samples: {len(filtered_data_list)}')
    
    # 元のデータ形式を維持して結果を保存
    if new_format:
        result = {
            'metainfo': metainfo,
            'data_list': filtered_data_list
        }
    elif isinstance(data, list):
        result = filtered_data_list
    else:
        # 辞書形式の場合は元の構造を維持
        result = data.copy()
        if 'infos' in result:
            result['infos'] = filtered_data_list
        else:
            result = filtered_data_list
    
    # 出力ディレクトリが存在しない場合は作成
    os.makedirs(osp.dirname(output_path), exist_ok=True)
    
    # 結果を保存
    print_log(f'Saving filtered dataset to {output_path}')
    with open(output_path, 'wb') as f:
        pickle.dump(result, f)
    
    print_log('Done!')
    return len(filtered_data_list)

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
    parser.add_argument('--image-folder', required=False, default="./detected_images_10fold",
                        help='Folder containing detection images to filter by')
    parser.add_argument('--second-folder', required=False, default='./detected_images_half',
                        help='Second folder containing additional images to filter by (used if needed)')
    parser.add_argument('--camera-type', type=str, default='CAM_FRONT',
                        help='Camera type to use for filtering (default: CAM_FRONT)')
    parser.add_argument('--use-folder', action='store_true',
                        help='Use the folder structure instead of the sample_token for filtering')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    parser.add_argument('--visualize', type=str, default=None,
                      help='Folder to copy selected images for visualization')
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
                # 2つ目のフォルダが指定されていたら、出力ファイル名に含める
                if args.second_folder:
                    second_folder_basename = os.path.basename(args.second_folder.rstrip('/'))
                    folder_info = f"{image_folder_basename}_{second_folder_basename}"
                else:
                    folder_info = image_folder_basename
                random_part = f"_random{args.random_ratio}" if args.random_ratio > 0 else ""
                rare_part = f"_rare{args.rare_ratio}" if args.rare_ratio > 0 else ""
                output_dir = os.path.dirname(args.input)
                args.output = os.path.join(output_dir, f"{input_name}_{folder_info}{random_part}{rare_part}.pkl")
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
                args.second_folder,
                args.camera_type,
                args.random_ratio,
                args.rare_ratio,
                args.seed,
                args.visualize  # 可視化用フォルダを渡す
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
                args.second_folder,
                args.camera_type,
                args.random_ratio,
                args.rare_ratio,
                args.seed,
                args.visualize  # 可視化用フォルダを渡す
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
                # 2つ目のフォルダが指定されていたら、出力ファイル名に含める
                image_folder_basename = os.path.basename(args.image_folder.rstrip('/'))
                if args.second_folder:
                    second_folder_basename = os.path.basename(args.second_folder.rstrip('/'))
                    folder_info = f"{image_folder_basename}_{second_folder_basename}"
                else:
                    folder_info = image_folder_basename
                random_part = f"_random{args.random_ratio}" if args.random_ratio > 0 else ""
                rare_part = f"_rare{args.rare_ratio}" if args.rare_ratio > 0 else ""
                output_dir = os.path.dirname(args.input)
                args.output = os.path.join(output_dir, f"{input_name}_{folder_info}{random_part}{rare_part}.pkl")
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
                args.second_folder,
                args.camera_type,
                args.random_ratio,
                args.rare_ratio,
                args.seed,
                args.visualize  # 可視化用フォルダを渡す
            )

if __name__ == '__main__':
    main()
