#!/usr/bin/env python

"""
指定されたカテゴリ（top_concept）やアウトライア状態に基づいて
データセットのサブセットを作成するスクリプト
JSONファイルからのレアサンプルが足りない場合は、
2つ目の画像フォルダからサンプルを追加収集します
"""

import os
import pickle
import argparse
import random
import glob
import numpy as np
from os import path as osp
from mmengine import print_log
import json
import shutil  # 画像ファイルのコピーに使用
import datetime  # 日付の取得に使用
import os

def set_seed(seed=42):
    """再現性のための乱数シード設定"""
    random.seed(seed)
    np.random.seed(seed)

def create_filtered_subset_by_category(
    input_path, 
    output_path, 
    detection_results_path,
    second_folder_path=None,  # 追加: 2つ目の画像フォルダのパス
    target_concepts=None,
    include_final_outliers=True,
    camera_type='CAM_FRONT',
    random_ratio=0.0,
    rare_ratio=1.0,
    seed=42,
    visualize_folder=None  # 追加: 可視化用フォルダのパス
):
    """データセットから、指定されたカテゴリやアウトライア状態に基づいてサンプルを抽出する
    JSONからのレアサンプルが足りない場合は、2つ目のフォルダから画像を追加収集する

    Args:
        input_path (str): 元のinfoファイルのパス
        output_path (str): 出力するinfoファイルのパス
        detection_results_path (str): 検出結果のJSONファイルパス
        second_folder_path (str, optional): 2つ目の画像フォルダのパス（デフォルト: None）
        target_concepts (list): 選択対象のtop_conceptリスト（例: ["truck", "bicycle"]）
        include_final_outliers (bool): is_final_outlierがtrueのサンプルも含めるか
        camera_type (str): 使用するカメラタイプ（デフォルト: 'CAM_FRONT'）
        random_ratio (float): 無作為選択するサンプルの比率（デフォルト: 0.0）
        rare_ratio (float): レアケースから選択するサンプルの比率（デフォルト: 1.0）
        seed (int): 乱数シードの値（デフォルト: 42）
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
    
    # 検出結果JSONファイルを読み込む
    print_log(f'Loading detection results from {detection_results_path}')
    with open(detection_results_path, 'r') as f:
        detection_results = json.load(f)
    if 'final_results' in detection_results:
        detection_results = detection_results['final_results']
    # 対象となるサンプルトークンを収集
    target_sample_tokens = set()
    second_target_tokens = set()  # second_targets用の別セット
    
    # 指定されたカテゴリに基づいてサンプルトークンを収集
    second_targets = ['truck', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier', 'trailer', 'construction_vehicle', 'pedestrian_with_bicycle']
    other_concepts = set()
    print_log(f'Collecting sample tokens for concepts: {target_concepts}')
    for item in detection_results:
        if 'top_concept' in item and 'sample_token' in item:
            if item['top_concept'] in target_concepts:
                # メインターゲットは無条件で追加
                target_sample_tokens.add(item['sample_token'])
            elif item['top_concept'] in second_targets:
                if 'is_final_outlier' in item and item['is_final_outlier']:
                # second_targetsはトークンのみ保存
                    second_target_tokens.add(item['sample_token'])
            elif 'is_final_outlier' in item and item['is_final_outlier']:
                other_concepts.append(item['top_concept'])
    print_log(f'Other concepts: {other_concepts}')

    # second_target_tokensをソート
    sorted_second_targets = sorted(list(second_target_tokens))
    
    # 連続を避けながら1/3を選択
    selected_second_targets = []
    interval = 3  # 選択間隔（調整可能）
    
    for i in range(0, len(sorted_second_targets), interval):
        selected_second_targets.append(sorted_second_targets[i])
    
    # 選択したsecond_targetsから1/3をランダムに選択
    num_to_select = len(selected_second_targets) // 2
    if num_to_select > 0:
        selected_second_targets = random.sample(selected_second_targets, num_to_select)
        target_sample_tokens.update(selected_second_targets)

    print_log(f'Selected {len(target_sample_tokens)} primary targets and {len(selected_second_targets)} secondary targets')
    
    # 対象サンプルトークンに基づいてサンプルをフィルタリング
    rare_candidate_list = []  # レアサンプルの候補リスト
    already_selected_indices = set()
    selected_image_files = []  # 選択された画像ファイル名を記録
    
    # まずレアサンプルの候補を全て収集
    for i, sample in enumerate(data_list):
        # データ構造に応じて画像情報にアクセス
        sample_token = None
        img_name = None
        if new_format:
            if 'images' in sample and camera_type in sample['images']:
                img_path = sample['images'][camera_type].get('img_path', '')
                img_name = osp.basename(img_path)
                sample_token = osp.splitext(img_name)[0]
                
                
        else:
            # 古い形式（各データセットによって構造が異なる可能性あり）
            cams = sample.get('cams', {})
            if cams and camera_type in cams:
                img_path = cams[camera_type].get('data_path', '')
                img_name = osp.basename(img_path)
                sample_token = osp.splitext(img_name)[0]
            # NuScenes特有の構造の場合
            elif 'images' in sample and camera_type in sample['images']:
                img_path = sample['images'][camera_type].get('img_path', '')
                img_name = osp.basename(img_path)
                sample_token = osp.splitext(img_name)[0]
            
            # デバッグ情報を追加
            if sample_token in target_sample_tokens and not (sample_token and img_name):
                print_log(f'Found token {sample_token} but failed to get valid image info')
        
        # サンプルトークンが対象リストに含まれていれば候補に追加
        if sample_token and sample_token in target_sample_tokens:
            rare_candidate_list.append((i, sample, img_name))
    
    # レア画像からの選択数を計算
    num_rare_to_select = int(total_samples * rare_ratio)
    print_log(f'Will select up to {num_rare_to_select} rare samples from {len(rare_candidate_list)} candidates')
    
    # レアサンプルをランダムに選択
    filtered_data_list = []
    if rare_candidate_list:
        num_to_select = min(num_rare_to_select, len(rare_candidate_list))
        selected_rare = random.sample(rare_candidate_list, num_to_select)
        
        for i, sample, img_name in selected_rare:
            filtered_data_list.append(sample)
            already_selected_indices.add(i)
            if img_name:
                selected_image_files.append(img_name)
    
    print_log(f'Randomly selected {len(filtered_data_list)} rare samples')
    
    # レアサンプルが足りない場合、不足分をrandom_ratioに上乗せ
    shortage = num_rare_to_select - len(filtered_data_list)
    adjusted_random_ratio = random_ratio
     # JSONからのレアサンプルが足りない場合、2つ目のフォルダから追加
    second_folder_selected_images = []  # 第2フォルダから選択された画像ファイル名
    if second_folder_path and len(filtered_data_list) < num_rare_to_select:
        remaining_to_select = num_rare_to_select - len(filtered_data_list)
        print_log(f'Need {remaining_to_select} more samples, selecting from second folder: {second_folder_path}')
        
        # 2つ目のフォルダ内の画像ファイルリストを取得
        second_image_files = set()
        image_file_paths = {}  # 画像ファイル名とそのフルパスのマッピング
        
        for ext in ['jpg', 'png', 'jpeg']:
            for file_path in glob.glob(osp.join(second_folder_path, f'*.{ext}')):
                file_name = osp.basename(file_path)
                second_image_files.add(file_name)
                image_file_paths[file_name] = file_path
        
        print_log(f'Found {len(second_image_files)} images in second folder')
        
        # 既に選択された画像を除外
        second_image_files = second_image_files - set(selected_image_files)
        print_log(f'After removing already selected images, {len(second_image_files)} images remain')
        
        # 2つ目のフォルダの画像を含むサンプルのインデックスを収集
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
        
        print_log(f'Added {num_to_select} samples from second folder')
        
        # 選択されたサンプル数が目標に達しなかった場合の警告
        if len(filtered_data_list) < num_rare_to_select:
            print_log(f'Warning: Could only select {len(filtered_data_list)} samples, which is less than the target {num_rare_to_select}')
    
    selected_data_list = filtered_data_list
    print_log(f'Total rare samples selected: {len(selected_data_list)}')
    
    # 無作為選択するサンプルの数を計算
    num_random_to_select = int(total_samples * random_ratio)
    num_random_to_select += (num_rare_to_select - len(filtered_data_list))
    print_log(f'Will select {num_random_to_select} random samples')
    
    # 未選択のサンプルインデックスを取得
    remaining_indices = [i for i in range(total_samples) if i not in already_selected_indices]
    
    # 無作為選択するサンプル数の調整
    num_random_to_select = min(num_random_to_select, len(remaining_indices))
    
    # 無作為選択
    # selected_data_list = []
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
    
    # 可視化用フォルダが指定されている場合、選択された画像をコピー
    if visualize_folder:
        # 可視化用フォルダを作成
        os.makedirs(visualize_folder, exist_ok=True)
        
        # JSONから選択された画像用のサブフォルダを作成
        json_folder = osp.join(visualize_folder, 'json_selected_images')
        os.makedirs(json_folder, exist_ok=True)
        
        # JSONから選択された画像をコピー
        if selected_image_files:
            print_log(f'Copying {len(selected_image_files)} images selected from JSON to {json_folder}')
            
            # データセット内の画像パスを取得するための関数
            def get_dataset_image_path(img_name):
                # データセット内の画像パスを推測
                dataset_dir = osp.dirname(osp.dirname(input_path))
                possible_img_dirs = [
                    osp.join(dataset_dir, 'samples', camera_type),
                    osp.join(dataset_dir, 'images'),
                    osp.join(dataset_dir, 'img'),
                ]
                
                for img_dir in possible_img_dirs:
                    for ext in ['jpg', 'png', 'jpeg']:
                        img_path = osp.join(img_dir, f"{osp.splitext(img_name)[0]}.{ext}")
                        if osp.exists(img_path):
                            return img_path
                
                return None
            
            # JSONから選択された画像をコピー
            copied_count = 0
            for img_name in selected_image_files:
                src_path = get_dataset_image_path(img_name)
                if src_path and osp.exists(src_path):
                    dst_path = osp.join(json_folder, img_name)
                    shutil.copy2(src_path, dst_path)
                    copied_count += 1
            
            print_log(f'Successfully copied {copied_count} out of {len(selected_image_files)} JSON-selected images')
        
        # 第2フォルダから選択された画像をコピー
        if second_folder_path and second_folder_selected_images:
            second_folder_vis = osp.join(visualize_folder, 'second_folder_images')
            os.makedirs(second_folder_vis, exist_ok=True)
            print_log(f'Copying {len(second_folder_selected_images)} images from second folder to {second_folder_vis}')
            
            for img_name in second_folder_selected_images:
                if img_name in image_file_paths:
                    src_path = image_file_paths[img_name]
                    dst_path = osp.join(second_folder_vis, img_name)
                    shutil.copy2(src_path, dst_path)
        
        # 無作為選択された画像をコピー
        if random_selected_images:
            random_folder = osp.join(visualize_folder, 'random_images')
            os.makedirs(random_folder, exist_ok=True)
            print_log(f'Copying {len(random_selected_images)} randomly selected images to {random_folder}')
            
            # 無作為選択された画像をコピー
            copied_count = 0
            for img_name in random_selected_images:
                src_path = get_dataset_image_path(img_name)
                if src_path and osp.exists(src_path):
                    dst_path = osp.join(random_folder, img_name)
                    shutil.copy2(src_path, dst_path)
                    copied_count += 1
            
            print_log(f'Successfully copied {copied_count} out of {len(random_selected_images)} random images')
    
    print_log('Done!')
    return len(selected_data_list)

def main():
    parser = argparse.ArgumentParser(description='Create dataset subset based on categories and outlier status')
    parser.add_argument('--input', required=False, default="./data/nuscenes/nuscenes_infos_train.pkl", 
                        help='Input info pkl file')
    parser.add_argument('--output', required=False, 
                        help='Output info pkl file (if not specified, auto-generated from input)')
    parser.add_argument('--detection-results', required=False, default="outlier_detection_results/outlier_detection_0420_10-17-15_JST/detection_results.json",
                        help='Path to detection results JSON file')
    parser.add_argument('--second-folder', required=False, default=None,
                        help='Second folder containing additional images to filter by (used if needed)')
    parser.add_argument('--target-concepts', nargs='+', default=["motorcycle", "bicycle", "pedestrian_with_bicycle"],
                        help='List of target top_concept values to select (e.g., --target-concepts truck bicycle)')
    parser.add_argument('--include-final-outliers', action='store_true', default=True,
                        help='Include samples with is_final_outlier=true')
    parser.add_argument('--random-ratio', type=float, default=0.1,
                        help='Ratio of random samples to select (e.g., 0.2 for 20%)')
    parser.add_argument('--rare-ratio', type=float, default=0.1,
                        help='Ratio of rare case samples to select (e.g., 1.0 for 100%)')
    parser.add_argument('--camera-type', type=str, default='CAM_FRONT',
                        help='Camera type to use for filtering (default: CAM_FRONT)')
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
            # 出力ファイル名を自動生成
            input_basename = os.path.basename(args.input)
            input_name = os.path.splitext(input_basename)[0]
            
            # 検出結果ファイル名から情報を抽出
            detection_basename = os.path.basename(args.detection_results)
            detection_name = os.path.splitext(detection_basename)[0]
            
            # ターゲットコンセプトを文字列化
            concepts_str = "_".join(args.target_concepts) if args.target_concepts else "no_concepts"
            
            # アウトライア情報
            outlier_str = "_with_outliers" if args.include_final_outliers else ""
            
            # 2つ目のフォルダ情報
            second_folder_str = ""
            if args.second_folder:
                second_folder_basename = os.path.basename(args.second_folder.rstrip('/'))
                second_folder_str = f"_with_{second_folder_basename}"
            else:
                # second folderはdetection_resultsがあるdirの、final_outlierというフォルダ
                args.second_folder = os.path.join(os.path.dirname(args.detection_results), "final_outliers")
            # ランダム比率と希少比率
            random_part = f"_random{args.random_ratio}" if args.random_ratio > 0 else ""
            rare_part = f"_rare{args.rare_ratio}" if args.rare_ratio < 1.0 else ""
            
            # 出力パスを生成
            output_dir = os.path.dirname(args.input)
            today = datetime.datetime.now().strftime("%Y%m%d")
            # args.output = os.path.join(output_dir, f"{input_name}_{detection_name}_{concepts_str}{outleier_str}{second_folder_str}{random_part}{rare_part}.pkl")
            args.output = os.path.join(output_dir, f"{input_name}_{today}_{random_part}{rare_part}.pkl")
            val_output = args.output.replace("train", "val")
            
            print_log(f"Auto-generated output path for train: {args.output}")
            print_log(f"Auto-generated output path for val: {val_output}")
        else:
            val_output = args.output.replace("train", "val")
            print_log(f"Output path for val dataset: {val_output}")
        
        # trainデータセットの処理
        print_log("Processing train dataset...")
        create_filtered_subset_by_category(
            args.input,
            args.output,
            args.detection_results,
            args.second_folder,
            args.target_concepts,
            args.include_final_outliers,
            args.camera_type,
            args.random_ratio,
            args.rare_ratio,
            args.seed,
            args.visualize
        )
        
        # valデータセットの処理
        # print_log("Processing val dataset...")
        # create_filtered_subset_by_category(
        #     val_input,
        #     val_output,
        #     args.detection_results,
        #     args.second_folder,
        #     args.target_concepts,
        #     args.include_final_outliers,
        #     args.camera_type,
        #     args.random_ratio,
        #     args.rare_ratio,
        #     args.seed,
        #     args.visualize
        # )
    else:
        # "train" が含まれていない場合は、通常の処理
        if args.output is None:
            # 出力ファイル名を自動生成
            input_basename = os.path.basename(args.input)
            input_name = os.path.splitext(input_basename)[0]
            
            # 検出結果ファイル名から情報を抽出
            detection_basename = os.path.basename(args.detection_results)
            detection_name = os.path.splitext(detection_basename)[0]
            
            # ターゲットコンセプトを文字列化
            concepts_str = "_".join(args.target_concepts) if args.target_concepts else "no_concepts"
            
            # アウトライア情報
            outlier_str = "_with_outliers" if args.include_final_outliers else ""
            
            # 2つ目のフォルダ情報
            second_folder_str = ""
            if args.second_folder:
                second_folder_basename = os.path.basename(args.second_folder.rstrip('/'))
                second_folder_str = f"_with_{second_folder_basename}"
            
            # ランダム比率と希少比率
            random_part = f"_random{args.random_ratio}" if args.random_ratio > 0 else ""
            rare_part = f"_rare{args.rare_ratio}" if args.rare_ratio < 1.0 else ""
            
            # 出力パスを生成
            output_dir = os.path.dirname(args.input)
            # 今日の日付を取得
            today = datetime.datetime.now().strftime("%Y%m%d")
            # 出力ディレクトリを作成
            args.output = os.path.join(output_dir, f"{input_name}_{today}_{random_part}{rare_part}.pkl")
            print_log(f"Auto-generated output path: {args.output}")

        create_filtered_subset_by_category(
            args.input,
            args.output,
            args.detection_results,
            args.second_folder,
            args.target_concepts,
            args.include_final_outliers,
            args.camera_type,
            args.random_ratio,
            args.rare_ratio,
            args.seed,
            args.visualize
        )

if __name__ == '__main__':
    main() 