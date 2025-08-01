#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import datetime
import pytz
from outlier_detection_pipeline.detector import detect_outliers_single_folder

def create_parser():
    parser = argparse.ArgumentParser(description="希少物体検出パイプライン")
    parser.add_argument("--images_folder", type=str, default="data_nuscenes/samples",
                        help="入力画像が格納されているフォルダのパス")
    parser.add_argument("--output_dir", type=str, default="outlier_detection_results",
                        help="結果を保存するフォルダのパス")
    parser.add_argument("--qwen_model_size", type=str, default="2B", choices=["2B", "7B"],
                        help="Qwen2VLモデルのサイズ")
    parser.add_argument("--contamination", type=float, default=0.1,
                        help="Isolation Forestのcontamination値（アウトライアの割合）")
    parser.add_argument("--target_classes", nargs="+", 
                        default=["construction_vehicle", "bicycle", "motorcycle", "trailer", "truck"],
                        help="特に注目するクラス名のリスト")
    parser.add_argument("--common_classes", nargs="+",
                        default=["car", "pedestrian", "traffic_light", "traffic_sign"],
                        help="一般的なクラス名のリスト")
    parser.add_argument("--concept_list", nargs="+", default=None,
                        help="候補ラベルのリスト（指定しない場合はデフォルト値を使用）")
    parser.add_argument("--save_crops", action="store_true", default=True,
                        help="切り抜き画像を保存するかどうか")
    parser.add_argument("--save_descriptions", action="store_true", default=True,
                        help="説明テキストを保存するかどうか")
    parser.add_argument("--save_probability_plots", action="store_true", default=True,
                        help="確率プロットを保存するかどうか")
    parser.add_argument("--cleanup_temp_files", action="store_true", default=False,
                        help="処理終了後に一時ファイルを削除するかどうか")
    parser.add_argument("--max_images", type=int, default=None,
                        help="処理する最大画像数（指定しない場合は全て処理）")
    parser.add_argument("--seed", type=int, default=42,
                        help="乱数シード")
    parser.add_argument("--sampling_rate", type=float, default=0.2,
                        help="サンプリング率 (0.0-1.0)")
    parser.add_argument("--skip_visualization", action="store_true", default=True,
                        help="t-SNE可視化をスキップするかどうか")
    parser.add_argument("--skip_caption", action="store_true", default=False,
                        help="キャプション生成をスキップするかどうか")
    parser.add_argument("--parallel", action="store_true", default=False,
                        help="並列処理を有効にするかどうか")
    parser.add_argument("--workers", type=int, default=None,
                        help="並列処理のワーカー数（指定しない場合は自動設定）")
    parser.add_argument("--weight_text", type=float, default=0.2,
                        help="テキスト類似度の重み（0.0〜1.0）")
    parser.add_argument("--process_all_blip", action="store_true", default=True,
                        help="BLIPモデル使用時に希少クラスフィルタリングを無効にするかどうか")
    return parser

def main():
    # コマンドライン引数の解析
    parser = create_parser()
    args = parser.parse_args()
    
    # 基本ディレクトリの取得
    base_samples_dir = args.images_folder
    
    # サンプルディレクトリ内のカメラフォルダをすべて取得
    camera_folders = ['CAM_FRONT']  # CAM_FRONTだけに固定
    print(f"処理対象のカメラフォルダ: {camera_folders}")
    
    # 各カメラフォルダに対して処理を実行
    for camera_folder in camera_folders:
        print(f"\n==== カメラフォルダ {camera_folder} の処理を開始 ====\n")
        
        # この方向のカメラの画像フォルダと出力ディレクトリを設定
        camera_images_folder = os.path.join(base_samples_dir, camera_folder)
        camera_output_dir = os.path.join(args.output_dir, camera_folder)
        
        # 出力ディレクトリが存在しない場合は作成
        os.makedirs(camera_output_dir, exist_ok=True)
        
        # このカメラフォルダに対して処理を実行
        detect_outliers_single_folder(
            images_folder=camera_images_folder,
            output_dir=camera_output_dir,
            qwen_model_size=args.qwen_model_size,
            contamination=args.contamination,
            target_classes=args.target_classes,
            common_classes=args.common_classes,
            concept_list=args.concept_list,
            save_crops=args.save_crops,
            save_descriptions=args.save_descriptions,
            save_probability_plots=args.save_probability_plots,
            cleanup_temp_files=args.cleanup_temp_files,
            max_images=args.max_images,
            seed=args.seed,
            sampling_rate=args.sampling_rate,  # 全体の1/10を処理
            skip_visualization=args.skip_visualization,
            skip_caption=args.skip_caption,
            parallel=args.parallel,
            workers=args.workers,
            process_all_blip=args.process_all_blip,
            weight_text=args.weight_text
        )
        
        print(f"\n==== カメラフォルダ {camera_folder} の処理が完了 ====\n")
    
    print("\n全カメラフォルダの処理が完了しました。\n")

if __name__ == "__main__":
    main()