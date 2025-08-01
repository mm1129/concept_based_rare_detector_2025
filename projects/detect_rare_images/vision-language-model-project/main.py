#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from src.outlier_detection_pipeline import detect_outliers_single_folder
import os
def main():
    parser = argparse.ArgumentParser(description="Rare Image Detection Tool")
    parser.add_argument("--images-folder", default="./data_nuscenes/samples/CAM_FRONT", help="Path to folder containing input images")
    parser.add_argument("--output-dir", default="./outlier_detection_results", help="Path to folder for saving results")
    parser.add_argument("--qwen-model-size", choices=["2B", "7B"], default="2B", help="Size of Qwen model to use")
    parser.add_argument("--contamination", type=float, default=0.2, help="Outlier rate setting (0.0-1.0)")
    parser.add_argument("--target-classes", nargs="+", default=["construction_vehicle", "bicycle", "motorcycle", "trailer", "truck"], 
                        help="List of class names to focus on")
    parser.add_argument("--common-classes", nargs="+", default=["car", "pedestrian", "traffic_light", "traffic_sign"],
                        help="List of common class names (these are not treated as outliers)")
    parser.add_argument("--concept-list", nargs="+", default=None,
                        help="List of candidate labels (uses optimized list if not specified)")
    parser.add_argument("--save-crops", action="store_true", default=True,
                        help="Save cropped images")
    parser.add_argument("--save-descriptions", action="store_true", default=True,
                        help="Save description texts")
    parser.add_argument("--save-probability-plots", action="store_true", default=True,
                        help="Save probability plots")
    parser.add_argument("--cleanup-temp-files", action="store_true", default=False,
                        help="Delete temporary files after processing")
    parser.add_argument("--max-images", type=int, default=None, help="Maximum number of images to process")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--sampling-rate", type=float, default=0.5, help="Sampling rate (0.0-1.0)")
    parser.add_argument("--skip-visualization", action="store_true", default=False,
                        help="Skip t-SNE visualization (visualization is enabled if this option is not specified)")
    parser.add_argument("--skip-caption", action="store_true", default=False,
                        help="Skip caption generation")
    parser.add_argument("--parallel", action="store_true", default=False, help="Enable parallel processing")
    parser.add_argument("--workers", type=int, default=None, help="Number of workers for parallel processing")
    parser.add_argument("--use-blip", action="store_true", default=False, help="Use BLIP model")
    parser.add_argument("--process-all-blip", action="store_true", default=True, 
                        help="Enable rare class filtering when using BLIP model")
    parser.add_argument("--weight-text", type=float, default=0.5, 
                        help="Weight for text similarity (0.0-1.0) - higher values prioritize text similarity")
    parser.add_argument("--outliers-only", action="store_true", default=True, help="Process outliers only")
    parser.add_argument("--tsne-threshold", type=float, default=80.0, 
                        help="Percentile threshold for t-SNE outlier detection (0.0-100.0, default: 70.0)")
    parser.add_argument("--use-lof", action="store_true", default=False,
                        help="Use LOF for outlier detection")
    parser.add_argument("--prioritize-trucks", action="store_true", default=False,
                        help="Prioritize truck class processing")
    args = parser.parse_args()
    if args.max_images is None:
        args.output_dir = os.path.join(args.output_dir, "all_images")
        args.save_crops = False
        args.save_descriptions = False
        args.save_probability_plots = False
    
    detect_outliers_single_folder(
        images_folder=args.images_folder,
        output_dir=args.output_dir,
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
        sampling_rate=args.sampling_rate,
        skip_visualization=args.skip_visualization,
        skip_caption=args.skip_caption,
        parallel=args.parallel,
        workers=args.workers,
        use_blip=args.use_blip,
        process_all_blip=args.process_all_blip,
        weight_text=args.weight_text,
        outliers_only=args.outliers_only,
        tsne_threshold=args.tsne_threshold,
        use_lof=args.use_lof,
        prioritize_trucks=args.prioritize_trucks
    )

if __name__ == "__main__":
    main() 