#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
import random
import shutil
import datetime
import pytz
import json
import time
import signal
import sys
import concurrent.futures
from functools import partial
import multiprocessing
from collections import Counter
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.manifold import TSNE
from .feature_extractor import adaptive_concept_similarity
from .caption_generator import generate_descriptions_batch
from .feature_extractor import enhanced_ensemble_similarity

# Import internal modules
from .utils import (
    get_device, set_seed, get_optimal_batch_size,
    setup_japanese_font, signal_handler, get_optimized_concept_list, 
    is_potential_rare_class, map_to_target_class, map_class_name_to_nusc
)
from .yolo_detector import DetectorYOLO
from .feature_extractor import CLIPFeatureExtractor, parse_text_with_probability
from .outlier_detection import (
    train_isolation_forest, predict_outliers, filter_rare_classes, 
    detect_class_aware_outliers
)
from .visualization import visualize_tsne, visualize_class_tsne, plot_concept_probabilities, visualize_outliers_tsne, visualize_multiple_outliers_tsne

try:
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
except ImportError:
    print("Qwen models not found. Install via: pip install transformers")

try:
    from .caption_generator import generate_descriptions_batch, get_blip_generator
except ImportError:
    print("caption_generator module not found.")

# Avoid CUDA errors for multiprocessing
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

# Helper function for setting up signal handlers
def setup_signal_handlers():
    """Set up signal handlers"""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

def process_image_batch(file_batch, images_folder, cropped_dir, detector, batch_idx=0):
    """Process images in batch for parallel execution"""
    batch_results = []
    for file_name in file_batch:
        try:
            image_path = os.path.join(images_folder, file_name)
            cropped_info = detector.detect_and_crop(
                image_path, out_dir=cropped_dir, conf_thres=0.25
            )
            batch_results.extend(cropped_info)
        except Exception as e:
            print(f"Image processing error {file_name}: {e}")
    return batch_results

def detect_outliers_single_folder(
    images_folder,
    output_dir,
    qwen_model_size="2B",
    contamination=0.2,
    target_classes=None,
    common_classes=None,
    concept_list=None,
    save_crops=True,
    save_descriptions=True,
    save_probability_plots=True,
    cleanup_temp_files=False,
    max_images=None,
    seed=42,
    sampling_rate=0.2,
    skip_visualization=False,
    skip_caption=True,
    parallel=False,
    workers=None,
    use_blip=True,
    process_all_blip=False,
    weight_text=0.5,
    outliers_only=False,
    tsne_threshold=70.0,
    use_lof=False,
    prioritize_trucks=False
):
    """
    Detect outliers in images from a single folder
    
    Args:
        images_folder: Path to folder containing images
        output_dir: Path to folder for saving results
        qwen_model_size: Size of Qwen model ("2B" or "7B")
        contamination: IsolationForest contamination value (proportion of outliers)
        target_classes: List of class names to focus on (e.g., ["construction_vehicle", "bicycle"])
        common_classes: List of common class names (these are not treated as outliers)
        concept_list: List of candidate labels
        save_crops: Whether to save cropped images
        save_descriptions: Whether to save description texts
        save_probability_plots: Whether to save probability plots
        cleanup_temp_files: Whether to delete temporary files after processing
        max_images: Maximum number of images to process (None to process all)
        seed: Random seed
        sampling_rate: Sampling rate (0.0-1.0)
        skip_visualization: Whether to skip t-SNE visualization
        skip_caption: Whether to skip caption generation
        parallel: Whether to enable parallel processing
        workers: Number of workers for parallel processing (auto if not specified)
        use_blip: Whether to use BLIP model (False to use Qwen2VL)
        process_all_blip: Whether to disable rare class filtering when using BLIP model
        weight_text: Weight for text similarity (0.0-1.0) - higher values prioritize text similarity
        outliers_only: If True, process outliers only and ignore rare classes
        tsne_threshold: Percentile threshold for t-SNE outlier detection (0.0-100.0)
        use_lof: Whether to use LOF (False to use only Isolation Forest and t-SNE)
        prioritize_trucks: Whether to prioritize truck class processing
    """
    # Default values
    if target_classes is None:
        target_classes = ["construction_vehicle", "bicycle", "motorcycle", "trailer", "truck"]
    
    if common_classes is None:
        common_classes = ["car", "pedestrian", "traffic_light", "traffic_sign"]
        
    # Set up signal handlers
    setup_signal_handlers()
    
    # Set up fonts for plotting
    setup_japanese_font()
    
    # Set random seed
    set_seed(seed)
    
    # Record start time
    start_time_overall = time.time()
    
    # Set JST timezone and get current time
    jst = pytz.timezone('Asia/Tokyo')
    now = datetime.datetime.now(jst)
    timestamp = now.strftime("%m%d_%H-%M-%S_JST")
    
    # Create output directory
    main_out_dir = os.path.join(output_dir, f"outlier_detection_{timestamp}")
    os.makedirs(main_out_dir, exist_ok=True)
    
    # Create directory for final outliers
    final_outlier_dir = os.path.join(main_out_dir, "final_outliers")
    os.makedirs(final_outlier_dir, exist_ok=True)
    
    # Save configuration
    config = {
        "images_folder": images_folder,
        "output_dir": output_dir,
        "qwen_model_size": qwen_model_size,
        "contamination": contamination,
        "target_classes": target_classes,
        "common_classes": common_classes,
        "save_crops": save_crops,
        "save_descriptions": save_descriptions,
        "save_probability_plots": save_probability_plots,
        "cleanup_temp_files": cleanup_temp_files,
        "max_images": max_images,
        "seed": seed,
        "timestamp": timestamp,
        "device": get_device(),
        "sampling_rate": sampling_rate,
        "skip_visualization": skip_visualization,
        "skip_caption": skip_caption,
        "parallel": parallel,
        "workers": workers,
        "use_blip": use_blip,
        "process_all_blip": process_all_blip,
        "weight_text": weight_text,
        "outliers_only": outliers_only,
        "tsne_threshold": tsne_threshold,
        "use_lof": use_lof,
        "prioritize_trucks": prioritize_trucks
    }
    
    config_path = os.path.join(main_out_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Results will be saved to {main_out_dir}")
    print(f"Configuration: {config}")
    
    # Get list of image files
    all_files = [f for f in os.listdir(images_folder) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif"))]
    random.shuffle(all_files)
    
    # Limit maximum number of images
    if max_images is not None and max_images < len(all_files):
        all_files = all_files[:max_images]
        print(f"Processing target: {len(all_files)} images (maximum limit)")
    else:
        print(f"Processing target: {len(all_files)} images")
    
    # Reduce processing target by sampling
    if sampling_rate < 1.0:
        orig_count = len(all_files)
        all_files = random.sample(all_files, int(len(all_files) * sampling_rate))
        print(f"Sampling applied: {orig_count} → {len(all_files)} images (sampling rate: {sampling_rate:.1f})")
    
    # Create temporary directory
    cropped_dir = os.path.join(main_out_dir, "cropped_objects")
    os.makedirs(cropped_dir, exist_ok=True)
    
    # Initialize YOLO model
    detector = DetectorYOLO(model_path="yolov8l.pt")
    
    # 1. Object detection and cropping
    print("Executing object detection and cropping...")
    
    # Decide whether to process in parallel or sequentially
    all_cropped_info = []
    if parallel and workers is not None:
        # Parallel processing
        batch_size = max(1, len(all_files) // workers)
        batches = [all_files[i:i+batch_size] for i in range(0, len(all_files), batch_size)]
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            process_func = partial(process_image_batch, images_folder=images_folder, 
                                  cropped_dir=cropped_dir, detector=detector)
            results = list(executor.map(process_func, batches, range(len(batches))))
            
        for batch_result in results:
            all_cropped_info.extend(batch_result)
    else:
        # Sequential processing
        for idx, file_name in enumerate(all_files):
            try:
                # Progress display
                if idx % 100 == 0 or idx == len(all_files) - 1:
                    elapsed = time.time() - start_time_overall
                    progress = (idx + 1) / len(all_files) * 100
                    eta = (elapsed / (idx + 1)) * (len(all_files) - (idx + 1)) if idx > 0 else 0
                    print(f"Object detection progress: {progress:.1f}% ({idx+1}/{len(all_files)}) | Elapsed: {elapsed:.1f} seconds | Remaining: {eta:.1f} seconds")
                
                image_path = os.path.join(images_folder, file_name)
                cropped_info = detector.detect_and_crop(
                    image_path, out_dir=cropped_dir, conf_thres=0.3
                )
                all_cropped_info.extend(cropped_info)
            except Exception as e:
                print(f"Image processing error {file_name}: {e}")
                continue
    
    print(f"Detected objects: {len(all_cropped_info)}")
    
    # If no detected objects, exit
    if len(all_cropped_info) == 0:
        print("No detected objects found. Processing will end.")
        return

    if prioritize_trucks:
        # Extract truck class
        truck_cropped_info = []
        non_truck_cropped_info = []
        for info in all_cropped_info:
            class_name = info[1].lower()
            if "truck" in class_name:
                truck_cropped_info.append(info)
            else:
                non_truck_cropped_info.append(info)

        print(f"Number of truck class objects: {len(truck_cropped_info)}")
        print(f"Number of other objects: {len(non_truck_cropped_info)}")

        # Early filtering: Directly determine rare classes from YOLO detection results
        filtered_non_truck_info = []
        for info in non_truck_cropped_info:
            class_name = info[1]
            if is_potential_rare_class(class_name):
                filtered_non_truck_info.append(info)
        
        # Add a certain number of random samples (diversity assurance)
        if len(filtered_non_truck_info) < len(non_truck_cropped_info) * 0.3:
            remaining = [info for info in non_truck_cropped_info if info not in filtered_non_truck_info]
            sample_size = min(len(remaining), int(len(non_truck_cropped_info) * 0.1))
            filtered_non_truck_info.extend(random.sample(remaining, sample_size))
        
        # Combine truck and filtered other objects (truck at the top)
        all_cropped_info = truck_cropped_info + filtered_non_truck_info
        
        print(f"Processing target: {len(all_cropped_info)} objects (truck: {len(truck_cropped_info)} objects, rare classes: {len(filtered_non_truck_info)} objects)")
    else:
        # Previous processing: Early filtering
        filtered_cropped_info = []
        for info in all_cropped_info:
            class_name = info[1]
            if is_potential_rare_class(class_name):
                filtered_cropped_info.append(info)
        
        # Add a certain number of random samples (diversity assurance)
        if len(filtered_cropped_info) < len(all_cropped_info) * 0.3:
            remaining = [info for info in all_cropped_info if info not in filtered_cropped_info]
            sample_size = min(len(remaining), int(len(all_cropped_info) * 0.1))
            filtered_cropped_info.extend(random.sample(remaining, sample_size))
        
        print(f"Early filtering: {len(all_cropped_info)} objects → {len(filtered_cropped_info)} objects")
        all_cropped_info = filtered_cropped_info
    
    # 2. Initialize CLIP model
    clip_extractor = CLIPFeatureExtractor(model_name="laion/CLIP-ViT-B-32-laion2B-s34B-b79K")
    
    # 3. Feature extraction
    print("Executing feature extraction...")
    obj_paths = [info[0] for info in all_cropped_info]
    features = clip_extractor.get_image_embeddings_batch(obj_paths, batch_size=32)
    
    # Exclude objects for which feature extraction failed
    valid_indices = [i for i, feat in enumerate(features) if feat is not None]
    valid_features = features[valid_indices]
    valid_cropped_info = [all_cropped_info[i] for i in valid_indices]
    
    print(f"Number of valid feature vectors: {len(valid_features)}/{len(all_cropped_info)}")
    class_names_list = [info[1] for info in valid_cropped_info]
    class_counts = Counter(class_names_list)
    
    # 4. Visualization by t-SNE
    if not skip_visualization:
        print("Executing t-SNE visualization...")
        # Visualization by class and t-SNE dimension reduction
        class_names = [info[1] for info in valid_cropped_info]
        class_counts = Counter(class_names)
        top_classes = [cls for cls, count in class_counts.most_common(20)]
        
        # Perform t-SNE visualization and receive the reduced result
        visualization_out_dir = os.path.join(main_out_dir, "visualization")
        os.makedirs(visualization_out_dir, exist_ok=True)
        
        _, tsne_result = visualize_class_tsne(
            features=valid_features,
            class_names=class_names,
            title="t-SNE Visualization (By Class)",
            out_dir=visualization_out_dir
        )
        
        # Confirm that the reduced result is not None
        if tsne_result is not None:
            # 5-4. Outlier detection by t-SNE
            print(f"Executing outlier detection by t-SNE (threshold_percentile={tsne_threshold})...")
            tsne_outlier_labels = detect_outliers_tsne(tsne_result, n_neighbors=20, threshold_percentile=tsne_threshold)
        else:
            print("t-SNE dimension reduction failed. Skipping outlier detection by t-SNE.")
            tsne_outlier_labels = None
    else:
        print("Skipping t-SNE visualization (--skip-visualization option is specified)")
        tsne_outlier_labels = None
        tsne_result = None
    
    # 5. Outlier detection by IsolationForest
    print(f"Executing outlier detection by IsolationForest (contamination={contamination})...")
    iso_model = train_isolation_forest(valid_features, contamination=contamination)
    iso_outlier_labels = predict_outliers(valid_features, iso_model)

    # Preparation for combining multiple results
    methods_results = [iso_outlier_labels]
    methods_names = ["Isolation Forest"]

    # 5-2. Outlier detection by LOF (optional)
    lof_outlier_labels = None
    if use_lof:
        print(f"Executing outlier detection by LOF (contamination={contamination})...")
        lof_outlier_labels = detect_outliers_lof(valid_features, n_neighbors=20, contamination=contamination)
        methods_results.append(lof_outlier_labels)
        methods_names.append("LOF")

    # Add t-SNE result if available
    if tsne_outlier_labels is not None:
        methods_results.append(tsne_outlier_labels)
        methods_names.append("t-SNE")
    
    # 5-3. Combine multiple results
    methods_str = ", ".join(methods_names)
    print(f"Combining results of {methods_str}...")
    
    combined_outlier_labels = combine_outlier_results(methods_results, method='union')
    outlier_flags = (combined_outlier_labels == -1).astype(int)  # 1: outlier, 0: inlier

    # Record results before combining
    iso_outlier_flags = (iso_outlier_labels == -1).astype(int)
    
    # Record LOF result if available
    lof_outlier_flags = None
    if lof_outlier_labels is not None:
        lof_outlier_flags = (lof_outlier_labels == -1).astype(int)
    
    # Record t-SNE result if available
    tsne_outlier_flags = None
    if tsne_outlier_labels is not None:
        tsne_outlier_flags = (tsne_outlier_labels == -1).astype(int)
    
    # Display detection results
    result_str = f"Detection results: Isolation Forest {np.sum(iso_outlier_flags)} objects"
    if lof_outlier_flags is not None:
        result_str += f", LOF {np.sum(lof_outlier_flags)} objects"
    if tsne_outlier_flags is not None:
        result_str += f", t-SNE {np.sum(tsne_outlier_flags)} objects"
    result_str += f", Combined {np.sum(outlier_flags)} objects"
    print(result_str)

    # Visualization of outlier detection results by t-SNE
    if tsne_result is not None and not skip_visualization:
        print("Visualizing outlier detection results by t-SNE...")
        
        # Visualize results of each detection method separately
        outlier_flags_dict = {}
        
        # Isolation Forest
        outlier_flags_dict["Isolation Forest"] = iso_outlier_flags
        
        # LOF (if used)
        if lof_outlier_flags is not None:
            outlier_flags_dict["LOF"] = lof_outlier_flags
        
        # t-SNE (if used)
        if tsne_outlier_flags is not None:
            outlier_flags_dict["t-SNE"] = tsne_outlier_flags
        
        # Combined result
        outlier_flags_dict["Combined"] = outlier_flags
        
        # Visualize results of each method separately
        for method_name, flags in outlier_flags_dict.items():
            visualize_outliers_tsne(
                tsne_result=tsne_result,
                outlier_flags=flags,
                method_name=method_name,
                out_dir=visualization_out_dir
            )
        
        # Visualize results of all methods in one figure
        visualize_multiple_outliers_tsne(
            tsne_result=tsne_result,
            outlier_flags_dict=outlier_flags_dict,
            title="Comparison of Outlier Detection Methods",
            out_dir=visualization_out_dir
        )
    
    # Use rare class filtering as well
    rare_indices = filter_rare_classes(valid_cropped_info, class_names_list)
    
    # Determine processing target indices
    if outliers_only:
        # Process outliers only
        process_indices = list(np.where(outlier_flags == 1)[0])
        print(f"Outliers only processing mode: Rare classes will not be included")
    else:
        # Process both outliers and rare classes (previous behavior)
        process_indices = list(set(np.where(outlier_flags == 1)[0]).union(set(rare_indices)))
    
    process_indices.sort()  # Sort to maintain order
    
    print(f"Number of processing objects: {len(process_indices)} (outliers: {np.sum(outlier_flags)}, rare classes: {len(rare_indices)}{', rare classes will be excluded' if outliers_only else ''})")
    
    # Save image embeddings
    image_embeddings_dict = {}
    for i in process_indices:
        image_path = valid_cropped_info[i][0]
        image_embeddings_dict[image_path] = valid_features[i]
    
    # 11. Create dictionary to keep YOLO detected class names
    detected_classes_dict = {
        valid_cropped_info[i][0]: valid_cropped_info[i][1]
        for i in process_indices
    }
    
    # 12. List processing paths
    process_paths = [valid_cropped_info[i][0] for i in process_indices]
    
    # Initialize model for caption generation
    model = None
    processor = None
    blip_generator = None
    
    if not skip_caption:
        if use_blip:
            try:
                print("Initializing BLIP model...")
                # Initialize BLIP model only once (singleton pattern)
                blip_generator = get_blip_generator()
                if blip_generator is None:
                    print("Failed to initialize BLIP model. Falling back to Qwen2VL.")
                    use_blip = False
                else:
                    print("BLIP model initialization completed")
                    if process_all_blip:
                        print("Rare class filtering mode is disabled - Captions will be generated for all images selected by BLIP")
            except Exception as e:
                print(f"BLIP model initialization error: {e}")
                print("Falling back to Qwen2VL model")
                use_blip = False
        
        if not use_blip:
            try:
                print(f"Initializing Qwen2VL model (size: {qwen_model_size})...")
                if qwen_model_size == "2B":
                    model_id = "Qwen/Qwen2-VL-2B-Instruct"
                else:
                    model_id = "Qwen/Qwen2-VL-7B-Instruct"
                
                model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
                processor = AutoProcessor.from_pretrained(model_id)
                print("Qwen2VL model initialization completed")
            except Exception as e:
                print(f"Qwen2VL model initialization error: {e}")
                print("Skipping caption generation")
                skip_caption = True
    else:
        print("Model will not be initialized for skipping caption generation")
        
    # Prepare concept list
    if concept_list is None:
        # concept_list = get_optimized_concept_list()
        concept_list = [
            # NuScenes / MS COCO Classes
            "pedestrian", "car", "truck", "bicycle", "motorcycle", 
            "bus", "train", "traffic_light", "traffic_sign",
            "construction_vehicle", "trailer", "barrier",
            "traffic_cone", "pedestrian",
            "static_object"
        ]

    # Caption generation and final outlier determination
    if process_paths:
        print("Executing caption generation and final determination...")
        
        # Path to interim results file
        interim_results_file = os.path.join(main_out_dir, "interim_results.json")
        final_results = []
        target_class_results = {target: [] for target in target_classes}
        
        # Checkpoint file confirmation
        checkpoint_file = os.path.join(main_out_dir, "process_checkpoint.json")
        if os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, 'r') as f:
                    checkpoint_data = json.load(f)
                    processed_batches = checkpoint_data.get('processed_batches', 0)
                    final_results = checkpoint_data.get('final_results', [])
                    print(f"Checkpoint loaded: {processed_batches} batches processed ({len(final_results)} objects)")
            except Exception as e:
                print(f"Checkpoint loading error: {e}")
                processed_batches = 0
        else:
            processed_batches = 0
            
        # Sub-batch size
        sub_batch_size = 100  # Number of images to process at once
        
        # Caption processing
        for batch_idx in range(0, len(process_paths), sub_batch_size):
            # Skip already processed batches
            if batch_idx < processed_batches * sub_batch_size:
                print(f"Batch {batch_idx//sub_batch_size + 1} is already processed. Skipping...")
                continue
            
            sub_batch_paths = process_paths[batch_idx:batch_idx+sub_batch_size]
            sub_batch_indices = process_indices[batch_idx:min(batch_idx+sub_batch_size, len(process_indices))]
            
            print(f"Sub-batch processing: {batch_idx+1}〜{min(batch_idx+sub_batch_size, len(process_paths))}/{len(process_paths)}...")
            
            # Sub-batch caption generation
            if not skip_caption:
                # Set optimal batch size
                caption_batch_size = get_optimal_batch_size(initial_size=8)
                
                # Which to use BLIP/Qwen2VL
                if use_blip and blip_generator is not None:
                    sub_batch_descriptions = generate_descriptions_batch(
                        sub_batch_paths,
                        model=model,
                        processor=processor,
                        batch_size=caption_batch_size,
                        detected_classes=detected_classes_dict,
                        blip_generator=blip_generator,  # Use pre-initialized BLIP instance
                        process_all_blip=process_all_blip,  # Pass rare class filtering flag
                        use_blip=True,  # Explicitly use BLIP
                        outliers_only=outliers_only  # Propagate outliers only mode
                    )
                else:
                    sub_batch_descriptions = generate_descriptions_batch(
                        sub_batch_paths,
                        model=model,
                        processor=processor,
                        batch_size=caption_batch_size,
                        detected_classes=detected_classes_dict,
                        use_blip=False,  # Explicitly use Qwen2VL
                        outliers_only=outliers_only  # Propagate outliers only mode
                    )
            else:
                # If caption is skipped, generate pseudo caption by mapping class name to nuScenes class
                sub_batch_descriptions = []
                for path in sub_batch_paths:
                    original_class = detected_classes_dict.get(path, "")
                    mapped_classes = map_class_name_to_nusc(original_class)
                    
                    # Generate pseudo caption string
                    pseudo_caption = f"YOLO detected: {original_class}. "
                    if mapped_classes and mapped_classes[0] != original_class:
                        pseudo_caption = f"nuScenes classes: {', '.join(mapped_classes)}."
                    
                    sub_batch_descriptions.append(pseudo_caption)
            
            # Sub-batch result processing
            batch_results = []
            for i, desc in enumerate(sub_batch_descriptions):
                if i >= len(sub_batch_paths):
                    continue
                if desc is None:
                    continue
                    
                path = sub_batch_paths[i]
                
                # Normal processing
                if i >= len(sub_batch_indices):
                    continue
                idx = sub_batch_indices[i]
                info = valid_cropped_info[idx]
                is_outlier = outlier_flags[idx] == 1
                
                path, cls_name, conf, bbox, original_path = info
                
                # Extract sample token
                sample_token = os.path.basename(original_path).split('.')[0]
                
                # Debug: Output detected class name
                construction_keywords = ["truck", "dump", "concrete", "crane", "loader", "excavator", "bulldozer", "construction"]
                # if any(keyword in cls_name.lower() for keyword in construction_keywords):
                #     print(f"[DEBUG] 建設車両関連キーワード検出: クラス名='{cls_name}', 画像={os.path.basename(path)}")
                
                # Check if it belongs to specific class
                target_matches = []
                
                # 1. Direct matching - Class name completely matches target_classes
                for target in target_classes:
                    if target.lower() in cls_name.lower():
                        # print(f"[DEBUG] 直接マッチング成功: '{cls_name}' → '{target}'")
                        target_matches.append(target)
                
                # 2. Direct matching failed, use mapping function
                if not target_matches:
                    # print(f"[DEBUG] マッピング開始: クラス名='{cls_name}'")
                    mapped_target = map_to_target_class(cls_name, target_classes)
                    if mapped_target in target_classes:
                        # print(f"[DEBUG] マッピング成功: '{cls_name}' → '{mapped_target}'")
                        target_matches.append(mapped_target)
                    else:
                        print(f"[DEBUG] マッピング失敗: '{cls_name}' → '{mapped_target}' (マッピング先が target_classes に含まれていません)")
                
                # Caption analysis
                top_concept = None
                is_final_outlier = False
                probs_dict = None
                image_embedding = None  # Added to initialize variable
                
                if desc:
                    # Caption determination logic
                    desc_lower = desc.lower()
                    is_failed_caption = False
                    
                    if any(phrase in desc_lower for phrase in ["i'm sorry", "unable to", "i'm unable to", "cannot process"]):
                        is_failed_caption = True
                    
                    if not is_failed_caption:
                        # Get image embedding
                        image_embedding = image_embeddings_dict.get(path)
                        
                        # Calculate similarity (use image embedding as well)
                        if not use_blip:  # Use new similarity calculation method for Qwen2VL
                            # Use higher-level ensemble method
                            # probs_dict = enhanced_ensemble_similarity(
                            #     path,
                            #     concept_list,
                            #     clip_extractor,
                            #     desc
                            # )
                            # Use directly if desc is nuScenes class or "others"
                            nuscenes_classes = ["car", "truck", "trailer", "bus", "construction_vehicle", 
                                               "bicycle", "motorcycle", "pedestrian", "traffic_cone", "barrier", "others"]
                            
                            if desc and desc.lower().strip() in [cls.lower() for cls in nuscenes_classes]:
                                # print(f"[DEBUG] nuScenesクラス直接検出: '{desc}'")
                                top_concept = desc.lower()
                                
                                # Add to target_matches if nuScenes class is in target_classes
                                if top_concept in [target.lower() for target in target_classes]:
                                    detected_target = [t for t in target_classes if t.lower() == top_concept][0]
                                    # Always add if truck is already in target_matches
                                    if detected_target == "construction_vehicle":
                                        # If truck is already in target_matches, replace
                                        if "truck" in target_matches:
                                            target_matches.remove("truck")
                                            # print(f"[DEBUG] 'truck'を削除して'{detected_target}'に置き換えます")
                                        if detected_target not in target_matches:
                                            target_matches.append(detected_target)
                                            # print(f"[DEBUG] nuScenesクラス '{desc}' を target_matches に追加")
                                    # Other classes are not added if they are already in target_matches
                                    elif detected_target not in target_matches:
                                        target_matches.append(detected_target)
                                        # print(f"[DEBUG] nuScenesクラス '{desc}' を target_matches に追加")
                                
                                # If not in common_classes, treat as outlier
                                if top_concept not in common_classes:
                                    is_final_outlier = True
                                # probs_dict remains None
                            
                            # Continue processing
                            if not top_concept:
                                probs_dict = adaptive_concept_similarity(
                                    path,
                                    concept_list,
                                    clip_extractor,
                                    desc
                                )
                            
                            # Debug: Display top 3 concepts and score
                            # if probs_dict:
                                top_concepts = list(probs_dict.items())[:3]
                                top_concepts_str = ", ".join([f"{c}: {s:.3f}" for c, s in top_concepts])
                                print(f"[Qwen] Top concepts: {top_concepts_str}")
                        else:  # Use existing method for BLIP
                            # Use existing weighted average method
                            probs_dict = parse_text_with_probability(
                                desc, 
                                concept_list, 
                                clip_extractor, 
                                image_embedding=image_embedding,  # Pass image embedding
                                weight_text=weight_text  # Pass text similarity weight
                            )
                        
                        if probs_dict:
                            # If top 2 probability values are almost the same, treat both as top_concept
                            if len(probs_dict) >= 2:
                                top_items = list(probs_dict.items())[:2]
                                top_concept = top_items[0][0]
                                if top_concept not in common_classes:
                                    is_final_outlier = True
                                # If difference in probability values is small (e.g., 0.05 or less)
                                if abs(top_items[0][1] - top_items[1][1]) < 0.05:
                                    top_concept = f"{top_items[0][0]}, {top_items[1][0]}"
                            else:
                                top_concept = next(iter(probs_dict))
                                # Final outlier determination criteria: top_concept not in common_classes
                                if top_concept not in common_classes:
                                    is_final_outlier = True
                    else:
                        # Outlier determination for skipped caption
                        is_final_outlier = is_potential_rare_class(cls_name)
                else:
                    # If description is not obtained, determine outlier based on detected class
                    is_final_outlier = is_potential_rare_class(cls_name)
                
                # Save result
                result_info = {
                    'path': path,
                    'original_path': original_path,
                    'sample_token': sample_token,
                    'class_name': cls_name,
                    'confidence': float(conf),
                    'bbox': bbox,
                    'is_outlier': bool(is_outlier),
                    'outlier_code': int(is_outlier),
                    'top_concept': top_concept,
                    'is_final_outlier': is_final_outlier,
                    'is_yolo_potential': any(target.lower() in cls_name.lower() for target in target_classes),
                    'target_matches': target_matches,
                    'description': desc,
                    'used_image_embedding': image_embedding is not None,  # Whether image embedding is used
                    'weight_text': weight_text if image_embedding is not None else 1.0,  # Text weight
                    'outlier_source': {
                        'isolation_forest': bool(iso_outlier_flags[idx]), 
                        'lof': bool(lof_outlier_flags[idx]) if lof_outlier_flags is not None else None,
                        'tsne': bool(tsne_outlier_flags[idx]) if tsne_outlier_flags is not None else None,
                        'combined': bool(outlier_flags[idx])
                    }
                }
                
                batch_results.append(result_info)
                final_results.append(result_info)
                
                # Save results for specific class
                for target in target_matches:
                    target_class_results[target].append(result_info)
                
                # Create directory for specific class and copy image
                # Process if target_matches exist regardless of outlier status
                if target_matches:
                    for target in target_matches:
                        target_dir = os.path.join(main_out_dir, target)
                        if not os.path.exists(target_dir):
                            os.makedirs(target_dir, exist_ok=True)
                            print(f"[DEBUG] Creating target directory: {target}")
                        
                        if os.path.exists(original_path):
                            dest_path = os.path.join(target_dir, os.path.basename(original_path))
                            shutil.copy(original_path, dest_path)
                            print(f"[DEBUG] Copying image: {os.path.basename(original_path)} → {target}")
                
                # Copy file and information
                # For outliers
                if is_final_outlier:
                    dest_dir = final_outlier_dir
                
                    # Copy original image
                    if os.path.exists(original_path) and dest_dir:
                        shutil.copy(original_path, os.path.join(dest_dir, os.path.basename(original_path)))
                    
                    # Copy cropped image
                    if save_crops and os.path.exists(path):
                        shutil.copy(path, os.path.join(dest_dir, os.path.basename(path)))
                    
                    # Save description file
                    if save_descriptions:
                        txt_path = os.path.join(dest_dir, f"{os.path.splitext(os.path.basename(path))[0]}_desc.txt")
                        with open(txt_path, "w", encoding="utf-8") as f:
                            f.write(f"Sample Token: {sample_token}\n")
                            f.write(f"Is Outlier: {is_outlier}\n")
                            f.write(f"YOLO Class: {cls_name}\n")
                            f.write(f"Target Matches: {target_matches}\n")
                            f.write(f"Final outlier (Top concept not common): {is_final_outlier}\n")
                            f.write(f"Cropped Image Path: {path}\n")
                            f.write(f"Original Image Path: {original_path}\n")
                            f.write(f"Confidence={conf:.2f}, bbox={bbox}\n")
                            f.write(f"Used Image Embedding: {image_embedding is not None}\n")
                            f.write(f"Text Weight: {weight_text if image_embedding is not None else 1.0}\n\n")
                            
                            f.write("Generated caption:\n")
                            if desc is not None:
                                f.write(str(desc) + "\n\n")
                            else:
                                f.write("No caption generated.\n\n")
                            
                            if top_concept:
                                f.write(f"Top concept: {top_concept} (in common classes: {top_concept in common_classes})\n")
                    
                    # Probability visualization
                    if save_probability_plots and probs_dict:
                        plot_concept_probabilities(
                            probs_dict,
                            out_path=os.path.join(dest_dir, f"{os.path.splitext(os.path.basename(path))[0]}_probs.png"),
                            top_n=10,
                            title=f"Top 10 Concept Similarities (Text Weight: {weight_text:.1f})"
                        )
            
            # Save checkpoint (for resuming in case of interruption)
            new_checkpoint_data = {
                'processed_batches': batch_idx//sub_batch_size + 1,
                'final_results': final_results
            }
            with open(checkpoint_file, 'w') as f:
                json.dump(new_checkpoint_data, f)
            
        # Save results as JSON
        final_results_file = os.path.join(main_out_dir, "final_results.json")
        with open(final_results_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        # Save results for each class
        for target, results in target_class_results.items():
            if results:
                class_results_file = os.path.join(main_out_dir, f"{target}_results.json")
                with open(class_results_file, 'w') as f:
                    json.dump(results, f, indent=2)
        
        # Perform outlier detection by class
        if len(valid_features) > 0:
            print("Executing outlier detection by class...")
            class_outlier_flags, class_outlier_info = detect_class_aware_outliers(
                valid_features, class_names_list, contamination=contamination
            )
            
            with open(os.path.join(main_out_dir, "class_outlier_info.json"), 'w') as f:
                json.dump({cls: {k: v for k, v in info.items() if k != 'outlier_indices'} 
                          for cls, info in class_outlier_info.items()}, f, indent=2)
    
    # Delete temporary files
    if cleanup_temp_files:
        print("Deleting temporary files...")
        try:
            if save_crops:
                # Original cropped_dir is maintained
                pass
            else:
                shutil.rmtree(cropped_dir)
        except Exception as e:
            print(f"Temporary file deletion error: {e}")
    
    # Processing completed
    elapsed_total = time.time() - start_time_overall
    print(f"Processing completed. Total processing time: {elapsed_total:.1f} seconds")
    print(f"Results saved to {main_out_dir}")
    
    return final_results

def detect_outliers_lof(features, n_neighbors=20, contamination=0.1):
    """Outlier detection using LOF"""
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    y_pred = lof.fit_predict(features)
    # LOF: -1 is outlier, 1 is inlier
    return y_pred

def detect_outliers_tsne(tsne_results, n_neighbors=10, threshold_percentile=95):
    """Outlier detection using t-SNE results"""
    # Calculate distances to nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(tsne_results)
    distances, _ = nbrs.kneighbors(tsne_results)
    # Calculate average distance
    avg_distances = distances.mean(axis=1)
    # Set threshold
    threshold = np.percentile(avg_distances, threshold_percentile)
    # Points with distances greater than threshold are outliers
    outlier_flags = (avg_distances > threshold)
    # Convert to Isolation Forest format (-1: outlier, 1: inlier)
    y_pred = np.ones(len(tsne_results))
    y_pred[outlier_flags] = -1
    return y_pred

def combine_outlier_results(outlier_results_list, method='union'):
    """
    Combine multiple outlier detection results
    
    Args:
        outlier_results_list: List of results from each method (-1: outlier, 1: inlier)
        method: 'union' (union) or 'intersection' (intersection)
    
    Returns:
        Combined result (-1: outlier, 1: inlier)
    """
    results = np.array(outlier_results_list)
    if method == 'union':
        # Any method determines outlier (-1)
        combined = np.where(np.any(results == -1, axis=0), -1, 1)
    elif method == 'intersection':
        # All methods determine outlier (-1)
        combined = np.where(np.all(results == -1, axis=0), -1, 1)
    else:
        raise ValueError(f"Unknown method: {method}")
    return combined