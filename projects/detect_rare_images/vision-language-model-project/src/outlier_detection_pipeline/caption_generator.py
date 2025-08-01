#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import time
import sys
from .utils import encode_image, get_device, get_optimal_batch_size

# BLIPモデル用のインポート
try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
except ImportError:
    print("BLIP models not found. Install via 'pip install transformers'")

# Qwen2VLモデル用のインポート
try:
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
except ImportError:
    print("Qwen2VL models not found. Install via 'pip install transformers' and ensure 'qwen_vl_utils.py' is in your path.")

device = get_device()

# グローバルに保持するBLIPインスタンス（シングルトンパターン）
_blip_generator_instance = None

# BLIP版のキャプション生成クラス
class BLIPCaptionGenerator:
    """
    Class for generating captions using BLIP model
    """
    def __init__(self, model_name="Salesforce/blip-image-captioning-large", device=None):
        if device is None:
            device = get_device()
        self.device = device

        print(f"Loading BLIP model {model_name}...")
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name)
        self.model.eval()
        self.model.to(self.device)
        print("BLIP model loaded")

    def generate_caption(self, image_path, max_length=50):
        """
        Generate caption for an image using BLIP model
        """
        try:
            from PIL import Image
            pil_image = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=pil_image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_length=max_length)
                caption = self.processor.decode(generated_ids[0], skip_special_tokens=True)
                
            return caption
        except Exception as e:
            print(f"Error generating caption with BLIP: {e}")
            return None

# シングルトンパターンでBLIPインスタンスを取得するヘルパー関数
def get_blip_generator(model_name="Salesforce/blip-image-captioning-large"):
    """
    Get singleton instance of BLIPCaptionGenerator
    Only loads the model on first call, reuses the same instance afterwards
    """
    global _blip_generator_instance
    if _blip_generator_instance is None:
        try:
            _blip_generator_instance = BLIPCaptionGenerator(model_name=model_name)
        except Exception as e:
            print(f"Failed to initialize BLIP model: {e}")
            return None
    return _blip_generator_instance

# BLIP版のバッチキャプション生成関数
def generate_descriptions_batch_blip(image_paths, blip_generator, batch_size=8, detected_classes=None, process_all=False, outliers_only=False):
    """
    Generate captions in batch processing using BLIP model
    
    Args:
        image_paths: List of image file paths
        blip_generator: BLIPCaptionGenerator instance
        batch_size: Batch size
        detected_classes: Dictionary of detected classes for each image {path: class_name}
        process_all: True=no additional filtering, False=process rare classes only
        outliers_only: If True, process all input images as outliers (skip rare class filtering)
    """
    results = []
    
    # アウトライアのみモードの場合、process_allを上書きしてフィルタリングをスキップ
    if outliers_only:
        process_all = True
        print("BLIP caption generation: Skipping rare class filtering in outliers-only mode")
    
    # フィルタリング（希少クラスのみor全画像）
    if not process_all:
        # 処理対象を希少クラスに絞る
        rare_class_paths = []
        rare_class_indices = []
        rare_results = []
        for i, path in enumerate(image_paths):
            detected_class = detected_classes.get(path, "") if detected_classes else ""
            
            # 希少クラスのみ詳細処理
            is_rare = any(keyword in detected_class.lower() for keyword in 
                        ["construction", "bicycle", "motorcycle", "truck", "fence", "barrier", "traffic_cone", "traffic_light", "traffic_sign"])
            
            if is_rare:
                rare_class_paths.append(path)
                rare_class_indices.append(i)
            else:
                # 非希少クラスはクラス名をそのまま返す
                results.append(detected_class)
        
        # 希少クラスのみキャプション生成
        print(f"BLIP caption generation: Processing {len(rare_class_paths)} out of {len(image_paths)} images (rare classes only)")
        
        # 進捗表示用
        total_images = len(rare_class_paths)
        start_time = time.time()
        
        # 希少クラスの画像を処理
        for i, path in enumerate(rare_class_paths):
            # 進捗表示（10枚ごとまたは最後）
            if i % 10 == 0 or i == len(rare_class_paths) - 1:
                elapsed = time.time() - start_time
                progress = (i + 1) / total_images * 100
                eta = (elapsed / (i + 1)) * (total_images - (i + 1)) if i > 0 else 0
                print(f"Rare class caption generation: {progress:.1f}% ({i+1}/{total_images}) | Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s")
            
            if not os.path.isfile(path):
                rare_results.append(None)
                continue
                
            if detected_classes and path in detected_classes and detected_classes[path] in ['bicycle', 'motorcycle']:
                rare_results.append(detected_classes[path])
                continue
                
            try:
                # BLIPでキャプション生成
                caption = blip_generator.generate_caption(path, max_length=50)
                rare_results.append(caption)
                
                # 10枚ごとにメモリを完全にクリア
                if (i + 1) % 10 == 0:
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"BLIP: Error processing image {path}: {e}")
                rare_results.append(None)
        
        # リソースをクリア
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 結果を元の順序に戻す
        final_results = [None] * len(image_paths)
        
        # 非希少クラスの結果を配置
        non_rare_idx = 0
        for i in range(len(image_paths)):
            if i not in rare_class_indices:
                if non_rare_idx < len(results):
                    final_results[i] = results[non_rare_idx]
                    non_rare_idx += 1
        
        # 希少クラスの結果を配置
        for i, idx in enumerate(rare_class_indices):
            if i < len(rare_results):
                final_results[idx] = rare_results[i]
    
    else:
        # 追加フィルタリングなし - detector.pyから渡された全ての画像を処理
        final_results = []
        total_images = len(image_paths)
        print(f"BLIP caption generation: Processing all {total_images} images (no filtering)")
        
        start_time = time.time()
        for i, path in enumerate(image_paths):
            # 進捗表示（10枚ごとまたは最後）
            if i % 10 == 0 or i == total_images - 1:
                elapsed = time.time() - start_time
                progress = (i + 1) / total_images * 100
                eta = (elapsed / (i + 1)) * (total_images - (i + 1)) if i > 0 else 0
                print(f"Caption generation: {progress:.1f}% ({i+1}/{total_images}) | Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s")
            
            if not os.path.isfile(path):
                final_results.append(None)
                continue
                
            detected_class = detected_classes.get(path, "") if detected_classes else ""
            if detected_class in ['bicycle', 'motorcycle']:
                final_results.append(detected_class)
                continue
                
            try:
                # BLIPでキャプション生成
                caption = blip_generator.generate_caption(path, max_length=50)
                final_results.append(caption)
                
                # 20枚ごとにメモリを完全にクリア
                if (i + 1) % 20 == 0:
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"BLIP: Error processing image {path}: {e}")
                final_results.append(detected_class if detected_class else None)
        
        # リソースをクリア
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    return final_results

# Qwen2VL版のバッチキャプション生成関数（元の実装をコメントアウト）
def generate_descriptions_batch_qwen(image_paths, model, processor, batch_size=8, detected_classes=None, outliers_only=False):
    """バッチ処理で画像キャプションを生成（Qwen2VL版）"""
    results = []
    
    # アウトライアのみモードの場合、フィルタリングを無効化してすべての入力画像を処理
    if outliers_only:
        print("Qwen2VL caption generation: Skipping rare class filtering in outliers-only mode")
        # 全ての画像を処理するために空のリストにする
        rare_class_paths = image_paths
        rare_class_indices = list(range(len(image_paths)))
        rare_results = []
        # resultsはフィルタリングなしの場合は使わないため空のままでよい
    else:
        # 処理対象を希少クラスに絞る
        rare_class_paths = []
        rare_class_indices = []
        rare_results = []
        for i, path in enumerate(image_paths):
            detected_class = detected_classes.get(path, "") if detected_classes else ""
            
            # 希少クラスのみ詳細処理
            is_rare = any(keyword in detected_class.lower() for keyword in 
                        ["construction", "bicycle", "motorcycle", "truck", "fence", "barrier", "traffic_cone", "traffic_light", "traffic_sign"]) 
            
            if is_rare:
                rare_class_paths.append(path)
                rare_class_indices.append(i)
            else:
                # 非希少クラスはクラス名をそのまま返す
                results.append(detected_class)
    
    # キャプション生成情報を表示
    if outliers_only:
        print(f"Qwen2VL caption generation: Processing all {len(image_paths)} images (no filtering)")
    else:
        print(f"Qwen2VL caption generation: Processing {len(rare_class_paths)} out of {len(image_paths)} images")
    
    # 希少クラスに対するキャプション生成
    try:
        # 最適なバッチサイズを取得
        optimal_batch_size = get_optimal_batch_size(initial_size=batch_size)
        if optimal_batch_size != batch_size:
            print(f"Adjusted batch size from {batch_size} to {optimal_batch_size} (memory optimization)")
            batch_size = optimal_batch_size
            
        # 進捗表示用
        total_images = len(rare_class_paths)
        start_time = time.time()
        
        # 希少クラスの画像を処理
        for i, path in enumerate(rare_class_paths):
            # 進捗表示（10枚ごとまたは最後）
            if i % 10 == 0 or i == len(rare_class_paths) - 1:
                elapsed = time.time() - start_time
                progress = (i + 1) / total_images * 100
                eta = (elapsed / (i + 1)) * (total_images - (i + 1)) if i > 0 else 0
                print(f"Caption generation progress: {progress:.1f}% ({i+1}/{total_images}) | Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s")
            
            if not os.path.isfile(path):
                rare_results.append(None)
                continue
                
            detected_class = detected_classes.get(path, "") if detected_classes else ""
            if detected_class in ['bicycle', 'motorcycle']:
                rare_results.append(detected_class)
                continue
                
            try:
                base64_image = encode_image(path)
                if not base64_image:
                    rare_results.append(None)
                    continue
                    
                # YOLOで検出されたクラス名を取得
                class_hint = f" Note that YOLO detected '{detected_class}' in this image, but verify this independently." if detected_class else ""
                    
                message = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": f"data:image/jpeg;base64,{base64_image}",
                            },
                            {
                                "type": "text",
                                "text": "You are an AI vision system specialized in describing images seen from cameras in cars when driving. "
                                        "Focus on unusual or dangerous items that may be relevant to driving safety. "
                                        "Analyze the following image and identify which nuScenes class it belongs to. "
                                        "Give a short answer with just the class name from this list: car, truck, trailer, bus, construction_vehicle, bicycle, motorcycle, pedestrian, traffic_cone, barrier. "
                                        "If it doesn't match any of these classes, answer 'others'. "
                                        f"{class_hint}"
                            },
                        ],
                    }
                ]
                
                text = processor.apply_chat_template(
                    message, tokenize=False, add_generation_prompt=True
                )
                image_inputs, _ = process_vision_info(message)
                
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=None,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = inputs.to(device)
                
                try:
                    # サンプリングかビームかを明示的に設定し、top_p/top_k の警告を抑制
                    with torch.cuda.amp.autocast(enabled=True):
                        generated_ids = model.generate(
                            **inputs,
                            max_new_tokens=150,
                            do_sample=False,      # 警告を消すためサンプリングを有効化
                            top_p=1.0,           # 固定値で警告を解消
                            top_k=50,            # 必要に応じて調整
                            num_beams=1,
                            temperature=1.0
                        )

                        generated_ids_trimmed = [
                        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                        ]
                        output_text = processor.batch_decode(
                            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                        )
                        # print(f"生成されたキャプション: {output_text}")
                        rare_results.append(output_text[0] if output_text else None)
                except Exception as e:
                    print(f"Error generating caption for {path}: {e}")
                    rare_results.append(None)
                    
                    
                # メモリ解放
                del inputs, image_inputs
                if 'generated_ids' in locals():
                    del generated_ids
                
                # 10枚ごとにメモリを完全にクリア（高速化のため頻度を下げる）
                if (i + 1) % 10 == 0:
                    torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Error processing image {path}: {e}")
                rare_results.append(None)
    
    except Exception as e:
        print(f"Batch processing error: {e}")
        import traceback
        traceback.print_exc()
        
        # 未処理の希少クラス画像に対してはNoneを追加
        while len(rare_results) < len(rare_class_paths):
            rare_results.append(None)
    
    finally:
        # リソースをクリア
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # アウトライアのみモードの場合は、結果をそのまま返す
    if outliers_only:
        return rare_results
    
    # 通常モードの場合は、結果を元の順序に戻す
    final_results = [None] * len(image_paths)
    
    # 非希少クラスの結果を配置
    non_rare_idx = 0
    for i in range(len(image_paths)):
        if i not in rare_class_indices:
            if non_rare_idx < len(results):
                final_results[i] = results[non_rare_idx]
                non_rare_idx += 1
    
    # 希少クラスの結果を配置
    for i, idx in enumerate(rare_class_indices):
        if i < len(rare_results):
            final_results[idx] = rare_results[i]
    
    return final_results

# 新しいインターフェースでQwen2VL版のキャプション生成を保持しつつ、
# BLIP版を優先使用するように定義
def generate_descriptions_batch(image_paths, model=None, processor=None, batch_size=8, detected_classes=None, blip_generator=None, model_name="Salesforce/blip-image-captioning-large", process_all_blip=False, use_blip=True, outliers_only=False):
    """
    Generate image captions in batch processing
    
    Selects caption generator based on use_blip flag.
    - use_blip=True: Use BLIP version
    - use_blip=False: Use Qwen2VL version
    
    Args:
        image_paths: List of image file paths
        model: Qwen2VL model (ignored when using BLIP)
        processor: Qwen2VL processor (ignored when using BLIP)
        batch_size: Batch size
        detected_classes: Dictionary of detected classes for each image {path: class_name}
        blip_generator: Existing BLIPCaptionGenerator instance (for reuse)
        model_name: BLIP model name (only used for new instance)
        process_all_blip: Whether to skip rare class filtering when using BLIP model
        use_blip: Whether to use BLIP model (False to use Qwen2VL)
        outliers_only: If True, process outliers only and ignore rare classes
        
    Returns:
        List of captions
    """
    # 明示的にQwen2VLを使用する場合
    if not use_blip:
        print("Using Qwen2VL caption generator")
        if model is not None and processor is not None:
            return generate_descriptions_batch_qwen(image_paths, model, processor, batch_size, detected_classes, outliers_only)
        else:
            raise ValueError("model and processor parameters are required to use Qwen2VL model")
    
    # BLIPモデルを使用する場合
    try:
        # 引数でBLIPジェネレータが渡された場合はそれを使用
        if blip_generator is not None:
            print("Using existing BLIP caption generator")
        else:
            # シングルトンからBLIPジェネレータを取得
            blip_generator = get_blip_generator(model_name)
            if blip_generator is None:
                raise Exception("Failed to initialize BLIP model")
            print("Initialized BLIP caption generator")
        
        filter_mode = "disabled" if process_all_blip else "enabled"
        print(f"Rare class filtering: {filter_mode} (when using BLIP model)")
        return generate_descriptions_batch_blip(
            image_paths, 
            blip_generator, 
            batch_size, 
            detected_classes,
            process_all=process_all_blip,  # フィルタリングフラグ
            outliers_only=outliers_only  # アウトライアのみモード
        )
    except Exception as e:
        print(f"BLIP initialization error: {e}, falling back to Qwen2VL")
        # Qwen2VL版にフォールバック
        if model is not None and processor is not None:
            return generate_descriptions_batch_qwen(image_paths, model, processor, batch_size, detected_classes, outliers_only)
        else:
            raise ValueError("Qwen2VL model and processor are required but not provided") 