#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import base64
import numpy as np
import torch
from torch.nn.functional import softmax
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import random
import shutil
import datetime
import json
import argparse
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from ultralytics.nn.tasks import DetectionModel
from collections import Counter
import time
import openai
import signal
import sys

# YOLO (Ultralytics)
try:
    import ultralytics
    from ultralytics import YOLO
except ImportError:
    print("Ultralytics YOLO not found. Install via 'pip install ultralytics'.")

from dotenv import load_dotenv
load_dotenv()
from openai import Client
client = Client()
openai.api_key = os.getenv('OPENAI_API_KEY')

from transformers import CLIPProcessor, CLIPModel

# t-SNE
from sklearn.manifold import TSNE
# Isolation Forest
from sklearn.ensemble import IsolationForest

###############################################################################
# encode_image: base64 encoding for image
###############################################################################
def encode_image(image_path):
    if not os.path.isfile(image_path):
        return None
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode("utf-8")


###############################################################################
# 1. 2D Detection (YOLO)
###############################################################################
def get_device():
    if torch.cuda.is_available():
        return "cuda:0"
    else:
        return "cpu"
device = get_device()
class DetectorYOLO:
    """
    YOLOv8-based object detection class.
    """
    def __init__(self, model_path="yolov8l.pt", device=None):
        if device is None:
            device = get_device()
        self.device = device
        self.model = YOLO(model_path)
        self.model.to(self.device)

    def detect_and_crop(self, image_path, out_dir="cropped_objects", conf_thres=0.3):
        """
        Detect objects in the image. Crop the objects whose confidence
        is above the given threshold and save them.

        Returns:
            cropped_info_list: [(out_path, class_name, conf, xyxy), ...]
        """
        os.makedirs(out_dir, exist_ok=True)
        try:
            # 画像が存在するか確認
            if not os.path.isfile(image_path):
                print(f"画像ファイルが見つかりません: {image_path}")
                return []
            
            original_img = Image.open(image_path).convert("RGB")
            original_width, original_height = original_img.size
            
            # 処理用に画像をリサイズ
            img = original_img.copy()
            width, height = img.size
            max_size = 1280
            scale_factor = 1.0
            
            if width > max_size or height > max_size:
                if width > height:
                    new_width = max_size
                    new_height = int(height * (max_size / width))
                    scale_factor = original_width / new_width
                else:
                    new_height = max_size
                    new_width = int(width * (max_size / height))
                    scale_factor = original_height / new_height
                
                img = img.resize((new_width, new_height), Image.LANCZOS)
                # 詳細なリサイズログは削除して処理を高速化
            
            # バッチ処理モードを使用
            results = self.model.predict(
                source=img,
                conf=conf_thres,
                iou=0.45,
                max_det=100,
                device=self.device,
                verbose=False  # 冗長な出力を抑制
            )
            
            if len(results) == 0:
                return []

            boxes = results[0].boxes
            names = self.model.names
            cropped_info_list = []

            # 検出結果が多すぎる場合は上位のみ処理
            max_objects = 50  # 1画像あたりの最大処理オブジェクト数
            num_boxes = min(len(boxes), max_objects)
            
            if len(boxes) > max_objects:
                # 信頼度でソートして上位のみ処理
                conf_values = boxes.conf.cpu().numpy()
                sorted_indices = np.argsort(-conf_values)[:max_objects]
            else:
                sorted_indices = range(len(boxes))

            for i in sorted_indices:
                # リサイズされた画像での座標
                xyxy_resized = boxes.xyxy[i].cpu().numpy().astype(int)
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())
                class_name = names[cls_id] if cls_id in names else f"class_{cls_id}"

                # 元の解像度に座標を戻す
                if scale_factor != 1.0:
                    x1, y1, x2, y2 = xyxy_resized
                    x1_orig = int(x1 * scale_factor)
                    y1_orig = int(y1 * scale_factor)
                    x2_orig = int(x2 * scale_factor)
                    y2_orig = int(y2 * scale_factor)
                    
                    # 元画像の範囲内に収める
                    x1_orig = max(0, min(x1_orig, original_width - 1))
                    y1_orig = max(0, min(y1_orig, original_height - 1))
                    x2_orig = max(0, min(x2_orig, original_width))
                    y2_orig = max(0, min(y2_orig, original_height))
                    
                    xyxy = (x1_orig, y1_orig, x2_orig, y2_orig)
                else:
                    xyxy = xyxy_resized

                # 元の解像度の画像から切り抜く
                x1, y1, x2, y2 = xyxy
                cropped = original_img.crop((x1, y1, x2, y2))

                basename_no_ext = os.path.splitext(os.path.basename(image_path))[0]
                out_path = os.path.join(
                    out_dir, f"{basename_no_ext}_{i}_{class_name}.jpg"
                )
                cropped.save(out_path)
                cropped_info_list.append((out_path, class_name, conf, xyxy, image_path))

            return cropped_info_list
        except Exception as e:
            print(f"画像処理エラー {image_path}: {e}")
            return []

###############################################################################
# 2. Vision-Language Model (CLIP) for feature extraction
###############################################################################
class CLIPFeatureExtractor:
    """
    Obtain feature vectors (embeddings) for images or text using CLIP.
    """
    def __init__(self, model_name="openai/clip-vit-base-patch32", device=None):
        if device is None:
            device = get_device()
        self.device = device

        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()
        self.model.to(self.device)

    def get_image_embedding(self, image_path):
        """
        Get the CLIP image feature (np.array, shape=(dim,)) from the given image path.
        """
        pil_image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
        return image_features.cpu().numpy().squeeze(0)

    def get_image_embeddings_batch(self, image_paths, batch_size=32):
        """複数の画像のCLIP特徴をバッチ処理で一度に取得"""
        all_embeddings = []
        total = len(image_paths)
        
        # 進捗表示を追加
        start_time = time.time()
        
        for i in range(0, total, batch_size):
            # 進捗表示（100バッチごとまたは最後）
            if i % (batch_size * 100) == 0 or i + batch_size >= total:
                elapsed = time.time() - start_time
                progress = min(i + batch_size, total) / total * 100
                eta = (elapsed / (i + batch_size)) * (total - (i + batch_size)) if i > 0 else 0
                print(f"特徴抽出進捗: {progress:.1f}% ({min(i + batch_size, total)}/{total}) | 経過: {elapsed:.1f}秒 | 残り: {eta:.1f}秒")
            
            batch_paths = image_paths[i:i+batch_size]
            batch_images = []
            valid_indices = []
            
            # エラー処理を追加して無効な画像をスキップ
            for j, path in enumerate(batch_paths):
                try:
                    img = Image.open(path).convert("RGB")
                    batch_images.append(img)
                    valid_indices.append(j)
                except Exception as e:
                    print(f"画像読み込みエラー {path}: {e}")
            
            if not batch_images:
                continue
            
            inputs = self.processor(images=batch_images, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                batch_features = self.model.get_image_features(**inputs)
            
            batch_features = batch_features.cpu().numpy()
            
            # 結果を元の順序に戻す
            for j, idx in enumerate(valid_indices):
                while len(all_embeddings) < i + idx:
                    all_embeddings.append(None)  # 無効な画像の場所にはNoneを入れる
                all_embeddings.append(batch_features[j])
            
            # バッチ処理後にメモリを解放
            del batch_images, inputs, batch_features
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # 最後まで埋める
        while len(all_embeddings) < total:
            all_embeddings.append(None)
        
        return np.array([emb for emb in all_embeddings if emb is not None])

    def get_text_embedding(self, text_list):
        """
        Get the CLIP text feature vectors (np.array, shape=(len(text_list), dim))
        for a list of texts.
        """
        inputs = self.processor(text=text_list, return_tensors="pt",
                                padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
        return text_features.cpu().numpy()


###############################################################################
# 3. text analysis by cosine similarity 
###############################################################################
def parse_text_with_probability(base_text, candidate_labels, clip_extractor):
    """
    compare the base text with candidate labels using CLIP text embeddings,
    calculate cosine similarity, and return a dict of label-to-probability.
    """
    if not base_text:
        return {}

    text_emb = clip_extractor.get_text_embedding([base_text])
    labels_emb = clip_extractor.get_text_embedding(candidate_labels)

    text_emb_norm = text_emb / np.linalg.norm(text_emb, axis=1, keepdims=True)
    labels_emb_norm = labels_emb / np.linalg.norm(labels_emb, axis=1, keepdims=True)

    cos_sims = []
    for i in range(labels_emb_norm.shape[0]):
        sim = (text_emb_norm[0] * labels_emb_norm[i]).sum()
        cos_sims.append(sim)
    cos_sims = np.array(cos_sims)

    tensor_sims = torch.tensor(cos_sims)
    probs = cos_sims
    label_probs = {candidate_labels[i]: float(probs[i]) for i in range(len(candidate_labels)) if probs[i] >= 0}
    label_probs = {k: v for k, v in sorted(label_probs.items(), key=lambda x: x[1], reverse=True)}
    return label_probs


###############################################################################
# 4. Image Captioning (Qwen2-VL)
###############################################################################
def generate_description(image_path, model, processor):
    """
    Get image caption using Qwen model.
    Deprecated: Use generate_descriptions_batch() instead for better performance.
    Note: This function is kept for backwards compatibility but may be removed in future versions.
    """
    """
    get image caption using Qwen model.
    """
    base64_image = encode_image(image_path)
    if not base64_image:
        return None
                
    messages = [
        {
            "role": "system",
            "content": (
                "You are an AI vision system specialized in describing images. "
                "Focus on unusual or domain-relevant items."
            )
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Analyze the following image and list up to 10 notable objects/scenarios."
                    )
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }
    ]
    try:
        max_tokens = 150
        messages = [
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
                                "Analyze the following image and extract the most notable objects/scenarios."
                                f"Note that max tokens is {max_tokens}."
                    },
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(device)

        max_tokens = 150
        generated_ids = model.generate(**inputs, max_new_tokens=max_tokens)

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        print(output_text)
        response = output_text[0]
        return response

    except Exception as e:
        print(f"Qwen model error generating caption for {image_path}: {e}")
        return None

def generate_descriptions_batch(image_paths, model, processor, batch_size=8, detected_classes=None):
    """バッチ処理で画像キャプションを生成（メモリ管理強化版）"""
    results = []
    
    try:
        # 最適なバッチサイズを取得
        optimal_batch_size = get_optimal_batch_size(initial_size=batch_size)
        if optimal_batch_size != batch_size:
            print(f"バッチサイズを {batch_size} から {optimal_batch_size} に調整しました（メモリ最適化）")
            batch_size = optimal_batch_size
            
        # 進捗表示用
        total_images = len(image_paths)
        start_time = time.time()
        
        # 画像を処理
        for i, path in enumerate(image_paths):
            # 進捗表示（100枚ごとまたは最後）
            if i % 100 == 0 or i == len(image_paths) - 1:
                elapsed = time.time() - start_time
                progress = (i + 1) / total_images * 100
                eta = (elapsed / (i + 1)) * (total_images - (i + 1)) if i > 0 else 0
                print(f"キャプション生成: {progress:.1f}% ({i+1}/{total_images}) | 経過: {elapsed:.1f}秒 | 残り: {eta:.1f}秒")
            
            if not os.path.isfile(path):
                results.append(None)
                continue
                
            try:
                base64_image = encode_image(path)
                if not base64_image:
                    results.append(None)
                    continue
                    
                # YOLOで検出されたクラス名を取得
                detected_class = detected_classes.get(path, "") if detected_classes else ""
                class_hint = f" Note that YOLO says the image is about a {detected_class}." if detected_class else ""
                    
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
                                        "Analyze the following image and list up to 10 notable objects/scenarios."
                                        f"{class_hint}"
                                        f"Note that max tokens is 150."
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
                    with torch.cuda.amp.autocast(enabled=True):  # 混合精度を有効化
                        generated_ids = model.generate(
                            **inputs, 
                            max_new_tokens=150,
                            do_sample=False,
                            num_beams=1,
                            temperature=1.0
                        )
                    
                    generated_ids_trimmed = [
                        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    output_text = processor.batch_decode(
                        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )
                    results.append(output_text[0] if output_text else None)
                except Exception as e:
                    print(f"Error generating caption for {path}: {e}")
                    results.append(None)
                    
                # メモリ解放
                del inputs, image_inputs
                if 'generated_ids' in locals():
                    del generated_ids
                
                # 100枚ごとにメモリを完全にクリア
                if (i + 1) % 100 == 0:
                    torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Error processing image {path}: {e}")
                results.append(None)
                
        # 最終的なメモリ使用状況をモニタリング
        if torch.cuda.is_available():
            used_memory = torch.cuda.memory_allocated(0) / (1024**3)
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"GPUメモリ使用状況: {used_memory:.2f}GB / {total_memory:.2f}GB ({used_memory/total_memory*100:.1f}%)")
            
    except Exception as e:
        print(f"Batch processing error: {e}")
        import traceback
        traceback.print_exc()
        
        # 未処理の画像に対してはNoneを追加
        while len(results) < len(image_paths):
            results.append(None)
            
    finally:
        # リソースをクリア
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    return results

###############################################################################
# 5. visualization: t-SNE
###############################################################################
def visualize_tsne(features, labels=None, title="t-SNE Visualization"):
    """
    features: shape=(N, D)
    labels:   shape=(N,) or None
    """
    if len(features) == 0:
        print("No features to visualize.")
        return

    tsne = TSNE(n_components=2, random_state=42,
                perplexity=min(30, max(5, len(features) - 1)))
    reduced = tsne.fit_transform(features)

    plt.figure(figsize=(10, 7))
    if labels is not None:
        scatter = plt.scatter(
            reduced[:, 0], reduced[:, 1],
            c=labels, cmap="tab10", alpha=0.7
        )
        plt.legend(*scatter.legend_elements(), title="Classes")
    else:
        plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.7)
    plt.title(title)
    plt.xlabel("Dim1")
    plt.ylabel("Dim2")
    save_name = title.lower().replace(" ", "_")
    out_tsne_path = os.path.join("out_images", "tsne_visualization.png")
    plt.savefig(out_tsne_path)
    plt.close()

###############################################################################
# 6. Outlier detection (IsolationForest)
###############################################################################
def train_isolation_forest(features, contamination=0.1):
    """
    Train IsolationForest with the features of the baseline.
    """
    iso = IsolationForest(contamination=0.1, random_state=42)
    iso.fit(features)
    return iso


def predict_outliers(features, iso_model):
    """
    detect outliers using the pre-trained iso_model.
    1: inlier, -1: outlier
    """
    if len(features) == 0:
        return []
    return iso_model.predict(features)


###############################################################################
# train baseline model
###############################################################################
def train_baseline_model(baseline_folder, qwen_model_size, use_concept_list, concept_list, main_out_dir, lim1=0.8, train_indices=None):
    detector = DetectorYOLO(model_path="yolov8l.pt")
    clip_extractor = CLIPFeatureExtractor(model_name="openai/clip-vit-base-patch32")

    cropped_dir = os.path.join(main_out_dir, "baseline_cropped_objects")
    os.makedirs(cropped_dir, exist_ok=True)

    all_files = [f for f in os.listdir(baseline_folder) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif"))]
    random.shuffle(all_files)
    
    if train_indices is not None:
        baseline_files = [all_files[i] for i in train_indices]
    else:
        baseline_files = all_files[:int(lim1 * len(all_files))]

    all_cropped_info = []
    for file_name in baseline_files:
        try:
            image_path = os.path.join(baseline_folder, file_name)
            cropped_info = detector.detect_and_crop(
                image_path, out_dir=cropped_dir, conf_thres=0.3
            )
            all_cropped_info.extend(cropped_info)
        except Exception as e:
            print(f"Error processing baseline image {file_name}: {e}")
            continue

    features = []
    class_names = []
    for (obj_path, cls_name, conf, bbox, _) in all_cropped_info:
        emb = clip_extractor.get_image_embedding(obj_path)
        features.append(emb)
        class_names.append(cls_name)

    features = np.array(features)
    if len(features) == 0:
        print("No features extracted from baseline. Exiting...")
        return None, None, None, None, None, None

    iso_model = train_isolation_forest(features, contamination=0.15)

    unique_labels = list(set(class_names))
    label2id = {lb: i for i, lb in enumerate(unique_labels)}
    label_ids = [label2id[lb] for lb in class_names]

    if use_concept_list:
        print("Using predefined concept list for candidate labels.")
        candidate_labels = concept_list
        extended_candidate_labels = candidate_labels
    else:
        print("Generating candidate labels from detected classes and extending with features.")
        candidate_labels = list(set(class_names))
        
        top_classes = [cls for cls, _ in Counter(class_names).most_common(10)]
        print(f"Generating feature dictionary for top {len(top_classes)} classes...")
        feature_dict = generate_feature_dict(top_classes, qwen_model_size=qwen_model_size, prompt_type="important") 
        
        extended_candidate_labels = []
        for label in candidate_labels:
            if label in feature_dict:
                extended_candidate_labels.extend(feature_dict[label])
            else:
                extended_candidate_labels.append(label)
        
        extended_candidate_labels = list(set(extended_candidate_labels))
        print(f"Extended candidate labels from {len(candidate_labels)} to {len(extended_candidate_labels)}")

    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    baseline_tsne = tsne.fit_transform(features)
    
    plt.figure(figsize=(10,7))
    scatter = plt.scatter(baseline_tsne[:,0], baseline_tsne[:,1], alpha=0.5)
    plt.title("Baseline t-SNE Visualization")
    baseline_tsne_path = os.path.join(main_out_dir, "baseline_tsne.png")
    plt.savefig(baseline_tsne_path)
    plt.close()

    class_counts = Counter(class_names)
    top_classes = [clas for clas, count in class_counts.most_common(20)]
    print("Top 20 classes:", top_classes)
    plot_indices = [i for i, lb in enumerate(class_names) if lb in top_classes]
    filtered_tsne = baseline_tsne[plot_indices]
    filtered_labels = [class_names[i] for i in plot_indices]
    plt.figure(figsize=(15, 10))
    cmap = plt.get_cmap("tab20")
    for i, lb in enumerate(top_classes):
        indices = [j for j, x in enumerate(filtered_labels) if x == lb]
        plt.scatter(filtered_tsne[indices, 0], filtered_tsne[indices, 1], label=lb, color=cmap(i/len(top_classes)), alpha=0.7)
    plt.title("Baseline t-SNE Visualization (Top 20 Classes)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    baseline_tsne_class_path = os.path.join(main_out_dir, "baseline_tsne_class.png")
    plt.savefig(baseline_tsne_class_path, bbox_inches='tight')
    plt.close()

    return clip_extractor, extended_candidate_labels, features, baseline_tsne, class_names, iso_model


###############################################################################
# detect_new_images
###############################################################################
def detect_new_images(
    new_images_folder,
    clip_extractor,
    candidate_labels,
    qwen_model_size,
    main_out_dir,
    common_classes,
    baseline_features=None,
    baseline_tsne=None,
    baseline_class_names=None,
    iso_model=None,
    threshold_percentile=95,
    lim1=0.8,
    lim2=1.0,
    save_outliers=True,
    save_outlier_list=True,
    test_indices=None,
    only_additional_candidates=False,
    potential_classes=None,
    save_crops=True,
    save_descriptions=True,
    save_probability_plots=True,
    cleanup_temp_files=False,  # 一時ファイル削除オプション
):
    start_time_overall = time.time()
    detector = DetectorYOLO(model_path="yolov8l.pt")
    run_specific_out_dir = os.path.join(main_out_dir, f"detection_lim{lim1}-{lim2}")
    os.makedirs(run_specific_out_dir, exist_ok=True)

    all_files = [f for f in os.listdir(new_images_folder) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif"))]
    random.shuffle(all_files)
    
    if test_indices is not None:
        new_files = [all_files[i] for i in test_indices]
    else:
        new_files = all_files[int(lim1 * len(all_files)): int(lim2 * len(all_files))]

    total_files = len(new_files)
    print(f"処理対象: {total_files}枚の画像")

    # 1. オブジェクト検出と切り抜き
    all_cropped_info = []
    for idx, file_name in enumerate(new_files):
        try:
            # 進捗表示を追加
            if idx % 100 == 0 or idx == len(new_files) - 1:
                elapsed = time.time() - start_time_overall
                progress = (idx + 1) / total_files * 100
                eta = (elapsed / (idx + 1)) * (total_files - (idx + 1)) if idx > 0 else 0
                print(f"進捗: {progress:.1f}% ({idx+1}/{total_files}) | 経過時間: {elapsed:.1f}秒 | 残り時間: {eta:.1f}秒")
            
            image_path = os.path.join(new_images_folder, file_name)
            cropped_info = detector.detect_and_crop(
                image_path, out_dir=run_specific_out_dir, conf_thres=0.3
            )
            all_cropped_info.extend(cropped_info)
        except Exception as e:
            print(f"Error processing new image {file_name}: {e}")
            continue

    # 2. バッチ処理で特徴抽出
    print(f"\n特徴抽出: {len(all_cropped_info)}個の物体から特徴を抽出中...")
    start_time_feature = time.time()
    
    obj_paths = [info[0] for info in all_cropped_info]
    features_new = clip_extractor.get_image_embeddings_batch(obj_paths, batch_size=32)
    
    print(f"特徴抽出完了: {time.time() - start_time_feature:.1f}秒")

    if len(features_new) == 0:
        print("No features extracted from new images. Exiting...")
        return
    combined_features = np.vstack([baseline_features, features_new])

    # t-SNEによる次元削減と座標化
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    combined_tsne = tsne.fit_transform(combined_features)
    baseline_tsne_2d = combined_tsne[:len(baseline_features)]
    new_tsne = combined_tsne[len(baseline_features):]

    # 新たに計算したt-SNE座標を使用する
    nbrs = NearestNeighbors(n_neighbors=10).fit(baseline_tsne_2d)  # baseline_tsneではなくbaseline_tsne_2dを使用
    distances, indices = nbrs.kneighbors(new_tsne)
    avg_distances = distances.mean(axis=1)
    threshold_tsne = np.percentile(avg_distances, threshold_percentile)
    tsne_outliers = (avg_distances > threshold_tsne).astype(int)

    if_labels = predict_outliers(features_new, iso_model)
    if_outliers = ((if_labels==-1).astype(int))
    combined_outlier_code = tsne_outliers*2 + if_outliers

    plt.figure(figsize=(18,12))

    from collections import Counter

    class_counts = Counter(baseline_class_names)
    top_classes = [cls for cls, _ in class_counts.most_common(20)]
    cmap = plt.get_cmap('tab20')

    plt.scatter(baseline_tsne_2d[:,0], baseline_tsne_2d[:,1],
                c='lightgray', alpha=0.4, label="Baseline")

    color_map = {
        0: ('blue', 'Inlier (both normal)'),
        1: ('red',  'IF outlier only'),
        2: ('orange','t-SNE outlier only'),
        3: ('purple','Both outlier'),
    }
    for code in [0,1,2,3]:
        indices_code = np.where(combined_outlier_code == code)[0]
        if len(indices_code) == 0:
            continue
        x_sub = new_tsne[indices_code,0]
        y_sub = new_tsne[indices_code,1]
        plt.scatter(x_sub, y_sub, c=color_map[code][0],
                    label=color_map[code][1], alpha=0.8)

    plt.title("Combined t-SNE Visualization (IsolationForest + t-SNE)")
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
    plt.tight_layout()
    combined_tsne_path = os.path.join(main_out_dir, "combined_tsne_class_if.png")
    plt.savefig(combined_tsne_path, bbox_inches='tight')
    plt.close()

    out_folders = {
        0: "inlier_inlier",
        1: "if_outlier_only",
        2: "tsne_outlier_only",
        3: "both_outlier"
    }
    for k, v in out_folders.items():
        folder_path = os.path.join(run_specific_out_dir, v)
        os.makedirs(folder_path, exist_ok=True)

    # YOLOクラス名に基づいて追加の候補を特定
    additional_indices = []
    # 関連クラスリスト: 凡例を参考に 'fire hydrant', 'train', 'boat' を追加
    potential_classes = ['construction_vehicle', 'motorcycle', 'bicycle', 'truck', 'bus', 'trailer',
                         'emergency_vehicle', 'police_car', 'ambulance', 'fire_truck',
                         'tractor', 'snowplow', 'garbage_truck', 'military_vehicle',
                         ] # <- 追加   
    for i, info in enumerate(all_cropped_info):
        original_index_in_new = i # combined_outlier_code は new_features に対するインデックス
        if original_index_in_new < len(combined_outlier_code): # インデックス範囲チェック
            if combined_outlier_code[original_index_in_new] == 0: # インライアのみ対象
                _, cls_name, _, _, _ = info
                # クラス名を小文字に変換して比較（YOLOモデルによって大文字/小文字が異なる場合があるため）
                if cls_name.lower() in potential_classes:
                    additional_indices.append(original_index_in_new)

    # 重複を除いてマージ
    original_outlier_indices = set(np.where(combined_outlier_code != 0)[0])
    additional_indices_set = set(additional_indices)
    
    # 引数に基づいて処理対象を決定
    if only_additional_candidates:
        # 追加候補のみを処理
        combined_indices = list(additional_indices_set)
        print(f"追加候補のみを処理します（IF/tSNEアウトライアは無視）")
    else:
        # 通常通り、アウトライアと追加候補を合わせて処理
        combined_indices = list(original_outlier_indices.union(additional_indices_set))
    
    combined_indices.sort() # 順序を保つためソート

    print(f"\n潜在的なアウトライア (IF/tSNE): {len(original_outlier_indices)}個")
    print(f"YOLOクラス名に基づく追加候補: {len(additional_indices_set)}個")
    print(f"合計処理対象: {len(combined_indices)}個 処理中...")

    # outlier_indices を combined_indices に置き換える
    outlier_indices = combined_indices # キャプション生成等の対象インデックスリスト

    # アウトライアだけを処理する場合は、QwenモデルをGPUに維持する前に確認
    if len(outlier_indices) > 0:
        # Qwenモデルのロード（処理対象がある場合のみ）
        model_name = f"Qwen/Qwen2-VL-{qwen_model_size}-Instruct"
        print(f"{model_name}モデルをロード中...")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(model_name)
    else:
        print("処理対象のオブジェクトが検出されなかったため、キャプション生成をスキップします")
        return # 処理対象がない場合はここで終了

    detected_images_dir = os.path.join(main_out_dir, "detected_images_overall")
    os.makedirs(detected_images_dir, exist_ok=True)

    # 結果保存用リストの初期化
    outlier_info = [] # 最終的なアウトライア情報 (保存用、後で final_outliers を使う)
    cv_info = []
    motor_info = []
    final_outliers = [] # 最終的なアウトライア (JSON保存用)
    non_final_outliers = [] # 最終的ではないが処理されたアウトライア (JSON保存用)


    cv_folder = os.path.join(main_out_dir, "construction_vehicle")
    os.makedirs(cv_folder, exist_ok=True)
    motor_folder = os.path.join(main_out_dir, "motorcycle")
    os.makedirs(motor_folder, exist_ok=True)

    # final_folder は使われていないようなのでコメントアウト (必要なら復活)
    # final_folder = os.path.join(main_out_dir, "final_outliers")
    # os.makedirs(final_folder, exist_ok=True)

    total_process_count = len(outlier_indices) # 処理対象の総数
    processed_count = 0 # 処理済みカウント
    start_time_caption = time.time()

    # 中間保存用の変数を追加
    CHECKPOINT_INTERVAL = 1000  # 1000枚ごとに保存
    last_checkpoint = 0

    # メモリ使用状況をモニタリングする関数
    def monitor_memory():
        if torch.cuda.is_available():
            used_memory = torch.cuda.memory_allocated(0) / (1024**3)
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"GPUメモリ使用状況: {used_memory:.2f}GB / {total_memory:.2f}GB ({used_memory/total_memory*100:.1f}%)")
    
    try:
        import signal
        def signal_handler(signum, frame):
            print("\nシグナルを受信しました。クリーンアップを実行します...")
            if 'model' in locals():
                del model
            if 'processor' in locals():
                del processor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            sys.exit(0)

        # SIGTERMとSIGINTのハンドラを設定
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        # 処理対象のパスをリストアップ (outlier_indices を使用)
        outlier_paths = [all_cropped_info[i][0] for i in outlier_indices if i < len(all_cropped_info)] # 範囲チェック追加

        # バッチサイズを設定（メモリに合わせて調整）
        caption_batch_size = get_optimal_batch_size()

        # YOLOで検出されたクラス名を保持する辞書を作成 (outlier_indices を使用)
        detected_classes_dict = {
            info[0]: info[1]
            for idx in outlier_indices
            if idx < len(all_cropped_info) # 範囲チェック追加
            for info in [all_cropped_info[idx]]
        }

        # バッチ処理でキャプション生成
        if outlier_paths:
            print(f"バッチ処理で{len(outlier_paths)}個のオブジェクトのキャプションを生成中...")

            sub_batch_size = 100  # 一度に処理する画像数

            for i in range(0, len(outlier_paths), sub_batch_size):
                sub_batch_paths = outlier_paths[i:i+sub_batch_size]
                # sub_batch_indices は outlier_indices の中の、現在のサブバッチに対応する部分
                sub_batch_indices = outlier_indices[i:min(i+sub_batch_size, len(outlier_indices))]

                print(f"サブバッチ処理中: {i+1}〜{min(i+sub_batch_size, len(outlier_paths))}/{len(outlier_paths)}...")

                # サブバッチのキャプション生成
                sub_batch_descriptions = generate_descriptions_batch(
                    sub_batch_paths,
                    model,
                    processor,
                    batch_size=caption_batch_size,
                    detected_classes=detected_classes_dict
                )

                # このサブバッチに対して後続処理を実行
                for j, desc in enumerate(sub_batch_descriptions):
                    if j >= len(sub_batch_indices): # sub_batch_indices の範囲を超えないように
                        continue

                    idx = sub_batch_indices[j] # 現在処理中のオブジェクトの全体インデックス
                    if idx >= len(all_cropped_info): # all_cropped_info の範囲チェック
                        continue

                    # combined_outlier_code から out_code を取得 (範囲チェック付き)
                    # idx が combined_outlier_code の範囲外になることは基本的にないはずだが念のため
                    out_code = combined_outlier_code[idx] if idx < len(combined_outlier_code) else -1 # 不明な場合は -1

                    (path, cls_name, conf, bbox, original_path) = all_cropped_info[idx]

                    # 保存先フォルダ名を決定 (out_code=0 かつ追加候補の場合は専用フォルダも検討可能)
                    if out_code == 0 and idx in additional_indices_set:
                        folder_name = "yolo_potential_inliers" # 例: YOLO候補だがIF/tSNEではインライア
                    else:
                        folder_name = out_folders.get(out_code, "others") # IF/tSNEの結果に基づくフォルダ

                    save_folder = os.path.join(run_specific_out_dir, folder_name)
                    os.makedirs(save_folder, exist_ok=True) # フォルダが存在しない場合作成

                    is_final_outlier = False
                    top_concept = None
                    probs_dict = {} # 初期化
                    # is_processed_for_caption は不要、このループ内のものは全て処理対象
                    processed_count += 1 # 処理カウントをインクリメント

                    # --- キャプション生成と類似度計算 ---
                    # 進捗表示
                    if processed_count % 100 == 0 or processed_count == total_process_count:
                        elapsed = time.time() - start_time_caption
                        progress = (processed_count / total_process_count) * 100
                        eta = (elapsed / processed_count) * (total_process_count - processed_count) if processed_count > 0 else 0
                        elapsed_min, elapsed_sec = divmod(elapsed, 60)
                        eta_min, eta_sec = divmod(eta, 60)
                        print(f"Caption progress: {progress:.1f}% ({processed_count}/{total_process_count}) | "
                              f"経過時間: {int(elapsed_min)}分{elapsed_sec:.1f}秒 | 残り時間: {int(eta_min)}分{eta_sec:.1f}秒")

                    # チェックポイントの保存
                    if processed_count - last_checkpoint >= CHECKPOINT_INTERVAL:
                        try:
                            checkpoint_path = os.path.join(main_out_dir, f"checkpoint.json")
                            current_results = {
                                'timestamp': datetime.datetime.now().isoformat(),
                                'processed_count': processed_count,
                                'final_outliers': final_outliers, # 保存するリストを統一
                                'non_final_outliers': non_final_outliers,
                                'cv_info': cv_info,
                                'motor_info': motor_info
                            }
                            with open(checkpoint_path, 'w') as f:
                                json.dump(current_results, f, indent=2)
                            print(f"Checkpoint saved at {checkpoint_path}")
                            last_checkpoint = processed_count
                        except Exception as e:
                            print(f"Error saving checkpoint: {e}")

                    desc_lower = desc.lower() if desc else ""

                    # キャプション判定ロジック
                    is_failed_caption = False
                    if not desc:
                        is_failed_caption = True
                    elif any(phrase in desc_lower for phrase in ["i'm sorry", "unable to", "i'm unable to", "cannot process"]):
                        is_failed_caption = True
                    elif not desc.rstrip().endswith(('.', '!', '?', '"', "'", ')', ']')):
                        # 不完全なキャプションかもしれないが、処理は続行
                        is_failed_caption = False # 失敗とはしない

                    if is_failed_caption:
                        print(f"Caption generation failed for {path}. Attempting to regenerate...")
                        retry_desc = generate_description(path, model, processor) # 個別生成関数を呼び出し
                        if retry_desc and not any(phrase in retry_desc.lower() for phrase in ["i'm sorry", "unable to"]):
                            desc = retry_desc
                            print(f"Successfully regenerated caption for {path}")
                            is_failed_caption = False
                        else:
                             print(f"Caption regeneration failed for {path}.")


                    # 類似度計算 (失敗していない場合)
                    if not is_failed_caption:
                        probs_dict = parse_text_with_probability(desc, candidate_labels, clip_extractor)
                        if len(desc.split()) < 5: # 短すぎるキャプションの閾値を調整
                            print(f"Warning: Caption for {path} is very short ({len(desc.split())} words): '{desc}'")
                    else:
                        print(f"Skipping similarity calculation for {path} due to caption failure.")
                        # probs_dict は空のまま

                    # top_concept の取得と is_final_outlier の判定
                    if probs_dict:
                        top_concept = next(iter(probs_dict))
                        # final outlier の判定基準: top_concept が common_classes にないこと
                        # (IF/tSNEアウトライアかYOLO追加候補かは問わない)
                        if top_concept not in common_classes:
                             is_final_outlier = True
                    # --- キャプション生成と類似度計算 終了 ---

                    # --- ファイルコピーと情報保存 ---
                    # ファイルコピー (final outlier の場合)
                    if is_final_outlier:
                        save_img = os.path.join(detected_images_dir, os.path.basename(original_path))
                        # コピー元が存在するか確認
                        if os.path.exists(original_path):
                            shutil.copy(original_path, save_img)
                        else:
                            print(f"Warning: Original image not found for copying: {original_path}")
                        # クロップ画像もコピー (save_cropsがTrueの場合のみ)
                        if os.path.exists(path) and save_crops:
                            shutil.copy(path, os.path.join(save_folder, os.path.basename(path)))
                        else:
                            if save_crops:  # 警告メッセージも条件付きで表示
                                print(f"Warning: Cropped image not found for copying: {path}")


                    # description ファイル作成 (save_descriptionsがTrueの場合のみ)
                    if save_descriptions:
                        txt_path = os.path.join(save_folder, f"{os.path.splitext(os.path.basename(path))[0]}_desc.txt")
                        with open(txt_path, "w", encoding="utf-8") as f:
                            f.write(f"Outlier code (IF/tSNE): {out_code} (0:inlier, 1:IFonly, 2:tSNEonly, 3:both, -1:unknown)\n")
                            f.write(f"YOLO Class: {cls_name}\n")
                            f.write(f"Is YOLO potential candidate: {idx in additional_indices_set}\n")
                            # f.write(f"Is Processed for Caption: True\n") # このループ内は常にTrue
                            f.write(f"Final outlier (Top concept not common): {is_final_outlier}\n")
                            f.write(f"Cropped Image Path: {path}\n")
                            f.write(f"Original Image Path: {original_path}\n")
                            f.write(f"Confidence={conf:.2f}, bbox={bbox}\n\n")

                            f.write("Generated caption:\n")
                            if desc is not None:
                                f.write(desc + "\n\n")
                            else:
                                f.write("No caption generated or regeneration failed.\n\n")

                            if is_failed_caption:
                                f.write("Caption generation failed. Similarity calculation skipped.\n")
                            elif not probs_dict:
                                f.write("No significant similarities found or calculation skipped.\n")
                            else:
                                f.write("Cosine Similarity w.r.t candidate labels:\n")
                                # 上位10件のみ表示するなど調整可能
                                sorted_probs = sorted(probs_dict.items(), key=lambda item: item[1], reverse=True)
                                for k_, v_ in sorted_probs[:10]:
                                    f.write(f"  {k_}: {v_:.4f}\n")
                                f.write(f"\nTop concept: {top_concept} (in common classes: {top_concept in common_classes})\n")

                    # 確率可視化 (probs_dict がある場合かつsave_probability_plotsがTrueの場合)
                    if len(probs_dict) > 0 and save_probability_plots:
                        plt.figure(figsize=(8, 4))
                        sorted_probs_keys = [k for k, v in sorted_probs[:10]]
                        sorted_probs_values = [v for k, v in sorted_probs[:10]]
                        plt.bar(sorted_probs_keys, sorted_probs_values, color='skyblue')
                        plt.xticks(rotation=45, ha='right') # ラベルが見切れないように調整
                        plt.title("Top 10 Concept Similarities")
                        plt.ylabel("Cosine Similarity")
                        plt.tight_layout()
                        prob_png = os.path.join(save_folder, f"{os.path.splitext(os.path.basename(path))[0]}_probs.png")
                        plt.savefig(prob_png)
                        plt.close()

                    # --- cv_info と motor_info の追加 (out_code != 0 の外に移動) ---
                    # top_concept が計算されている場合のみ実行
                    if top_concept:
                        is_yolo_potential = idx in additional_indices_set
                        sample_token = os.path.splitext(os.path.basename(original_path))[0]

                        if top_concept == 'construction_vehicle':
                            os.makedirs(cv_folder, exist_ok=True)
                            if os.path.exists(path):
                                shutil.copy(original_path, os.path.join(cv_folder, os.path.basename(path)))
                            cv_info.append({
                                'sample_token': sample_token,
                                'outlier_code': int(out_code),
                                'yolo_class': cls_name,
                                'is_yolo_potential': is_yolo_potential,
                                'original_path': original_path,
                                'confidence': float(conf),
                                'top_concept': top_concept
                            })
                        if top_concept == 'motorcycle':
                            os.makedirs(motor_folder, exist_ok=True)
                            if os.path.exists(path):
                                shutil.copy(original_path, os.path.join(motor_folder, os.path.basename(path)))
                            motor_info.append({
                                'sample_token': sample_token,
                                'outlier_code': int(out_code),
                                'yolo_class': cls_name,
                                'is_yolo_potential': is_yolo_potential,
                                'original_path': original_path,
                                'confidence': float(conf),
                                'top_concept': top_concept
                            })

                    # --- final_outliers / non_final_outliers への追加 ---
                    sample_token = os.path.splitext(os.path.basename(original_path))[0]
                    info_dict = {
                        'sample_token': sample_token,
                        'outlier_code': int(out_code),
                        'yolo_class': cls_name,
                        'is_yolo_potential': idx in additional_indices_set,
                        'original_path': original_path,
                        'confidence': float(conf),
                        'top_concept': top_concept, # None の可能性もある
                        'is_final_outlier': is_final_outlier
                    }

                    if is_final_outlier:
                        final_outliers.append(info_dict)
                    else:
                        # final でなくても処理対象になったものは non_final に記録
                        non_final_outliers.append(info_dict)

                # --- サブバッチ内のループ終了 ---

                # サブバッチ処理後にGPUメモリを解放
                del sub_batch_descriptions # 明示的に削除
                torch.cuda.empty_cache()

                # サブバッチごとに中間結果を保存 (保存するリストを更新)
                try:
                    sub_checkpoint_path = os.path.join(main_out_dir, f"sub_checkpoint.json") # ファイル名を一意に
                    sub_results = {
                        'timestamp': datetime.datetime.now().isoformat(),
                        'processed_count': processed_count,
                        'batch_start_index': i,
                        'final_outliers_in_batch': [fo for fo in final_outliers if fo['sample_token'] in [os.path.splitext(os.path.basename(all_cropped_info[sbi][4]))[0] for sbi in sub_batch_indices]], # このバッチで追加されたもの
                        'non_final_outliers_in_batch': [nfo for nfo in non_final_outliers if nfo['sample_token'] in [os.path.splitext(os.path.basename(all_cropped_info[sbi][4]))[0] for sbi in sub_batch_indices]], # このバッチで追加されたもの
                    }
                    with open(sub_checkpoint_path, 'w') as f:
                        json.dump(sub_results, f, indent=2)
                    print(f"サブバッチ {i // sub_batch_size + 1} の中間結果を保存しました: {sub_checkpoint_path}")
                except Exception as e:
                    print(f"サブバッチの中間結果保存エラー: {e}")

            # 100枚ごとにメモリ状況を確認
            if processed_count % 100 == 0:
                monitor_memory()
                import gc
                gc.collect()
                torch.cuda.empty_cache()
        else:
            print("処理対象のパスリストが空です。")

    except KeyboardInterrupt:
        print("Caption generation interrupted. Exiting...")
    except Exception as e:
        print(f"Error during caption generation loop: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        print("Cleaning up resources...")
        if 'model' in locals():
            del model
        if 'processor' in locals():
            del processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Save outlier information
        if save_outlier_list:
            try:
                # Final outliers
                if final_outliers:
                    final_outlier_path = os.path.join(main_out_dir, "final_outlier_results.json") # 名前変更
                    with open(final_outlier_path, 'w') as f:
                        json.dump(final_outliers, f, indent=2)
                    print(f"Final outlier information saved to {final_outlier_path}")

                # Non-final outliers
                if non_final_outliers:
                    non_final_outlier_path = os.path.join(main_out_dir, "processed_non_final_results.json") # 名前変更
                    with open(non_final_outlier_path, 'w') as f:
                        json.dump(non_final_outliers, f, indent=2)
                    print(f"Processed non-final object information saved to {non_final_outlier_path}")

                # Special category outliers
                if cv_info:
                    cv_info_path = os.path.join(main_out_dir, "construction_vehicle_results.json")
                    with open(cv_info_path, 'w') as f:
                        json.dump(cv_info, f, indent=2)
                    print(f"Construction vehicle information saved to {cv_info_path}")
                if motor_info:
                    motor_info_path = os.path.join(main_out_dir, "motorcycle_results.json")
                    with open(motor_info_path, 'w') as f:
                        json.dump(motor_info, f, indent=2)
                    print(f"Motorcycle information saved to {motor_info_path}")
            except Exception as e:
                print(f"Error saving outlier information: {e}")

    # 処理終了後に一時ファイルを削除
    if cleanup_temp_files:
        try:
            print("一時ファイルのクリーンアップを実行中...")
            temp_dirs = [
                os.path.join(run_specific_out_dir, folder)
                for folder in ["inlier_inlier", "if_outlier_only", "tsne_outlier_only"]
            ]
            
            for temp_dir in temp_dirs:
                if os.path.exists(temp_dir):
                    print(f"削除中: {temp_dir}")
                    shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"一時ファイル削除エラー: {e}")


###############################################################################
# Generate Feature Dictionary
###############################################################################
def generate_feature_dict(class_names, qwen_model_size="7B", prompt_type="important"):
    """
    クラス名のリストから特徴辞書を生成する
    
    Args:
        class_names: クラス名のリスト
        qwen_model_size: Qwenモデルのサイズ ("2B" or "7B")
        prompt_type: 生成する特徴の種類
    
    Returns:
        特徴辞書 {class_name: [features]}
    """
    print(f"Generating feature dictionary for {len(class_names)} classes using Qwen-{qwen_model_size}...")
    
    # 実際の実装では、ここでQwenモデルを使用して各クラスの重要な特徴を生成する
    feature_dict = {}
    
    # 単純なデモ実装として、各クラスのバリエーションを返す
    default_features = {
        "car": ["sedan", "suv", "hatchback", "sports car", "electric vehicle"],
        "truck": ["pickup truck", "delivery truck", "semi truck", "dump truck", "tow truck"],
        "bus": ["city bus", "school bus", "tour bus", "double decker bus", "minibus"],
        "pedestrian": ["walking person", "running person", "standing person", "person crossing street"],
        "motorcycle": ["sport bike", "cruiser", "scooter", "moped", "dirt bike"],
        "bicycle": ["mountain bike", "road bike", "city bike", "electric bike", "folding bike"],
        "traffic light": ["red light", "green light", "yellow light", "traffic signal", "pedestrian light"],
        "traffic sign": ["stop sign", "yield sign", "speed limit sign", "warning sign", "direction sign"]
    }
    
    # 与えられたクラス名に対して特徴を返す
    for class_name in class_names:
        if class_name in default_features:
            feature_dict[class_name] = default_features[class_name]
        else:
            # 未知のクラスの場合、クラス名自体を返す
            feature_dict[class_name] = [class_name]
    
    return feature_dict


###############################################################################
# Main
###############################################################################

concept_list = [
    "car", "truck", "construction_vehicle", "bus", "trailer", "barrier",
    "motorcycle", "bicycle", "pedestrian", "traffic_cone",
    "traffic_light", "traffic_sign", "road", "sidewalk", "building",
    "emergency_vehicle", "police_car", "ambulance", "fire_truck", "delivery_van",
    "taxi", "limousine", "golf_cart", "recreational_vehicle", "tractor",
    "snowplow", "street_sweeper", "garbage_truck", "military_vehicle", "food_truck",
    "go_kart", "segway", "hoverboard", "electric_scooter", "motorized_wheelchair",
    "parade_float", "horse_carriage", "trolley",
    "pedestrian_with_umbrella", "pedestrian_with_stroller", "pedestrian_with_shopping_cart",
    "pedestrian_with_walker", "pedestrian_with_wheelchair", "pedestrian_with_crutches",
    "pedestrian_with_bicycle", "pedestrian_with_luggage", "pedestrian_in_costume",
    "pedestrian_with_pet", "pedestrian_skating", "pedestrian_in_group",
    "pedestrian_dancing", "pedestrian_jogger",
    "dog", "cat", "deer", "raccoon", "horse", "squirrel", "bird", "wild_animal",
    "fallen_tree", "tire", "mattress", "cardboard_box", "shopping_cart",
    "luggage", "suitcase", "backpack", "furniture", "appliance", "trash_bin",
    "trash_bag", "construction_debris",
    "crane", "bulldozer", "excavator", "forklift", "cement_mixer", "road_roller",
    "backhoe", "cherry_picker", "construction_worker", "maintenance_worker",
    "jackhammer", "generator", "construction_materials", "portable_toilet",
    "pothole", "manhole", "speed_bump", "road_sign_fallen", "traffic_light_fallen",
    "road_blockade", "water_puddle", "barricade", "detour_sign",
    "construction_sign", "temporary_sign",
    "snow_pile", "flood_water", "fog_patch", "ice_patch", "fallen_power_line",
    "parade_performer", "film_crew", "street_performer", "protest_group",
    "unusual_cargo", "art_installation", "movie_prop", "sports_equipment",
    "hot_air_balloon", "drone", "bouncy_castle", "advertisement_mascot"
]

# シード値を固定する関数を追加
def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_optimal_batch_size(initial_size=8, min_size=1):
    """利用可能なGPUメモリに基づいて最適なバッチサイズを決定する"""
    if not torch.cuda.is_available():
        return initial_size
    
    # 現在の空きメモリを取得
    try:
        free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
        free_memory_gb = free_memory / (1024**3)
    except Exception as e:
        print(f"Error getting free memory: {e}")
        return initial_size
    
    # メモリに基づいてバッチサイズを調整
    if free_memory_gb > 10:
        return initial_size
    elif free_memory_gb > 6:
        return max(initial_size // 2, min_size)
    elif free_memory_gb > 3:
        return max(initial_size // 4, min_size)
    else:
        return min_size
    
    # 注: 実際の閾値はモデルサイズに応じて調整する必要があります

def main():
    parser = argparse.ArgumentParser(description="Detect rare images using object detection, CLIP, and outlier detection.")
    parser.add_argument("--baseline_folder", type=str, default="data_nuscenes/samples/CAM_FRONT", help="Path to the baseline image folder.")
    parser.add_argument("--new_images_folder", type=str, default=None, help="Path to the new image folder (defaults to baseline_folder).")
    parser.add_argument("--output_base_dir", type=str, default="rare_whole_architecture", help="Base directory for saving results.")
    parser.add_argument("--lim1", type=float, default=0.5, help="Start fraction for baseline/new image split.")
    parser.add_argument("--lim2", type=float, default=1.0, help="End fraction for baseline/new image split.")
    parser.add_argument("--qwen_model_size", type=str, default="2B", choices=["2B", "7B"], help="Size of the Qwen2VL model to use (2B or 7B).")
    parser.add_argument("--no_concept_list", action='store_true', help="Don't use the predefined concept list, use dynamic labels instead.")
    parser.add_argument("--threshold_percentile", type=int, default=80, help="Percentile threshold for t-SNE outlier detection (lower values detect more outliers).")
    parser.add_argument("--no_save_outlier_list", action='store_true', help="Do not save the outlier detection results to a JSON file.")
    parser.add_argument("--common_classes", nargs='+', 
                        default=["car", "pedestrian", "traffic_light"], #"traffic light", "traffic sign"], #"truck", "bus", 
                        help="List of common classes to filter out from outliers")
    parser.add_argument("--split_mode", type=str, choices=["standard", "10fold"], default="10fold", 
                        help="Data split mode: 'standard' for simple lim1/lim2 split or '10fold' for 10-part split with odd parts for baseline")
    parser.add_argument("--max_lim", type=float, default=1.0, 
                        help="Maximum fraction of images to use from dataset (default: 1.0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--only_additional_candidates", action='store_true', 
                        help="Only process additional candidates based on YOLO class names, ignore IF/tSNE outliers")
    parser.add_argument("--potential_classes", nargs='+', 
                        default=['construction_vehicle', 'motorcycle', 'bicycle', 'truck', 'bus', 'trailer',
                                'emergency_vehicle', 'police_car', 'ambulance', 'fire_truck',
                                'tractor', 'snowplow', 'garbage_truck', 'military_vehicle',
                                'fire_hydrant', 'train', 'boat'],
                        help="List of potential class names to consider as additional candidates")
    parser.add_argument("--no_save_crops", action='store_true', 
                        help="Do not save cropped images (reduces I/O time)")
    parser.add_argument("--no_save_descriptions", action='store_true', 
                        help="Do not save description text files (reduces I/O time)")
    parser.add_argument("--no_save_probability_plots", action='store_true', 
                        help="Do not save probability bar plots (reduces I/O time)")
    parser.add_argument("--minimal_io", action='store_true', 
                        help="Enable all I/O reduction options (equivalent to --no_save_crops --no_save_descriptions --no_save_probability_plots)")
    parser.add_argument("--cleanup_temp", action='store_true', 
                        help="処理終了後に一時ファイルを削除して容量を節約")
    
    args = parser.parse_args()

    # シード値を設定
    set_seed(args.seed)
    # args.baseline_folder = "projects/detect_rare_images/data/data/nuscenes_image"
    # args.new_images_folder = "projects/detect_rare_images/data/test_data"
    if args.new_images_folder is None:
        args.new_images_folder = args.baseline_folder
    else:
        args.split_mode = "standard"
        args.lim1 = 0.1
        args.lim2 = 1.0

    timestamp = datetime.datetime.now().strftime("%m%d_%H-%M-%S_JST")

    main_out_dir = os.path.join(args.output_base_dir, f"run_yolo_{args.split_mode}_{args.max_lim}_{timestamp}")
    os.makedirs(main_out_dir, exist_ok=True)
    print(f"Results will be saved in: {main_out_dir}")

    # Invert the logic to make concept_list the default
    use_concept_list = not args.no_concept_list

    print(f"Configuration:")
    print(f"  Baseline Folder: {args.baseline_folder}")
    print(f"  New Images Folder: {args.new_images_folder}")
    print(f"  Output Directory: {main_out_dir}")
    print(f"  Split Mode: {args.split_mode}")
    
    if args.split_mode == "standard":
        print(f"  Image Split: {args.lim1} to {args.lim2}")
        lim1 = args.lim1
        lim2 = args.lim2
        # 標準モードでは、既存のlim1とlim2をそのまま使用
        train_indices = None
        test_indices = None
    else:  # "10fold"モード
        print(f"  10-fold Split - max_lim: {args.max_lim}")
        print(f"  Using parts 3,7 for baseline, others for testing")
        lim1 = 0
        lim2 = args.max_lim
        # 後で使用するインデックスリストを初期化
        train_indices = []
        test_indices = []
        
    print(f"  Qwen Model Size: {args.qwen_model_size}")
    print(f"  Use Concept List: {use_concept_list}")
    print(f"  t-SNE Threshold Percentile: {args.threshold_percentile}")
    print(f"  Save Outlier List: {not args.no_save_outlier_list}")
    print(f"  Common Classes: {args.common_classes}")
    print(f"  Only Additional Candidates: {args.only_additional_candidates}")
    if args.only_additional_candidates:
        print(f"  Potential Classes: {args.potential_classes}")

    # max_limが指定されていない場合のデフォルト設定
    if args.max_lim == 1.0 and not args.minimal_io:
        save_crops = not args.no_save_crops
        save_descriptions = not args.no_save_descriptions
        save_probability_plots = not args.no_save_probability_plots
    else:
        # max_limが指定されている場合またはminimal_ioが指定されている場合は全てFalseに
        save_crops = not (args.no_save_crops or args.minimal_io)
        save_descriptions = not (args.no_save_descriptions or args.minimal_io)
        save_probability_plots = not (args.no_save_probability_plots or args.minimal_io)

    # 設定情報の表示に追加
    print(f"  Save Cropped Images: {save_crops}")
    print(f"  Save Description Files: {save_descriptions}")
    print(f"  Save Probability Plots: {save_probability_plots}")

    # 設定情報の保存（デバッグ用）
    config_path = os.path.join(main_out_dir, "config.json")
    config = vars(args)
    config["timestamp"] = timestamp
    config["device"] = get_device()
    
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        print(f"設定保存エラー: {e}")
    
    # 処理開始時間を記録
    start_time_overall = time.time()
    
    # 標準モードと10foldモードで処理を分ける
    if args.split_mode == "standard":
        # 従来の方法でベースラインモデルを学習
        clip_extractor, candidate_labels, baseline_features, baseline_tsne, class_names, iso_model = train_baseline_model(
            baseline_folder=args.baseline_folder,
            qwen_model_size=args.qwen_model_size,
            use_concept_list=use_concept_list,
            concept_list=concept_list,
            main_out_dir=main_out_dir,
            lim1=lim1,
            train_indices=None  # 標準モードでは特定のインデックスは使用しない
        )
        
        if baseline_features is None:
            return
        
        detect_new_images(
            new_images_folder=args.new_images_folder,
            clip_extractor=clip_extractor,
            candidate_labels=candidate_labels,
            qwen_model_size=args.qwen_model_size,
            main_out_dir=main_out_dir,
            common_classes=args.common_classes,
            baseline_features=baseline_features,
            baseline_tsne=baseline_tsne,
            baseline_class_names=class_names,
            iso_model=iso_model,
            threshold_percentile=args.threshold_percentile, 
            lim1=lim1,
            lim2=lim2,
            save_outlier_list=(not args.no_save_outlier_list),
            test_indices=None,
            only_additional_candidates=args.only_additional_candidates,
            potential_classes=args.potential_classes,
            save_crops=save_crops,
            save_descriptions=save_descriptions,
            save_probability_plots=save_probability_plots,
            cleanup_temp_files=args.cleanup_temp
        )
    else:  # "10fold"モード
        # 10分割のインデックスリストを作成
        all_files = [f for f in os.listdir(args.baseline_folder) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif"))]
        random.shuffle(all_files)
        
        # max_limまでのファイル数を取得
        max_idx = int(args.max_lim * len(all_files))
        limited_files = all_files[:max_idx]
        
        # 10分割
        fold_size = len(limited_files) // 10
        
        # パート3,7をトレーニング用に、その他をテスト用に分ける
        for i in range(10):
            start_idx = i * fold_size
            end_idx = (i + 1) * fold_size if i < 9 else len(limited_files)
            
            if i == 3 or i == 7:  # パート3,7をトレーニング用に
                train_indices.extend(range(start_idx, end_idx))
            else:  # その他のパートをテスト用に
                test_indices.extend(range(start_idx, end_idx))
        
        print(f"  Training on {len(train_indices)} images (parts 3,7)")
        print(f"  Testing on {len(test_indices)} images (other parts)")
        
        # 10fold方式でベースラインモデルを学習
        clip_extractor, candidate_labels, baseline_features, baseline_tsne, class_names, iso_model = train_baseline_model(
            baseline_folder=args.baseline_folder,
            qwen_model_size=args.qwen_model_size,
            use_concept_list=use_concept_list,
            concept_list=concept_list,
            main_out_dir=main_out_dir,
            lim1=1.0,  # limは使用せず、train_indicesを使用
            train_indices=train_indices
        )
        
        if baseline_features is None:
            return
        
        # 10fold方式で異常検出を実行
        detect_new_images(
            new_images_folder=args.new_images_folder,
            clip_extractor=clip_extractor,
            candidate_labels=candidate_labels,
            qwen_model_size=args.qwen_model_size,
            main_out_dir=main_out_dir,
            common_classes=args.common_classes,
            baseline_features=baseline_features,
            baseline_tsne=baseline_tsne,
            baseline_class_names=class_names,
            iso_model=iso_model,
            threshold_percentile=args.threshold_percentile, 
            lim1=0,  # limは使用せず、test_indicesを使用
            lim2=1.0,
            save_outlier_list=(not args.no_save_outlier_list),
            test_indices=test_indices,
            only_additional_candidates=args.only_additional_candidates,
            potential_classes=args.potential_classes,
            save_crops=save_crops,
            save_descriptions=save_descriptions,
            save_probability_plots=save_probability_plots,
            cleanup_temp_files=args.cleanup_temp
        )

    # 処理終了時間を記録
    end_time_overall = time.time()
    elapsed_time = end_time_overall - start_time_overall
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"Using device: {get_device()}")
    print(f"\nAll Done. 総処理時間: {int(hours)}時間{int(minutes)}分{seconds:.1f}秒")
    
    # 処理時間を記録
    try:
        timing_path = os.path.join(main_out_dir, "timing.txt")
        with open(timing_path, 'w') as f:
            f.write(f"開始時刻: {timestamp}\n")
            f.write(f"終了時刻: {datetime.datetime.now().strftime('%m%d_%H-%M-%S_JST')}\n")
            f.write(f"総処理時間: {int(hours)}時間{int(minutes)}分{seconds:.1f}秒\n")
    except Exception as e:
        print(f"処理時間記録エラー: {e}")


if __name__ == "__main__":
    main()

