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
            img = Image.open(image_path).convert("RGB")
            
            width, height = img.size
            max_size = 1280
            if width > max_size or height > max_size:
                if width > height:
                    new_width = max_size
                    new_height = int(height * (max_size / width))
                else:
                    new_height = max_size
                    new_width = int(width * (max_size / height))
                img = img.resize((new_width, new_height), Image.LANCZOS)
                print(f"Resized image from {width}x{height} to {new_width}x{new_height}")

            results = self.model.predict(
                source=img,
                conf=conf_thres,
                iou=0.45,
                max_det=100,
                device=self.device
            )
            if len(results) == 0:
                return []

            boxes = results[0].boxes
            names = self.model.names
            cropped_info_list = []

            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())
                class_name = names[cls_id] if cls_id in names else f"class_{cls_id}"

                x1, y1, x2, y2 = xyxy
                cropped = img.crop((x1, y1, x2, y2))

                basename_no_ext = os.path.splitext(os.path.basename(image_path))[0]
                out_path = os.path.join(
                    out_dir, f"{basename_no_ext}_{i}_{class_name}.jpg"
                )
                cropped.save(out_path)
                cropped_info_list.append((out_path, class_name, conf, (x1, y1, x2, y2), image_path))

            return cropped_info_list
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
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
        
        for i in range(0, total, batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_images = [Image.open(path).convert("RGB") for path in batch_paths]
            
            inputs = self.processor(images=batch_images, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                batch_features = self.model.get_image_features(**inputs)
            
            batch_features = batch_features.cpu().numpy()
            all_embeddings.extend(batch_features)
        
        return np.array(all_embeddings)

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
                                "Analyze the following image and list up to 10 notable objects/scenarios."
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

def generate_descriptions_batch(image_paths, model, processor, batch_size=8):
    """
    Get image captions using Qwen model in batch processing.
    
    Args:
        image_paths: List of paths to images
        model: The Qwen2VL model
        processor: The processor for the model
        batch_size: Number of images to process in each batch
        
    Returns:
        List of descriptions corresponding to input image_paths
    """
    results = []
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        print(f"Processing caption batch {i//batch_size + 1}/{(len(image_paths)-1)//batch_size + 1}...")
        
        # Filter out paths with missing files
        valid_paths = []
        for path in batch_paths:
            if os.path.isfile(path):
                valid_paths.append(path)
            else:
                results.append(None)  # Add None for missing files
        
        if not valid_paths:
            continue
            
        # Prepare batch messages
        batch_messages = []
        max_tokens = 150
        
        for path in valid_paths:
            base64_image = encode_image(path)
            if not base64_image:
                batch_messages.append(None)
                continue
                
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
                                    f"Note that max tokens is {max_tokens}."
                        },
                    ],
                }
            ]
            batch_messages.append(message)
        
        # Skip empty batch
        if not batch_messages or all(msg is None for msg in batch_messages):
            continue
            
        try:
            # Process all valid images in batch
            all_image_inputs = []
            all_video_inputs = []
            all_texts = []
            
            for msg in batch_messages:
                if msg is None:
                    continue
                    
                text = processor.apply_chat_template(
                    msg, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs = process_vision_info(msg)
                
                all_texts.append(text)
                all_image_inputs.extend(image_inputs)  # Extend list of image inputs
                all_video_inputs.extend(video_inputs)  # Extend list of video inputs
                
            # Process batch inputs
            inputs = processor(
                text=all_texts,
                images=all_image_inputs,
                videos=all_video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(device)
            
            # Generate outputs
            generated_ids = model.generate(**inputs, max_new_tokens=max_tokens)
            
            # Process outputs
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_texts = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            # Add results for this batch
            valid_index = 0
            for path in batch_paths:
                if os.path.isfile(path):
                    results.append(output_texts[valid_index] if valid_index < len(output_texts) else None)
                    valid_index += 1
                # Missing files already have None in results
                
        except Exception as e:
            print(f"Qwen model error generating captions for batch: {e}")
            # Add None for all images in this failed batch
            for _ in range(len(batch_paths) - (len(results) - i)):
                results.append(None)
    
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
    test_indices=None
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

    # アウトライアに対してのみ処理を行う
    outlier_indices = np.where(combined_outlier_code != 0)[0]
    print(f"\n潜在的なアウトライア: {len(outlier_indices)}個 処理中...")

    # アウトライアだけを処理する場合は、QwenモデルをGPUに維持する前に確認
    if len(outlier_indices) > 0:
        # Qwenモデルのロード（アウトライアがある場合のみ）
        model_name = f"Qwen/Qwen2-VL-{qwen_model_size}-Instruct"
        print(f"{model_name}モデルをロード中...")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(model_name)
    else:
        print("アウトライアが検出されなかったため、キャプション生成をスキップします")

    detected_images_dir = os.path.join(main_out_dir, "detected_images_overall")
    os.makedirs(detected_images_dir, exist_ok=True)

    outlier_info = []
    cv_info = []
    motor_info = []

    cv_folder = os.path.join(main_out_dir, "construction_vehicle")
    os.makedirs(cv_folder, exist_ok=True)
    motor_folder = os.path.join(main_out_dir, "motorcycle")
    os.makedirs(motor_folder, exist_ok=True)

    final_folder = os.path.join(main_out_dir, "final_outliers")
    os.makedirs(final_folder, exist_ok=True)
    
    total_outlier_count = len([x for x in combined_outlier_code if x != 0])
    processed_outliers = 0
    start_time_caption = time.time()
    try:
        # アウトライア検出されたパスをリストアップ
        outlier_paths = [all_cropped_info[i][0] for i in outlier_indices]
        
        # バッチサイズを設定（メモリに合わせて調整）
        caption_batch_size = 3
        
        # バッチ処理でキャプション生成
        if outlier_paths:
            print(f"バッチ処理で{len(outlier_paths)}個のアウトライアのキャプションを生成中...")
            batch_descriptions = generate_descriptions_batch(outlier_paths, model, processor, batch_size=caption_batch_size)
        
            # 生成されたキャプションを使用して処理を続行
            for idx, desc in zip(outlier_indices, batch_descriptions):
                out_code, (path, cls_name, conf, bbox, original_path) = combined_outlier_code[idx], all_cropped_info[idx]
                folder_name = out_folders.get(out_code, "others")
                save_folder = os.path.join(run_specific_out_dir, folder_name)
                
                # Initialize is_final_outlier as False
                is_final_outlier = False
                top_concept = None
                # ToDo
                
# Only process potential outliers
            if out_code != 0:
                # Show progress for caption generation
                processed_outliers += 1
                if processed_outliers % 100 == 0 or processed_outliers == total_outlier_count:
                    elapsed = time.time() - start_time_caption
                    progress = (processed_outliers / total_outlier_count) * 100
                    eta = (elapsed / processed_outliers) * (total_outlier_count - processed_outliers) if processed_outliers > 0 else 0
                    elapsed_min, elapsed_sec = divmod(elapsed, 60)
                    eta_min, eta_sec = divmod(eta, 60)
                    print(f"Caption progress: {progress:.1f}% ({processed_outliers}/{total_outlier_count}) | " 
                        f"経過時間: {int(elapsed_min)}分{elapsed_sec:.1f}秒 | 残り時間: {int(eta_min)}分{eta_sec:.1f}秒")
                
                # Generate caption for potential outliers
                desc = generate_description(path, model, processor)
                
                desc_lower = desc.lower() if desc else ""
                if not desc or "unable to" in desc_lower or "i'm unable to" in desc_lower or "cannot process" in desc_lower:
                    print(f"Caption generation failed or indicated inability for {path}. Skipping similarity calculation.")
                    probs_dict = {}
                else:
                    probs_dict = parse_text_with_probability(desc, candidate_labels, clip_extractor) 
                    
                    # Check if the top concept is not in common classes
                    if probs_dict:
                        top_concept = next(iter(probs_dict))  # Get first key (highest similarity)
                        if top_concept not in common_classes:
                            is_final_outlier = True
                
                # Copy files based on final outlier status
                if is_final_outlier:
                    save_img = os.path.join(detected_images_dir, os.path.basename(original_path))
                    shutil.copy(original_path, save_img)
                
                    # 
                    shutil.copy(path, os.path.join(save_folder, os.path.basename(path)))
                
                # Create description file for all objects
                txt_path = os.path.join(save_folder, f"{os.path.splitext(os.path.basename(path))[0]}_desc.txt")
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(f"Outlier code: {out_code} (0:inlier_inlier, 1:IFonly, 2:tSNEonly, 3:both)\n")
                    f.write(f"Final outlier: {is_final_outlier}\n")
                    f.write(f"Cropped Image Path: {path}\n")
                    f.write(f"Original Image Path: {original_path}\n")
                    f.write(f"Class={cls_name}, conf={conf:.2f}, bbox={bbox}\n\n")
                    
                    if out_code != 0:  # Only for potential outliers
                        f.write("Generated caption:\n")
                        f.write(desc + "\n\n")
                        if probs_dict:
                            f.write("Cosine Similarity w.r.t candidate labels:\n")
                            for k_, v_ in probs_dict.items():
                                f.write(f"  {k_}: {v_:.4f}\n")
                            f.write(f"\nTop concept: {top_concept} (in common classes: {top_concept in common_classes})\n")
                        else:
                            f.write("Skipped similarity calculation due to caption generation failure.\n")

                # Create probability visualization for non-inliers with valid probs
                if out_code != 0 and len(probs_dict) > 0:
                    plt.figure(figsize=(8, 4))
                    plt.bar(list(probs_dict.keys())[:10], list(probs_dict.values())[:10], color='skyblue')
                    plt.xticks(rotation=45)
                    plt.title("Top 10 Concept Similarities")
                    plt.tight_layout()
                    prob_png = os.path.join(save_folder, f"{os.path.splitext(os.path.basename(path))[0]}_probs.png")
                    plt.savefig(prob_png)
                    plt.close()
                if top_concept:
                    if top_concept == 'construction_vehicle':
                        shutil.copy(path, os.path.join(cv_folder, os.path.basename(path)))
                        cv_info.append({
                            'sample_token': os.path.splitext(os.path.basename(original_path))[0],
                            'outlier_code': int(out_code),
                            'original_path': original_path,
                            'class_name': cls_name,
                            'confidence': float(conf),
                            'top_concept': top_concept
                        })
                    if top_concept == 'motorcycle':
                        shutil.copy(path, os.path.join(motor_folder, os.path.basename(path)))
                        motor_info.append({
                            'sample_token': os.path.splitext(os.path.basename(original_path))[0],
                            'outlier_code': int(out_code),
                            'original_path': original_path,
                            'class_name': cls_name,
                            'confidence': float(conf),
                            'top_concept': top_concept
                        })
                # Add to outlier info if it's a final outlier
                if is_final_outlier:
                    sample_token = os.path.splitext(os.path.basename(original_path))[0]
                    outlier_info.append({
                        'sample_token': sample_token,
                        'outlier_code': int(out_code),
                        'original_path': original_path,
                        'class_name': cls_name,
                        'confidence': float(conf),
                        'top_concept': top_concept
                    })
                elif out_code != 0:
                    # Add to secondary outlier list for non-final outliers
                    outlier_info.append({
                        'sample_token': os.path.splitext(os.path.basename(original_path))[0],
                        'outlier_code': int(out_code),
                        'original_path': original_path,
                        'class_name': cls_name,
                        'confidence': float(conf),
                        'top_concept': top_concept,
                        'is_final_outlier': False
                    })
                
                # Create separate lists for final and non-final outliers
                final_outliers = [item for item in outlier_info if item.get('is_final_outlier', True)]
                non_final_outliers = [item for item in outlier_info if not item.get('is_final_outlier', True)]
    except KeyboardInterrupt:
        print("Caption generation interrupted. Exiting...")
    except Exception as e:
        print(f"Error during caption generation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        if model:
            del model
        if processor:
            del processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    # Save outlier information if requested
        if save_outlier_list:
            if final_outliers:
                final_outlier_path = os.path.join(main_out_dir, "outlier_detection_results.json")
                with open(final_outlier_path, 'w') as f:
                    json.dump(final_outliers, f, indent=2)
                print(f"Final outlier information saved to {final_outlier_path}")
            
            if non_final_outliers:
                non_final_outlier_path = os.path.join(main_out_dir, "potential_outlier_results.json")
                with open(non_final_outlier_path, 'w') as f:
                    json.dump(non_final_outliers, f, indent=2)
                print(f"Potential outlier information saved to {non_final_outlier_path}")

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

    del model
    del processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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

    args = parser.parse_args()

    # シード値を設定
    set_seed(args.seed)

    if args.new_images_folder is None:
        args.new_images_folder = args.baseline_folder

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
        print(f"  Using odd parts (1,3,5,7,9) for baseline, even parts (0,2,4,6,8) for testing")
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
            test_indices=None  # 標準モードでは特定のインデックスは使用しない
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
        
        # 奇数パート(1,3,5,7,9)をトレーニング用に、偶数パート(0,2,4,6,8)をテスト用に分ける
        for i in range(10):
            start_idx = i * fold_size
            end_idx = (i + 1) * fold_size if i < 9 else len(limited_files)
            
            if i == 3 or i == 7: #i % 2 == 1:  # 奇数パート (インデックスは0始まりなのでi%2==1が奇数)
                train_indices.extend(range(start_idx, end_idx))
            else:  # 偶数パート
                test_indices.extend(range(start_idx, end_idx))
        
        print(f"  Training on {len(train_indices)} images (odd parts)")
        print(f"  Testing on {len(test_indices)} images (even parts)")
        
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
            test_indices=test_indices
        )

    print(f"Using device: {get_device()}")
    print("\nAll Done.")


if __name__ == "__main__":
    main()

