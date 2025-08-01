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
# from transformers import AutoProcessor, AutoModel, LlavaForConditionalGeneration
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from ultralytics.nn.tasks import DetectionModel
from collections import Counter
import time  # 追加: APIレート制限のためのsleep用

# YOLO (Ultralytics)
try:
    import ultralytics
    from ultralytics import YOLO
except ImportError:
    print("Ultralytics YOLO not found. Install via 'pip install ultralytics'.")

# transformers
try:
    import openai  # OpenAI API
except ImportError:
    print("OpenAI API not found. If you want to use GPT-based captioning, do 'pip install openai'.")

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
    
    if not isinstance(base_text, str) or not candidate_labels or not clip_extractor:
        print(f"Invalid input: base_text_type={type(base_text)}, candidate_labels_len={len(candidate_labels) if candidate_labels else 0}")
        return {}

    try:
        import re
        # 正規表現パターンを改善 - 末尾の項目もしっかり捕捉
        list_items = re.findall(r'\d+\.\s+(.*?)(?=\d+\.|$)', base_text, re.DOTALL)
        if list_items:
            # 各項目の不要な説明文や改行を削除
            processed_items = []
            for item in list_items:
                # ハイフンの後の説明部分や余計な修飾語を削除
                clean_item = re.sub(r'\s+-\s+.*$', '', item)
                # 改行と不要な空白を削除
                clean_item = re.sub(r'\s+', ' ', clean_item).strip()
                if clean_item:
                    processed_items.append(clean_item)
            
            processed_text = " ".join(processed_items)
        else:
            processed_text = base_text
            
        print(f"Processed text for semantic analysis: {processed_text[:100]}..." if len(processed_text) > 100 else processed_text)
        
        text_emb = clip_extractor.get_text_embedding([processed_text])
        labels_emb = clip_extractor.get_text_embedding(candidate_labels)

        text_emb_norm = text_emb / np.linalg.norm(text_emb, axis=1, keepdims=True)
        labels_emb_norm = labels_emb / np.linalg.norm(labels_emb, axis=1, keepdims=True)

        cos_sims = []
        for i in range(labels_emb_norm.shape[0]):
            sim = (text_emb_norm[0] * labels_emb_norm[i]).sum()
            cos_sims.append(sim)
        cos_sims = np.array(cos_sims)

        tensor_sims = torch.tensor(cos_sims)
        probs = softmax(tensor_sims, dim=0).numpy()
        
        label_probs = {candidate_labels[i]: float(probs[i]) for i in range(len(candidate_labels)) if probs[i] >= 0}
        label_probs = {k: v for k, v in sorted(label_probs.items(), key=lambda x: x[1], reverse=True)}
        return label_probs
        
    except Exception as e:
        print(f"Error in parse_text_with_probability: {e}")
        return {}

###############################################################################
# 4. Image Captioning (GPT-4o)
###############################################################################
def generate_description(image_path, model, processor):
    """
    get image caption using LLaVA model.
    
    """
    base64_image = encode_image(image_path)
    if not base64_image:
        return None
                
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

        generated_ids = model.generate(**inputs, max_new_tokens=max_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        response = output_text[0] if output_text and len(output_text) > 0 else ""
        print("Raw model response:", response) #[:500] + "..." if len(response) > 100 else response)
        return response
        
    except Exception as e:
        print(f"LLaVA model error: {e}")
        try:
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
            
            response = client.chat.completions.create(
                model="gpt-4o-mini", 
                messages=messages,
                max_tokens=150,
                temperature=0.5
            )
            description = response.choices[0].message.content.strip()
            return description
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return None

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
# 特徴辞書を生成する関数
###############################################################################
def generate_feature_dict(class_names, prompt_type="important", model=None, processor=None):
    """
    特定のクラス名に関連する特徴をQwenモデルを使用して生成し、辞書にまとめる関数
    
    Args:
        class_names: クラス名のリスト
        prompt_type: 使用するプロンプトタイプ ("important", "superclass", "around")
        model: Qwenモデル (指定がなければ新たにロード)
        processor: Qwenプロセッサ (指定がなければ新たにロード)
        
    Returns:
        feature_dict: クラス名をキー、関連特徴のリストを値とする辞書
    """
    if model is None or processor is None:
        print("Loading Qwen2VL model for feature generation...")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct", torch_dtype="auto", device_map="auto"
        )
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    
    prompts = {
        "important": "List the most important features for recognizing something as a \"goldfish\":\n\n-bright orange color\n-a small, round body\n-a long, flowing tail\n-a small mouth\n-orange fins\n\nList the most important features for recognizing something as a \"beerglass\":\n\n-a tall, cylindrical shape\n-clear or translucent color\n-opening at the top\n-a sturdy base\n-a handle\n\nList the most important features for recognizing something as a \"{}\":",
        "superclass": "Give superclasses for the word \"tench\":\n\n-fish\n-vertebrate\n-animal\n\nGive superclasses for the word \"beer glass\":\n\n-glass\n-container\n-object\n\nGive superclasses for the word \"{}\":",
        "around": "List the things most commonly seen around a \"tench\":\n\n- a pond\n-fish\n-a net\n-a rod\n-a reel\n-a hook\n-bait\n\nList the things most commonly seen around a \"beer glass\":\n\n- beer\n-a bar\n-a coaster\n-a napkin\n-a straw\n-a lime\n-a person\n\nList the things most commonly seen around a \"{}\":"
    }
    
    base_prompt = prompts[prompt_type]
    INTERVAL = 1.0
    
    feature_dict = {}
    unique_classes = list(set(class_names))
    print(f"Generating features for {len(unique_classes)} unique classes using Qwen2VL model...")
    
    system_prompt = "You are an assistant that provides visual descriptions of objects. Use only adjectives and nouns in your description. Ensure each description is unique, short, and direct. Do not use qualifiers like 'typically', 'generally', or similar words."
    
    for i, label in enumerate(unique_classes):
        feature_dict[label] = set()
        print(f"\nProcessing {i+1}/{len(unique_classes)}: {label}")
        
        try:
            for _ in range(2):
                messages = [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": base_prompt.format(label)
                    }
                ]
                
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                
                image_inputs, video_inputs = [], []
                
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = inputs.to(device)
                
                max_tokens = 256
                generated_ids = model.generate(**inputs, max_new_tokens=max_tokens)
                
                try:
                    # Make sure lengths match and handle edge cases safely
                    generated_ids_trimmed = []
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids):
                        if len(in_ids) <= len(out_ids):
                            generated_ids_trimmed.append(out_ids[len(in_ids):])
                        else:
                            # Handle case where in_ids is longer than out_ids
                            generated_ids_trimmed.append(out_ids)
                            
                    output_text = processor.batch_decode(
                        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )
                    
                    if output_text and len(output_text) > 0:
                        features = output_text[0]
                        if "\n-" in features:
                            features = features.split("\n-")
                        elif "\n" in features:
                            features = features.split("\n")
                        else:
                            features = [features]
                        
                        features = [feat.replace("\n", "") for feat in features]
                        features = [feat.strip() for feat in features]
                        features = [feat for feat in features if len(feat) > 0]
                        features = [feat[1:] if feat.startswith("-") else feat for feat in features]
                        feature_dict[label].update(features)
                    else:
                        print(f"Warning: Empty response received for {label}")
                        feature_dict[label].update([label]) # デフォルト値としてラベルを追加
                
                except IndexError as e:
                    print(f"Error processing model output for {label}: {e}")
                except Exception as e:
                    print(f"Unexpected error processing model output for {label}: {e}")
                
                time.sleep(INTERVAL)
            
            feature_dict[label].add(label)
            feature_dict[label] = sorted(list(feature_dict[label]))
            
        except Exception as e:
            print(f"Error generating features for {label}: {e}")
            feature_dict[label] = [label]
    
    return feature_dict

###############################################################################
# train baseline model
###############################################################################
def train_baseline_model(baseline_folder="projects/detect_rare_images/data/data/nuscenes_image", lim1=0.8):
    detector = DetectorYOLO(model_path="yolov8l.pt")
    clip_extractor = CLIPFeatureExtractor(model_name="openai/clip-vit-base-patch32")

    cropped_dir = "baseline_cropped_objects"
    os.makedirs(cropped_dir, exist_ok=True)

    all_files = [f for f in os.listdir(baseline_folder) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif"))]
    random.shuffle(all_files)
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
        return None, None, None, None

    iso_model = train_isolation_forest(features, contamination=0.2)

    unique_labels = list(set(class_names))
    label2id = {lb: i for i, lb in enumerate(unique_labels)}
    label_ids = [label2id[lb] for lb in class_names]

    candidate_labels = list(set(class_names))
    
    print("Loading Qwen2VL model for feature generation and captioning...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct", torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    
    top_classes = [cls for cls, _ in Counter(class_names).most_common(10)]
    print(f"Generating feature dictionary for top {len(top_classes)} classes...")
    feature_dict = generate_feature_dict(top_classes, prompt_type="important", model=model, processor=processor)
    
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
    plt.savefig("baseline_tsne.png")
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
    plt.savefig("baseline_tsne_class.png", bbox_inches='tight')
    plt.close()

    return clip_extractor, extended_candidate_labels, features, baseline_tsne, class_names, iso_model


###############################################################################
# detect_new_images
###############################################################################
def detect_new_images(
    new_images_folder,
    clip_extractor,
    candidate_labels,
    out_dir="new_cropped_objects",
    baseline_features=None,
    baseline_tsne=None,
    baseline_class_names=None,
    iso_model=None,
    threshold_percentile=95,
    lim1=0.8,
    lim2=1.0,
    save_outliers=True
):
    detector = DetectorYOLO(model_path="yolov8l.pt")
    today = datetime.datetime.today().strftime("%m%d")
    out_dir = f"{out_dir}_{today}_lim{lim1}-{lim2}"
    os.makedirs(out_dir, exist_ok=True)

    all_files = [f for f in os.listdir(new_images_folder) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif"))]
    random.shuffle(all_files)
    new_files = all_files[int(lim1 * len(all_files)): int(lim2 * len(all_files))]

    all_cropped_info = []
    for file_name in new_files:
        try:
            image_path = os.path.join(new_images_folder, file_name)
            cropped_info = detector.detect_and_crop(
                image_path, out_dir=out_dir, conf_thres=0.3
            )
            all_cropped_info.extend(cropped_info)
        except Exception as e:
            print(f"Error processing new image {file_name}: {e}")
            continue

    features_new = []
    for (obj_path, cls_name, conf, bbox, _) in all_cropped_info:
        emb = clip_extractor.get_image_embedding(obj_path)
        features_new.append(emb)
    features_new = np.array(features_new)

    if len(features_new) == 0:
        print("No features extracted from new images. Exiting...")
        return
    combined_features = np.vstack([baseline_features, features_new])

    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    combined_tsne = tsne.fit_transform(combined_features)
    baseline_tsne_2d = combined_tsne[:len(baseline_features)]
    new_tsne = combined_tsne[len(baseline_features):]

    nbrs = NearestNeighbors(n_neighbors=10).fit(baseline_tsne)
    distances, indices = nbrs.kneighbors(new_tsne)
    avg_distances = distances.mean(axis=1)
    threshold_tsne = np.percentile(avg_distances, threshold_percentile)
    tsne_outliers = (avg_distances > threshold_tsne).astype(int)

    if_labels = predict_outliers(features_new, iso_model)
    if_outliers = ((if_labels==-1).astype(int))
    combined_outlier_code = tsne_outliers*2 + if_outliers

    vis_dir = os.path.join(out_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

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
    plt.savefig(os.path.join(vis_dir, "combined_tsne_class_if.png"), bbox_inches='tight')
    plt.close()

    out_folders = {
        0: "inlier_inlier",
        1: "if_outlier_only",
        2: "tsne_outlier_only",
        3: "both_outlier"
    }
    for k, v in out_folders.items():
        folder_path = os.path.join(out_dir, v)
        os.makedirs(folder_path, exist_ok=True)

# 9) 画像の振り分けと説明出力
    # Load the model in full precision to avoid LayerNormKernelImpl error
    # model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", device_map="auto")
    # processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct", torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

    for code, folder in out_folders.items():
        os.makedirs(os.path.join(out_dir, folder), exist_ok=True)

    detected_images_dir = "detected_images"
    outliner_dir = "outliner"
    os.makedirs(detected_images_dir, exist_ok=True)
    os.makedirs(outliner_dir, exist_ok=True)
    
    error_text = ["I am unable to", "I'm unable to analyze", "unable to process", "I can't", "I cannot"]

    for out_code, (path, cls_name, conf, bbox, original_path) in zip(combined_outlier_code, all_cropped_info):
        folder_name = out_folders.get(out_code, "others")
        save_folder = os.path.join(out_dir, folder_name)
        
        shutil.copy(original_path, save_folder)
        
        if out_code != 0:
            save_img = os.path.join(detected_images_dir, os.path.basename(original_path))
            shutil.copy(original_path, save_img)

        base_name = os.path.basename(path)
        dst_path = os.path.join(save_folder, base_name)
        shutil.copy(path, dst_path)

        desc = generate_description(path, model, processor)

        is_error = desc is None
        if desc is not None:
            is_error = any(err_text in desc for err_text in error_text)
        
        if is_error:
            probs_dict = {}
        else:
            probs_dict = parse_text_with_probability(desc, candidate_labels, clip_extractor)

        txt_path = os.path.join(save_folder, f"{os.path.splitext(base_name)[0]}_desc.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"Outlier code: {out_code} (0:inlier_inlier, 1:IFonly, 2:tSNEonly, 3:both)\n")
            f.write(f"Cropped Image Path: {path}\n")
            f.write(f"Original Image Path: {original_path}\n")
            f.write(f"Class={cls_name}, conf={conf:.2f}, bbox={bbox}\n\n")
            f.write("Generated caption:\n")
            if desc:
                f.write(desc + "\n\n")
            else:
                f.write("Caption generation failed.\n\n")
                
            if probs_dict:
                f.write("Cosine Similarity w.r.t candidate labels:\n")
                for k_, v_ in probs_dict.items():
                    f.write(f"  {k_}: {v_:.4f}\n")
            else:
                f.write("Skipped similarity calculation due to caption generation failure or empty result.\n")

        if len(probs_dict) > 0:
            plt.figure(figsize=(8, 4))
            plt.bar(probs_dict.keys(), probs_dict.values(), color='skyblue')
            plt.xticks(rotation=45)
            plt.title("Cosine Similarity")
            plt.tight_layout()
            prob_png = os.path.join(save_folder, f"{os.path.splitext(base_name)[0]}_probs.png")
            plt.savefig(prob_png)
            plt.close()
    if save_outliers:
        outlier_info = []
        for out_code, (path, cls_name, conf, bbox, original_path) in zip(combined_outlier_code, all_cropped_info):
            if out_code != 0:
                sample_token = os.path.splitext(os.path.basename(original_path))[0]
                outlier_info.append({
                    'sample_token': sample_token,
                    'outlier_code': int(out_code),
                    'original_path': original_path,
                    'class_name': cls_name,
                    'confidence': float(conf)
                })
        DATE = datetime.datetime.today().strftime("%m%d")
        outlier_path = os.path.join(f"outlier_detection_results_whole_arch_{DATE}_lim{lim1}_tolim{lim2}.json")
        with open(outlier_path, 'w') as f:
            json.dump(outlier_info, f, indent=2)
        print(f"Outlier information saved to {outlier_path}")


###############################################################################
# Main
###############################################################################
def main():
# 1) train baseline model
    baseline_folder = "data_nuscenes/samples/CAM_FRONT" #"projects/detect_rare_images/data/test_data" # #/nuscenes
    lim1 = 0.02 #0.5 #0.2 #0.8
    lim2 = 0.03 #1.0 #0.3 #1.0
    clip_extractor, candidate_labels, baseline_features, baseline_tsne, class_names, iso_model = train_baseline_model(baseline_folder, lim1)
    if baseline_features is None:
        return
    
# 2) detect rare images in new folder
    new_images_folder = baseline_folder #"data_nuscenes/samples/CAM_FRONT" #"projects/detect_rare_images/data/test_data"
    detect_new_images(
        new_images_folder=new_images_folder,
        clip_extractor=clip_extractor,
        candidate_labels=candidate_labels,
        baseline_features=baseline_features,
        baseline_tsne=baseline_tsne,
        baseline_class_names=class_names,
        iso_model=iso_model,
        threshold_percentile=80,
        lim1=lim1,
        lim2=lim2
    )

    print(f"Using device: {get_device()}")
    print("\nAll Done.")


if __name__ == "__main__":
    main()
