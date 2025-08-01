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
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# NLTK のリソースをダウンロード (初回のみ必要)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# BLIP model
from transformers import BlipProcessor, BlipForConditionalGeneration

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
            
            # リサイズするオプション (ただし、検出精度は落ちる可能性がある)
            # 画像が大きすぎる場合のみリサイズ
            width, height = img.size
            max_size = 1280  # 最大サイズの閾値
            if width > max_size or height > max_size:
                # アスペクト比を維持したまま縮小
                if width > height:
                    new_width = max_size
                    new_height = int(height * (max_size / width))
                else:
                    new_height = max_size
                    new_width = int(width * (max_size / height))
                img = img.resize((new_width, new_height), Image.LANCZOS)
                print(f"Resized image from {width}x{height} to {new_width}x{new_height}")

            # Run inference with YOLO (パラメータ調整)
            results = self.model.predict(
                source=img,
                conf=conf_thres,
                iou=0.45,       # IoU閾値を調整
                max_det=100,    # 最大検出数を制限
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
# 2. Vision-Language Model (BLIP) for feature extraction and caption generation
###############################################################################
class BLIPFeatureExtractor:
    """
    Obtain feature vectors (embeddings) and captions for images using BLIP.
    """
    def __init__(self, model_name="Salesforce/blip-image-captioning-large", device=None):
        if device is None:
            device = get_device()
        self.device = device

        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name)
        self.model.eval()
        self.model.to(self.device)

    def get_image_embedding(self, image_path):
        """
        Get the BLIP image feature from the given image path.
        """
        pil_image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            # Extract visual features from the encoder
            outputs = self.model.vision_model(**inputs)
            # Use the pooled output as embedding：エンコーダーの出力からプールされた特徴表現を取得
            image_features = outputs.pooler_output
            
        # shape = (1, d)
        return image_features.cpu().numpy().squeeze(0)

    def get_text_embedding(self, text_list):
        """
        Get the BLIP text feature vectors for a list of texts.
        """
        all_embeddings = []
        
        for text in text_list:
            inputs = self.processor(text=text, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.text_encoder(**inputs)
                # Use the pooled output as embedding
                text_features = outputs.pooler_output
                
            all_embeddings.append(text_features.cpu().numpy().squeeze(0))
            
        # shape = (len(text_list), d)
        return np.array(all_embeddings)
    
    def generate_caption(self, image_path):
        """
        Generate image caption using BLIP model.
        """
        try:
            pil_image = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=pil_image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=50)
                caption = self.processor.decode(generated_ids[0], skip_special_tokens=True)
                
            return caption
        except Exception as e:
            print(f"Error generating caption with BLIP: {e}")
            return None

# Original CLIPFeatureExtractor class can be retained as fallback
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
        # shape = (1, d)
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
        # shape = (len(text_list), d)
        return text_features.cpu().numpy()


###############################################################################
# 3. text analysis by cosine similarity 
###############################################################################
def extract_concepts_from_captions(captions, top_n=100):
    """
    キャプションから頻出単語を抽出して概念リストを作成
    
    Args:
        captions: キャプションのリスト
        top_n: 抽出する上位単語数
    
    Returns:
        頻出単語のリスト
    """
    # ストップワード（一般的な単語）のロード
    stop_words = set(stopwords.words('english'))
    
    # 全単語のリスト
    all_words = []
    
    for caption in captions:
        if not caption:
            continue
            
        # 小文字に変換してトークン化
        words = word_tokenize(caption.lower())
        
        # ストップワードや記号を除去（アルファベット単語のみ残す）
        words = [word for word in words if word.isalpha() and word not in stop_words and len(word) > 2]
        
        all_words.extend(words)
    
    # 単語の出現頻度をカウント
    word_counts = Counter(all_words)
    
    # 頻出上位の単語を抽出
    top_concepts = [word for word, _ in word_counts.most_common(top_n)]
    
    return top_concepts


def parse_text_with_probability(base_text, candidate_labels, clip_extractor):
    """
    compare the base text with candidate labels using CLIP text embeddings,
    calculate cosine similarity, and return a dict of label-to-probability.
    """
    if not base_text:
        return {}

    # 1) Embed the main text and candidate labels with CLIP
    text_emb = clip_extractor.get_text_embedding([base_text])  # shape = (1, d)
    labels_emb = clip_extractor.get_text_embedding(candidate_labels)  # shape = (n, d)

    # 2) Compute cosine similarity
    text_emb_norm = text_emb / np.linalg.norm(text_emb, axis=1, keepdims=True)
    labels_emb_norm = labels_emb / np.linalg.norm(labels_emb, axis=1, keepdims=True)

    cos_sims = []
    for i in range(labels_emb_norm.shape[0]):
        sim = (text_emb_norm[0] * labels_emb_norm[i]).sum()
        cos_sims.append(sim)
    cos_sims = np.array(cos_sims)

    # 3) Apply softmax to produce probability-like values
    tensor_sims = torch.tensor(cos_sims)
    probs = cos_sims  # softmaxしない方がよいかも
    # 4) Create dictionary of label-to-probability, filtering out negative values
    label_probs = {candidate_labels[i]: float(probs[i]) for i in range(len(candidate_labels)) if probs[i] >= 0}
    label_probs = {k: v for k, v in sorted(label_probs.items(), key=lambda x: x[1], reverse=True)}
    return label_probs


###############################################################################
# 4. Image Captioning (Using BLIP or fallback to GPT-4o)
###############################################################################
def generate_description(image_path, blip_extractor=None, model=None, processor=None):
    """
    Get image caption using BLIP model with fallback to other models.
    """
    # First try using BLIP extractor if provided
    if blip_extractor is not None:
        caption = blip_extractor.generate_caption(image_path)
        if caption:
            return caption
    
    # Fallback to existing methods
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

        # Preparation for inference
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

        # Inference: Generation of the output
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
        print(f"LLaVA model error: {e}")
        # Fall back to OpenAI API
        try:
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
    iso = IsolationForest(contamination=0.2, random_state=42)
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
def train_baseline_model(baseline_folder="projects/detect_rare_images/data/data/nuscenes_image", train_ratio=0.7):
    detector = DetectorYOLO(model_path="yolov8l.pt")
    # Use BLIP model instead of CLIP for feature extraction and caption generation
    blip_extractor = BLIPFeatureExtractor(model_name="Salesforce/blip-image-captioning-large")
    # Keep CLIP as fallback
    clip_extractor = CLIPFeatureExtractor(model_name="openai/clip-vit-base-patch32")

    cropped_dir = "baseline_cropped_objects"
    os.makedirs(cropped_dir, exist_ok=True)

    all_files = [f for f in os.listdir(baseline_folder) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif"))]
    random.shuffle(all_files)
    # 全データのうちtrain_ratio分をトレーニング用に使用
    baseline_files = all_files[:int(train_ratio * len(all_files))]
    # 残りのファイルはテスト用に保存
    test_files = all_files[int(train_ratio * len(all_files)):]

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
    baseline_captions = []
    
    for (obj_path, cls_name, conf, bbox, _) in all_cropped_info:
        try:
            emb = blip_extractor.get_image_embedding(obj_path)
            caption = blip_extractor.generate_caption(obj_path)
            if caption:
                baseline_captions.append(caption)
        except Exception as e:
            print(f"Error with BLIP embedding, falling back to CLIP: {e}")
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

    print(f"Generating concepts from {len(baseline_captions)} baseline captions...")
    concept_list = extract_concepts_from_captions(baseline_captions, top_n=100)
    print(f"Top 20 concepts: {concept_list[:20]}")
    
    candidate_labels = list(set(class_names + concept_list))

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

    return blip_extractor, clip_extractor, candidate_labels, features, baseline_tsne, class_names, iso_model, test_files


###############################################################################
# detect_new_images
###############################################################################
def detect_new_images(
    new_images_folder,
    blip_extractor,
    clip_extractor,
    candidate_labels,
    out_dir="new_cropped_objects",
    baseline_features=None,
    baseline_tsne=None,
    baseline_class_names=None,
    iso_model=None,
    threshold_percentile=95,
    test_files=None,
    save_outliers=True
):
    detector = DetectorYOLO(model_path="yolov8l.pt")
    today = datetime.datetime.today().strftime("%m%d")
    out_dir = f"{out_dir}_{today}_blip"
    os.makedirs(out_dir, exist_ok=True)

    # テスト用ファイルのリストが渡された場合、そのリストを使用
    if test_files is not None:
        new_files = test_files
    else:
        all_files = [f for f in os.listdir(new_images_folder) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif"))]
        random.shuffle(all_files)
        new_files = all_files

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
        try:
            emb = blip_extractor.get_image_embedding(obj_path)
        except Exception as e:
            print(f"Error with BLIP embedding, falling back to CLIP: {e}")
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

    plt.figure(figsize=(18,12))
    class_counts = Counter(baseline_class_names)
    top_classes = [cls for cls, _ in class_counts.most_common(20)]
    cmap = plt.get_cmap('tab20')
    plt.scatter(baseline_tsne_2d[:,0], baseline_tsne_2d[:,1], c='lightgray', alpha=0.4, label="Baseline")

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
        plt.scatter(x_sub, y_sub, c=color_map[code][0], label=color_map[code][1], alpha=0.8)
    plt.title("Combined t-SNE Visualization (IsolationForest + t-SNE)")
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
    plt.tight_layout()
    plt.savefig("combined_tsne_class_if.png", bbox_inches='tight')
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

    model = None
    processor = None

    detected_images_dir = f"detected_images_blip_{today}"
    outliner_dir = "outliner"
    os.makedirs(detected_images_dir, exist_ok=True)
    os.makedirs(outliner_dir, exist_ok=True)
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

        desc = generate_description(path, blip_extractor, model, processor)
        if desc and not ("I am unable to" in desc or "I'm unable to analyze" in desc or "unable to process" in desc):
            probs_dict = parse_text_with_probability(desc, candidate_labels, clip_extractor)
        else:
            probs_dict = {}

        txt_path = os.path.join(save_folder, f"{os.path.splitext(base_name)[0]}_desc.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"Outlier code: {out_code} (0:inlier_inlier, 1:IFonly, 2:tSNEonly, 3:both)\n")
            f.write(f"Cropped Image Path: {path}\n")
            f.write(f"Original Image Path: {original_path}\n")
            f.write(f"Class={cls_name}, conf={conf:.2f}, bbox={bbox}\n\n")
            f.write("Generated caption:\n")
            f.write(desc + "\n\n")
            if probs_dict:
                f.write("Cosine Similarity with concepts and classes:\n")
                top_concepts = dict(list(probs_dict.items())[:10])
                for k_, v_ in top_concepts.items():
                    f.write(f"  {k_}: {v_:.4f}\n")
            else:
                f.write("Skipped similarity calculation due to caption generation failure.\n")

        if len(probs_dict) > 0:
            plt.figure(figsize=(10, 5))
            top_keys = list(probs_dict.keys())[:10]
            top_vals = [probs_dict[k] for k in top_keys]
            plt.bar(top_keys, top_vals, color='skyblue')
            plt.xticks(rotation=45, ha='right')
            plt.title("Cosine Similarity with Top Concepts")
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
        outlier_path = os.path.join(f"outlier_detection_results_blip_{today}.json")
        with open(outlier_path, 'w') as f:
            json.dump(outlier_info, f, indent=2)
        print(f"Outlier information saved to {outlier_path}")

###############################################################################
# Main
###############################################################################
def main():
    # パラメータ設定
    baseline_folder = "data_nuscenes/samples/CAM_FRONT"
    train_ratio = 0.7  # トレーニングデータの割合（0.0〜1.0）
    
    # 1) train baseline model (データの分割もここで行う)
    blip_extractor, clip_extractor, candidate_labels, baseline_features, baseline_tsne, class_names, iso_model, test_files = train_baseline_model(baseline_folder, train_ratio)
    if baseline_features is None:
        return
    
    # 2) detect rare images in new folder (test_filesを使用)
    detect_new_images(
        new_images_folder=baseline_folder,  # フォルダは同じだが、test_filesだけを使用
        blip_extractor=blip_extractor,
        clip_extractor=clip_extractor,
        candidate_labels=candidate_labels,
        baseline_features=baseline_features,
        baseline_tsne=baseline_tsne,
        baseline_class_names=class_names,
        iso_model=iso_model,
        threshold_percentile=80,
        test_files=test_files  # テスト用のファイルリストを渡す
    )

    print(f"Using device: {get_device()}")
    print("\nAll Done.")


if __name__ == "__main__":
    main()
