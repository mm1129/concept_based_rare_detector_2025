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
        # if torch.cuda.device_count() > 1: #>1
        #     return "cuda:0"  # Use the first GPU
        # else:
        #     return "cuda"
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
        # torch.serialization.add_safe_globals([DetectionModel])
        
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
# 3. text analysis by entity extraction and probability calculation
###############################################################################
def parse_text_with_probability(base_text, candidate_labels, clip_extractor):
    """
    Extract entities and concepts from the description text and calculate
    the probability of each entity/concept appearing in the image.
    """
    if not base_text:
        return {}
    
    try:
        # NLPライブラリをインポート
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
        except:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            nlp = spacy.load("en_core_web_sm")
    except ImportError:
        print("Spacy not installed. Installing...")
        try:
            import subprocess
            subprocess.run(["pip", "install", "spacy"])
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            import spacy
            nlp = spacy.load("en_core_web_sm")
        except Exception as e:
            print(f"Failed to install spacy: {e}")
            nlp = None
    
    # テキスト解析によるエンティティ抽出
    entities = []
    
    if nlp:
        # SpaCyを使用して名詞や名詞句を抽出
        doc = nlp(base_text)
        
        # 名詞や名詞句を抽出
        for chunk in doc.noun_chunks:
            entities.append(chunk.text.lower().strip())
        
        # 固有表現も抽出
        for ent in doc.ents:
            entities.append(ent.text.lower().strip())
            
        # 名詞を抽出
        for token in doc:
            if token.pos_ == "NOUN" or token.pos_ == "PROPN":
                entities.append(token.text.lower().strip())
    
    # バックアップ方法: 単純に文を分割し、キーワードを抽出
    if not entities:
        # 文を分割
        sentences = base_text.split('.')
        for sentence in sentences:
            # カンマや箇条書きで区切られた部分を抽出
            parts = [p.strip() for p in sentence.split(',')]
            parts += [p.strip() for p in sentence.split(' and ')]
            parts += [p.strip() for p in sentence.split('-')]
            
            # 箇条書きの番号や記号を削除
            filtered_parts = []
            for part in parts:
                # 先頭の数字や記号を削除
                cleaned = part.strip().lstrip('0123456789.- ').strip()
                if cleaned:
                    filtered_parts.append(cleaned.lower())
            
            entities.extend(filtered_parts)
    
    # 重複を削除して一意なエンティティリストを作成
    entities = list(set(entities))
    
    # 短すぎる単語、ストップワードを除去
    stopwords = ["the", "a", "an", "and", "or", "but", "if", "because", "as", "what", "which", 
                "this", "that", "these", "those", "then", "just", "so", "than", "such", 
                "when", "while", "where", "how", "why", "whether", "with", "without", "of", "in", "on"]
    entities = [e for e in entities if len(e) > 2 and e not in stopwords]
    
    # candidate_labelsとの関連性を確認
    relevant_entities = []
    for entity in entities:
        for label in candidate_labels:
            if label.lower() in entity or entity in label.lower():
                relevant_entities.append(entity)
                break
    
    # relevant_entitiesが少ない場合は、元のentitiesから追加
    if len(relevant_entities) < min(5, len(entities)):
        additional = [e for e in entities if e not in relevant_entities]
        relevant_entities.extend(additional[:5-len(relevant_entities)])
    
    # 頻度を算出 (元のテキスト内での出現回数をカウント)
    counts = {}
    total_words = len(base_text.split())
    
    for entity in relevant_entities:
        # 出現回数
        count = base_text.lower().count(entity.lower())
        
        # エンティティの重要度を計算
        # 1. 長さで重み付け (長いフレーズはより具体的)
        length_weight = min(1.0, len(entity.split()) / 3.0)
        
        # 2. キャンディデートラベルとの関連性
        relevance_weight = 0.0
        for label in candidate_labels:
            if label.lower() == entity.lower():
                relevance_weight = 1.0
                break
            elif label.lower() in entity.lower() or entity.lower() in label.lower():
                relevance_weight = max(relevance_weight, 0.8)
        
        # 3. 出現位置 (文章の先頭に近い方が重要な可能性)
        position = base_text.lower().find(entity.lower())
        position_weight = 1.0 - (position / len(base_text)) if position >= 0 else 0.5
        
        # 複合スコア計算
        weight = 0.4 * length_weight + 0.4 * relevance_weight + 0.2 * position_weight
        counts[entity] = count * (0.5 + weight)
    
    # 正規化して確率に変換
    total = sum(counts.values()) if sum(counts.values()) > 0 else 1.0
    probs = {entity: count / total for entity, count in counts.items()}
    
    # キャンディデートラベルとのテキスト類似性も考慮
    if clip_extractor and relevant_entities:
        try:
            # エンティティと候補ラベルのテキスト埋め込みを取得
            entities_emb = clip_extractor.get_text_embedding(relevant_entities)
            labels_emb = clip_extractor.get_text_embedding(candidate_labels)
            
            # コサイン類似度の計算
            entities_emb_norm = entities_emb / np.linalg.norm(entities_emb, axis=1, keepdims=True)
            labels_emb_norm = labels_emb / np.linalg.norm(labels_emb, axis=1, keepdims=True)
            
            # 各エンティティについて、最も類似するラベルとの類似度を計算
            for i, entity in enumerate(relevant_entities):
                max_sim = 0.0
                for j in range(labels_emb_norm.shape[0]):
                    sim = (entities_emb_norm[i] * labels_emb_norm[j]).sum()
                    max_sim = max(max_sim, sim)
                
                # 確率に類似度の要素を加味
                if entity in probs:
                    probs[entity] = (probs[entity] + max_sim) / 2.0
        except Exception as e:
            print(f"Error during text embedding similarity calculation: {e}")
    
    # 確率順にソート
    sorted_probs = {k: v for k, v in sorted(probs.items(), key=lambda x: x[1], reverse=True)}
    
    # 上位N件に絞る
    top_n = 10
    if len(sorted_probs) > top_n:
        sorted_probs = {k: sorted_probs[k] for k in list(sorted_probs.keys())[:top_n]}
    
    return sorted_probs


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
        inputs = inputs.to(device) #("cuda")

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
    # plt.show()
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
    iso = IsolationForest(contamination=contamination, random_state=42)
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

    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    baseline_tsne = tsne.fit_transform(features)
    
    # 可視化（変更点）
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

    return clip_extractor, candidate_labels, features, baseline_tsne, class_names, iso_model


###############################################################################
# detect_new_images
###############################################################################
def detect_new_images(
    new_images_folder, #="data_nuscenes/nuscenes/samples/CAM_FRONT",
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
    out_dir = f"{out_dir}_{today}_lim{lim1}-{lim2}_prob"
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
    tsne_outliers = (avg_distances > threshold_tsne).astype(int)  # 1: outlier, 0: inlier

    if_labels = predict_outliers(features_new, iso_model) # 1: inlier, -1: outlier
    if_outliers = ((if_labels==-1).astype(int))
    combined_outlier_code = tsne_outliers*2 + if_outliers

    plt.figure(figsize=(18,12))

    # ベースラインデータの色分け（上位20クラス）
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
    plt.savefig("combined_tsne_class_if.png", bbox_inches='tight')
    plt.close()

    # 8) 各結果毎に画像を保存するフォルダ
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

    # Pre-create folders for all outlier codes
    for code, folder in out_folders.items():
        os.makedirs(os.path.join(out_dir, folder), exist_ok=True)

    detected_images_dir = "detected_images"
    outliner_dir = "outliner"
    os.makedirs(detected_images_dir, exist_ok=True)
    os.makedirs(outliner_dir, exist_ok=True)
    
    for out_code, (path, cls_name, conf, bbox, original_path) in zip(combined_outlier_code, all_cropped_info):
        folder_name = out_folders.get(out_code, "others")
        save_folder = os.path.join(out_dir, folder_name)
        
        # Copy original image to the folder corresponding to out_code
        shutil.copy(original_path, save_folder)
        
        # For outliers, also copy to detected_images
        if out_code != 0:
            save_img = os.path.join(detected_images_dir, os.path.basename(original_path))
            shutil.copy(original_path, save_img)

        base_name = os.path.basename(path)
        dst_path = os.path.join(save_folder, base_name)
        # 画像をコピー (Windows の場合はshutil.copyが簡単)
        shutil.copy(path, dst_path)

        # 画像の説明文を生成
        desc = generate_description(path, model, processor)

        # 説明文生成に失敗した場合はスキップする
        if desc is None or "I am unable to" in desc or "I'm unable to analyze" in desc or "unable to process" in desc:
            probs_dict = {}  # 空の辞書を設定
        else:
            # 新しい要素分解ベースの確率計算メソッドを使用
            probs_dict = parse_text_with_probability(desc, candidate_labels, clip_extractor)

        # テキストファイルに書き出し
        txt_path = os.path.join(save_folder, f"{os.path.splitext(base_name)[0]}_desc.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"Outlier code: {out_code} (0:inlier_inlier, 1:IFonly, 2:tSNEonly, 3:both)\n")
            f.write(f"Cropped Image Path: {path}\n")
            f.write(f"Original Image Path: {original_path}\n")
            f.write(f"Class={cls_name}, conf={conf:.2f}, bbox={bbox}\n\n")
            f.write("Generated caption:\n")
            f.write(desc + "\n\n")
            if probs_dict:  # 辞書が空でない場合のみ確率情報を書き込む
                f.write("Extracted elements and their probabilities:\n")
                for k_, v_ in probs_dict.items():
                    f.write(f"  {k_}: {v_:.4f}\n")
            else:
                f.write("No elements could be extracted from the caption.\n")

        # 確率の棒グラフを保存
        if len(probs_dict) > 0:
            plt.figure(figsize=(10, 6))
            plt.bar(probs_dict.keys(), probs_dict.values(), color='skyblue')
            plt.xticks(rotation=45, ha='right')
            plt.title("Element Probabilities")
            plt.tight_layout()
            prob_png = os.path.join(save_folder, f"{os.path.splitext(base_name)[0]}_probs.png")
            plt.savefig(prob_png)
            plt.close()
    if save_outliers:
        outlier_info = []
        for out_code, (path, cls_name, conf, bbox, original_path) in zip(combined_outlier_code, all_cropped_info):
            if out_code != 0:  # アウトライア（inlier_inlier以外）の場合
                # NuScenesのサンプルデータトークンを取得（ファイル名からトークンを抽出）
                sample_token = os.path.splitext(os.path.basename(original_path))[0]
                outlier_info.append({
                    'sample_token': sample_token,
                    'outlier_code': int(out_code),
                    'original_path': original_path,
                    'class_name': cls_name,
                    'confidence': float(conf)
                })
        # 結果をJSONファイルとして保存
        
        outlier_path = os.path.join("outlier_detection_results.json")
        with open(outlier_path, 'w') as f:
            json.dump(outlier_info, f, indent=2)
        print(f"Outlier information saved to {outlier_path}")


###############################################################################
# Main
###############################################################################
def main():
    # 1) train baseline model
    baseline_folder = "data_nuscenes/samples/CAM_FRONT" #"projects/detect_rare_images/data/test_data" # #/nuscenes
    lim1 = 0.002 #0.5 #0.2 #0.8
    lim2 = 0.003 #1.0 #0.3 #1.0
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
        threshold_percentile=80, #60
        lim1=lim1,
        lim2=lim2
    )

    print(f"Using device: {get_device()}")
    print("\nAll Done.")


if __name__ == "__main__":
    main()
