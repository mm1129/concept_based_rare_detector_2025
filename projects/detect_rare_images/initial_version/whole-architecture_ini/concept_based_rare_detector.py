#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import base64
import numpy as np
import torch
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import json
import datetime
import shutil
import random
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
import nltk
from ultralytics import YOLO
from sklearn.manifold import TSNE  # t-SNE用に追加
from tqdm import tqdm  # 進捗表示用

# NLTK リソースのダウンロード (初回のみ)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

###############################################################################
# デバイス設定
###############################################################################
def get_device():
    if torch.cuda.is_available():
        return "cuda:0"
    else:
        return "cpu"

device = get_device()

###############################################################################
# 画像エンコード用の関数
###############################################################################
def encode_image(image_path):
    if not os.path.isfile(image_path):
        return None
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode("utf-8")

###############################################################################
# Qwenモデルによる画像キャプション生成関数
###############################################################################
def generate_description_qwen(image_path, model, processor, device):
    """
    Qwenモデルを使用して画像キャプションを生成
    """
    try:
        pil_image = Image.open(image_path).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": "Describe this image focusing on any unusual or potentially rare objects or scenarios relevant to autonomous driving."
                    }
                ]
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[text],
            images=[pil_image],
            padding=True,
            return_tensors="pt",
        ).to(device)
        max_tokens = 150
        generated_ids = model.generate(**inputs, max_new_tokens=max_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return response
    except Exception as e:
        print(f"Error generating caption with Qwen for {image_path}: {e}")
        return None

###############################################################################
# BLIPモデルによる画像キャプション生成クラス
###############################################################################
class BLIPFeatureExtractor:
    """
    BLIPモデルを使用して画像キャプションを生成
    """
    def __init__(self, model_name="Salesforce/blip-image-captioning-large", device=None):
        if device is None:
            device = get_device()
        self.device = device

        print(f"Loading BLIP model: {model_name} on {device}")
        self.model = BlipForConditionalGeneration.from_pretrained(model_name)
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model.eval()
        self.model.to(self.device)

    def get_text_embedding(self, text_list):
        """
        テキストの特徴ベクトルを取得
        """
        all_embeddings = []
        
        for text in text_list:
            inputs = self.processor(text=text, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.text_model(**inputs)
                hidden_states = outputs.last_hidden_state
                attention_mask = inputs.get("attention_mask", None)
                
                if attention_mask is not None:
                    expanded_mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                    sum_hidden = torch.sum(hidden_states * expanded_mask, dim=1)
                    sum_mask = torch.clamp(torch.sum(attention_mask, dim=1).unsqueeze(-1), min=1e-9)
                    text_features = sum_hidden / sum_mask
                else:
                    text_features = torch.mean(hidden_states, dim=1)
                    
            all_embeddings.append(text_features.cpu().numpy().squeeze(0))
            
        return np.array(all_embeddings)
    
    def get_image_embedding(self, image_path):
        """
        BLIPモデルを使用して画像特徴ベクトルを取得
        """
        try:
            pil_image = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=pil_image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.vision_model(**inputs)
                image_features = outputs.pooler_output
                
            return image_features.cpu().numpy().squeeze(0)
        except Exception as e:
            print(f"Error generating image embedding with BLIP: {e}")
            return None

    def generate_caption(self, image_path):
        """
        BLIPモデルを使用して画像キャプションを生成
        """
        try:
            pil_image = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=pil_image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=50)
                caption = self.processor.decode(generated_ids[0], skip_special_tokens=True)
            return caption.strip()
        except Exception as e:
            print(f"Error generating caption with BLIP for {image_path}: {e}")
            return None

###############################################################################
# CLIPモデルのためのクラス
###############################################################################
class CLIPFeatureExtractor:
    """
    CLIPモデルを使用してテキスト特徴量を抽出するクラス
    """
    def __init__(self, model_name="openai/clip-vit-base-patch32", device=None):
        if device is None:
            device = get_device()
        self.device = device

        print(f"Loading CLIP model: {model_name} on {device}")
        self.model = CLIPModel.from_pretrained(model_name, device_map=device) #"auto")
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()
        self.model.to(self.device)
        self.text_embedding_cache = {}
        self.image_embedding_cache = {}

    def get_text_embedding(self, text_list):
        """
        テキストリストの埋め込みベクトルを取得（バッチ処理とキャッシュ機能付き）
        """
        uncached_texts = []
        uncached_indices = []
        embeddings = [None] * len(text_list)
        
        for i, text in enumerate(text_list):
            if text in self.text_embedding_cache:
                embeddings[i] = self.text_embedding_cache[text]
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        if uncached_texts:
            batch_size = 64
            for i in range(0, len(uncached_texts), batch_size):
                batch_texts = uncached_texts[i:i + batch_size]
                batch_indices = uncached_indices[i:i + batch_size]
                
                inputs = self.processor(
                    text=batch_texts, 
                    return_tensors="pt",
                    padding=True, 
                    truncation=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    batch_features = self.model.get_text_features(**inputs)
                    batch_features = batch_features.cpu().numpy()
                
                for j, (text_idx, text) in enumerate(zip(batch_indices, batch_texts)):
                    normalized_embedding = batch_features[j] / np.linalg.norm(batch_features[j])
                    self.text_embedding_cache[text] = normalized_embedding
                    embeddings[text_idx] = normalized_embedding
        
        return np.array(embeddings)
    
    def get_image_embedding(self, image_path):
        """
        画像の埋め込みベクトルを取得（キャッシュ機能付き）
        """
        if image_path in self.image_embedding_cache:
            return self.image_embedding_cache[image_path]
            
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                
            embedding = image_features.cpu().numpy().squeeze(0)
            normalized_embedding = embedding / np.linalg.norm(embedding)
            
            self.image_embedding_cache[image_path] = normalized_embedding
            return normalized_embedding
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None

###############################################################################
# YOLOモデルによる物体検出クラス
###############################################################################
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
# コンセプト間類似度を計算する関数
###############################################################################
def compute_concept_similarity(caption, concept_list, clip_extractor=None):
    """
    キャプションとコンセプトリスト間の類似度を計算
    
    Args:
        caption: 画像から生成されたキャプション
        concept_list: コンセプトのリスト
        clip_extractor: CLIPFeatureExtractor インスタンス
    
    Returns:
        類似度が高い順にソートされたコンセプトと類似度の辞書
    """
    if not caption:
        return {}
    
    if clip_extractor is not None:
        caption_emb = clip_extractor.get_text_embedding([caption])
        concept_embs = clip_extractor.get_text_embedding(concept_list)
        
        similarities = {}
        for i, concept in enumerate(concept_list):
            sim = np.dot(caption_emb[0], concept_embs[i])
            similarities[concept] = float(sim)
    else:
        return {}
    
    similarities = {k: v for k, v in sorted(similarities.items(), key=lambda x: x[1], reverse=True)}
    return similarities

###############################################################################
# レア事象を検出する関数
###############################################################################
def detect_rare_objects(
    images_folder,
    concept_list,
    common_classes=["car", "truck", "bus", "pedestrian"],
    similarity_threshold=0.25,
    out_dir=None,
    caption_model_type='qwen',
    use_clip_for_embedding=True,
    n_top_concepts_for_rarity=3,
    process_percentage=100,
    batch_size=16,
    potential_classes=None,
    only_potential_classes=True
):
    # potential_classesがNoneの場合のデフォルト値を設定
    if potential_classes is None:
        potential_classes = [
            "construction_vehicle", "motorcycle", "bicycle", "truck", "bus", "trailer",
            "emergency_vehicle", "police_car", "ambulance", "fire_truck",
            "tractor", "snowplow", "garbage_truck", "military_vehicle",
        ]
    
    # 小文字に変換して比較を容易にする
    potential_classes_lower = [pc.lower() for pc in potential_classes]
    
    now = datetime.datetime.now()
    timestamp_str = now.strftime("%m%d_%H%M")
    if out_dir is None:
        out_dir = f"rare_objects_{timestamp_str}"
    
    os.makedirs(out_dir, exist_ok=True)
    rare_dir = os.path.join(out_dir, "rare")
    common_dir = os.path.join(out_dir, "common")
    os.makedirs(rare_dir, exist_ok=True)
    os.makedirs(common_dir, exist_ok=True)
    
    # 特定のコンセプト用のフォルダを作成
    cv_folder = os.path.join(out_dir, "construction_vehicle")
    motor_folder = os.path.join(out_dir, "motorcycle")
    os.makedirs(cv_folder, exist_ok=True)
    os.makedirs(motor_folder, exist_ok=True)
    
    # 特定のコンセプト用の情報リスト
    cv_info = []
    motor_info = []
    
    detector = DetectorYOLO(model_path="yolov8l.pt")
    
    clip_extractor = None
    if use_clip_for_embedding:
        clip_extractor = CLIPFeatureExtractor(model_name="openai/clip-vit-base-patch32")
        _ = clip_extractor.get_text_embedding(concept_list)
    
    if caption_model_type == 'qwen':
        print("Loading Qwen model...")
        model_size = "Qwen/Qwen2-VL-2B"
        caption_processor = AutoProcessor.from_pretrained(model_size)
        caption_model = Qwen2VLForConditionalGeneration.from_pretrained(model_size)
        caption_model.eval()
        caption_model.to(device)
    elif caption_model_type == 'blip':
        caption_model = BLIPFeatureExtractor(model_name="Salesforce/blip-image-captioning-large", device=device)
    
    all_image_files = [f for f in os.listdir(images_folder) 
                       if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif"))]
    
    num_files_to_process = int(len(all_image_files) * (process_percentage / 100.0))

    if process_percentage < 100:
        if num_files_to_process < len(all_image_files):
            image_files = random.sample(all_image_files, num_files_to_process)
            print(f"{process_percentage}% ({num_files_to_process}/{len(all_image_files)}) の画像をランダムに選択して処理します。")
        else:
            image_files = all_image_files
            print(f"指定された割合 ({process_percentage}%) が100%以上のため、全ての画像 ({len(all_image_files)}) を処理します。")
    else:
        image_files = all_image_files
        print(f"全ての画像 ({len(all_image_files)}) を処理します。")

    rare_objects = []
    common_objects = []

    total_files = len(image_files)
    print(f"処理対象の画像数: {total_files}")
    
    # 処理済み画像を追跡するためのセット
    processed_images = set()
    
    # 処理対象のクラスを表示
    if only_potential_classes:
        print(f"処理対象のクラス: {potential_classes}")
    else:
        print("全てのクラスを処理します（potential_classesを優先）")
    
    for idx, file_name in enumerate(tqdm(image_files, desc="画像処理中")):
        try:
            image_path = os.path.join(images_folder, file_name)
            
            # 既に処理済みの画像はスキップ
            if image_path in processed_images:
                continue
                
            cropped_info_list = detector.detect_and_crop(
                image_path, out_dir=os.path.join(out_dir, "crops"), conf_thres=0.3
            )
            
            if not cropped_info_list:
                continue
            
            # potential_classesに含まれるクラスが検出されたかどうかをチェック
            potential_objects = []
            for info in cropped_info_list:
                _, yolo_class, _, _, _ = info
                if yolo_class.lower() in potential_classes_lower:
                    potential_objects.append(info)
            
            # only_potential_classesがTrueの場合、potential_classesに含まれるオブジェクトがない場合はスキップ
            if only_potential_classes and not potential_objects:
                continue
            
            # この画像を処理済みとしてマーク
            processed_images.add(image_path)
            
            # 処理対象のオブジェクトリスト
            objects_to_process = potential_objects if only_potential_classes else cropped_info_list
            
            for crop_path, yolo_class, conf, bbox, orig_img_path in objects_to_process:
                # only_potential_classesがFalseの場合でも、potential_classesに含まれるクラスを優先的に処理
                is_potential = yolo_class.lower() in potential_classes_lower
                
                if caption_model_type == 'qwen':
                    caption = generate_description_qwen(crop_path, caption_model, caption_processor, device)
                elif caption_model_type == 'blip':
                    caption = caption_model.generate_caption(crop_path)
                
                if not caption:
                    continue
                if 'blurry' in caption.lower():
                    continue
                
                caption_lower = caption.lower()
                if "unable to" in caption_lower or "i'm unable to" in caption_lower or "cannot process" in caption_lower:
                    continue
                    
                similarities = compute_concept_similarity(caption, concept_list, clip_extractor)
                
                if not similarities:
                    continue
                
                top_concept = list(similarities.keys())[0]
                top_similarity = similarities[top_concept]
                
                top_n_concepts = list(similarities.keys())[:n_top_concepts_for_rarity]
                is_top_n_concepts_rare = all(concept not in common_classes for concept in top_n_concepts)
                
                # potential_classesに含まれるクラスの場合、より寛容な条件でレア判定
                if is_potential:
                    is_rare = top_similarity > (similarity_threshold * 0.8)  # 閾値を少し下げる
                else:
                    is_rare = is_top_n_concepts_rare and top_similarity > similarity_threshold
                
                result = {
                    "image_path": orig_img_path,
                    "sample_token": file_name,
                    "cropped_path": crop_path,
                    "yolo_class": yolo_class,
                    "confidence": float(conf),
                    "caption": caption,
                    "top_concept": top_concept,
                    "top_similarity": float(top_similarity),
                    "bbox": [int(x) for x in bbox],
                    "is_rare": is_rare,
                    "is_potential_class": is_potential
                }
                
                if result["is_rare"]:
                    rare_objects.append(result)
                    shutil.copy(crop_path, os.path.join(rare_dir, os.path.basename(crop_path)))
                    shutil.copy(orig_img_path, os.path.join(rare_dir, os.path.basename(orig_img_path)))
                    
                    txt_path = os.path.join(rare_dir, f"{os.path.splitext(os.path.basename(crop_path))[0]}_info.txt")
                    with open(txt_path, "w") as f:
                        f.write(f"Original Image: {orig_img_path}\n")
                        f.write(f"YOLO Class: {yolo_class}, Confidence: {conf:.4f}\n")
                        f.write(f"Caption: {caption}\n")
                        f.write(f"Top Concept: {top_concept}, Similarity: {top_similarity:.4f}\n")
                        f.write(f"Top 5 Concepts:\n")
                        for i, (concept, sim) in enumerate(list(similarities.items())[:5]):
                            f.write(f"  {concept}: {sim:.4f}\n")
                    
                    # 特定のコンセプトに対する特別な処理
                    sample_token = os.path.splitext(os.path.basename(orig_img_path))[0]
                    
                    if top_concept == 'construction_vehicle':
                        if os.path.exists(orig_img_path):
                            shutil.copy(orig_img_path, os.path.join(cv_folder, os.path.basename(orig_img_path)))
                        cv_info.append({
                            'sample_token': sample_token,
                            'yolo_class': yolo_class,
                            'is_potential_class': is_potential,
                            'original_path': orig_img_path,
                            'confidence': float(conf),
                            'top_concept': top_concept,
                            'top_similarity': float(top_similarity)
                        })
                    
                    if top_concept == 'motorcycle':
                        if os.path.exists(orig_img_path):
                            shutil.copy(orig_img_path, os.path.join(motor_folder, os.path.basename(orig_img_path)))
                        motor_info.append({
                            'sample_token': sample_token,
                            'yolo_class': yolo_class,
                            'is_potential_class': is_potential,
                            'original_path': orig_img_path,
                            'confidence': float(conf),
                            'top_concept': top_concept,
                            'top_similarity': float(top_similarity)
                        })
                else:
                    common_objects.append(result)
                    shutil.copy(crop_path, os.path.join(common_dir, os.path.basename(crop_path)))
                
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            continue
    
    print(f"検出されたレア物体数: {len(rare_objects)}")
    print(f"一般的な物体数: {len(common_objects)}")
    
    with open(os.path.join(out_dir, f"rare_objects_{timestamp_str}.json"), "w") as f:
        json.dump(rare_objects, f, indent=2)
    
    visualize_results(rare_objects, out_dir, common_objects)
    
    # 特定のコンセプト情報をJSONファイルに保存
    if cv_info:
        with open(os.path.join(out_dir, "construction_vehicle_results.json"), "w") as f:
            json.dump(cv_info, f, indent=2)
        print(f"建設車両情報を保存しました: {len(cv_info)}件")
    
    if motor_info:
        with open(os.path.join(out_dir, "motorcycle_results.json"), "w") as f:
            json.dump(motor_info, f, indent=2)
        print(f"バイク情報を保存しました: {len(motor_info)}件")
    
    return rare_objects, common_objects

###############################################################################
# 結果のビジュアライゼーション
###############################################################################
def visualize_results(rare_objects, out_dir, common_objects=None):
    """
    検出されたレア物体の概要をビジュアライズ
    """
    if not rare_objects:
        print("No rare objects to visualize")
        return
    
    concepts = {}
    for obj in rare_objects:
        concept = obj["top_concept"]
        if concept in concepts:
            concepts[concept] += 1
        else:
            concepts[concept] = 1
    
    sorted_concepts = {k: v for k, v in sorted(concepts.items(), key=lambda item: item[1], reverse=True)}
    
    plt.figure(figsize=(12, 8))
    concepts_to_plot = dict(list(sorted_concepts.items())[:20])
    plt.bar(concepts_to_plot.keys(), concepts_to_plot.values(), color='skyblue')
    plt.xticks(rotation=45, ha='right')
    plt.title("Top 20 Rare Concepts Detected")
    plt.xlabel("Concept")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "rare_concepts_distribution.png"))
    plt.close()
    
    plt.figure(figsize=(10, 6))
    similarities = [obj["top_similarity"] for obj in rare_objects]
    plt.hist(similarities, bins=20, color='skyblue', alpha=0.7)
    plt.title("Distribution of Top Concept Similarities")
    plt.xlabel("Similarity Score")
    plt.ylabel("Count")
    plt.axvline(0.25, color='red', linestyle='--', label="Threshold (0.25)")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "similarity_distribution.png"))
    plt.close()
    
    if common_objects:
        visualize_embeddings_tsne(rare_objects, common_objects, out_dir)

def visualize_embeddings_tsne(rare_objects, common_objects, out_dir):
    """
    t-SNEを使用してレア物体と一般的物体の埋め込みを視覚化（CLIPモデル使用）
    """
    clip_extractor = CLIPFeatureExtractor(model_name="openai/clip-vit-large-patch14")
    
    rare_paths = [obj["cropped_path"] for obj in rare_objects if os.path.exists(obj["cropped_path"])]
    common_paths = [obj["cropped_path"] for obj in common_objects if os.path.exists(obj["cropped_path"])]
    
    max_samples = 1000
    if len(common_paths) > max_samples:
        common_paths = random.sample(common_paths, max_samples)
    
    print(f"t-SNE視覚化: レア物体 {len(rare_paths)}個、一般的物体 {len(common_paths)}個")
    
    def batch_get_image_embeddings(image_paths, batch_size=32):
        embeddings = []
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            print(f"バッチ処理中: {i+1}～{min(i+batch_size, len(image_paths))}/{len(image_paths)}")
            
            batch_embeddings = []
            for path in tqdm(batch_paths, desc="バッチ内処理"):
                emb = clip_extractor.get_image_embedding(path)
                if emb is not None:
                    batch_embeddings.append(emb)
            
            embeddings.extend(batch_embeddings)
        return embeddings
    
    print("レア物体の埋め込み抽出中...")
    rare_embeddings = batch_get_image_embeddings(rare_paths)
    
    print("一般的物体の埋め込み抽出中...")
    common_embeddings = batch_get_image_embeddings(common_paths)
    
    if len(rare_embeddings) == 0 or len(common_embeddings) == 0:
        print("t-SNE視覚化のための十分な埋め込みがありません")
        return
    
    all_embeddings = np.vstack([rare_embeddings, common_embeddings])
    
    del rare_embeddings
    del common_embeddings
    
    labels = np.array([1] * len(rare_paths) + [0] * len(common_paths))
    
    print("t-SNE次元削減実行中...")
    tsne = TSNE(
        n_components=2, 
        random_state=42, 
        perplexity=min(30, len(all_embeddings) - 1),
        n_jobs=-1,
        verbose=1
    )
    embeddings_2d = tsne.fit_transform(all_embeddings)
    
    rare_count = len([p for p in rare_paths if os.path.exists(p)])
    rare_tsne = embeddings_2d[:rare_count]
    common_tsne = embeddings_2d[rare_count:]
    
    plt.figure(figsize=(12, 10))
    plt.scatter(common_tsne[:, 0], common_tsne[:, 1], c='blue', alpha=0.5, label='一般的物体')
    plt.scatter(rare_tsne[:, 0], rare_tsne[:, 1], c='red', alpha=0.7, label='レア物体')
    plt.title('一般的物体とレア物体のt-SNE視覚化（CLIP画像埋め込み）')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "tsne_visualization.png"), dpi=300)
    plt.close()
    
    print("t-SNE視覚化を完了しました")

###############################################################################
# メイン関数
###############################################################################
def main():
    images_folder = "projects/detect_rare_images/data/test_data" #"./data_nuscenes/samples/CAM_FRONT"
    
    # potential_classesリストを追加
    potential_classes = [
        "construction_vehicle", "motorcycle", "bicycle", "truck", "bus", "trailer",
        "emergency_vehicle", "police_car", "ambulance", "fire_truck",
        "tractor", "snowplow", "garbage_truck", "military_vehicle",
    ]
    
    concept_list = [
        "car", "truck", "construction_vehicle", "bus", "trailer", "barrier", "motorcycle",
        "bicycle", "pedestrian", "traffic_cone",
        
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
        "luggage", "suitcase", "backpack", "furniture", "appliance",
        "trash_bin", "trash_bag", "construction_debris",
        
        "crane", "bulldozer", "excavator", "forklift", "cement_mixer",
        "road_roller", "backhoe", "cherry_picker", "construction_worker",
        "maintenance_worker", "jackhammer", "generator", "construction_materials",
        "portable_toilet",
        
        "pothole", "manhole", "speed_bump", "road_sign_fallen", "traffic_light_fallen",
        "road_blockade", "water_puddle", "barricade", "detour_sign",
        "construction_sign", "temporary_sign",
        
        "snow_pile", "flood_water", "fog_patch", "ice_patch", "fallen_power_line",
        
        "parade_performer", "film_crew", "street_performer", "protest_group",
        "unusual_cargo", "art_installation", "movie_prop", "sports_equipment",
        "hot_air_balloon", "drone", "bouncy_castle", "advertisement_mascot"
    ]
    
    common_classes = ["car", "truck", "bus", "pedestrian", "traffic light", "traffic sign"]
    
    rare_objects, common_objects = detect_rare_objects(
        images_folder=images_folder,
        concept_list=concept_list,
        common_classes=common_classes,
        similarity_threshold=0.25,
        caption_model_type='qwen',
        use_clip_for_embedding=True,
        n_top_concepts_for_rarity=3,
        process_percentage=10,
        potential_classes=potential_classes,
        only_potential_classes=True
    )
    
    print(f"\n検出されたレア物体数: {len(rare_objects)}")
    print(f"一般的な物体数: {len(common_objects)}")
    print("\n完了")

if __name__ == "__main__":
    main()
