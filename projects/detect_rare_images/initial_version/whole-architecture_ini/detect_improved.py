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
from tqdm import tqdm
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics.pairwise import cosine_similarity

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

# GPUメモリ管理の最適化設定
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True  # TensorFloat-32を有効化
    torch.backends.cudnn.allow_tf32 = True
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
    def __init__(self, model_path="yolov8x.pt", device=None):
        if device is None:
            device = get_device()
        self.device = device
        
        # モデルをロード
        self.model = YOLO(model_path)
        
        # NMS設定を調整（可能な場合）
        # if hasattr(self.model.args, 'iou') and hasattr(self.model.args, 'conf'):
        #     self.model.args.iou = 0.5  # IoUしきい値
        #     self.model.args.conf = 0.25  # 信頼度しきい値
            
        self.model.to(self.device)

    def detect_and_crop(self, image_path, out_dir="cropped_objects", conf_thres=0.25):
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
            
            # バッチ処理モードを使用（最新のUltralytics APIに合わせた実装）
            results = self.model(
                img,
                conf=conf_thres,
                iou=0.5,
                max_det=100,
                device=self.device,
                verbose=False
            )
            
            # Soft-NMSを手動で適用（必要な場合）
            # 注: 最新のYOLOv8ではNMSは内部で最適化されているため、
            # 通常はデフォルト設定で十分です
            
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
    def __init__(self, model_name=None, device=None):
        if device is None:
            device = get_device()
        self.device = device
        
        # 環境変数でモデルを選択可能に
        EMBEDDER = os.getenv("EMBEDDER", "clip_l14")
        if EMBEDDER == "clip_l14":
            model_name = "openai/clip-vit-large-patch14"
        else:
            model_name = model_name or "openai/clip-vit-base-patch32"
            
        print(f"CLIPモデルを初期化: {model_name}")
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
            # L2正規化を適用
            image_features = image_features / torch.norm(image_features, dim=1, keepdim=True)
        return image_features.cpu().numpy().squeeze(0)

    def get_image_embeddings_batch(self, image_paths, batch_size=16):  # バッチサイズを16に変更
        """複数の画像のCLIP特徴をバッチ処理で一度に取得"""
        all_embeddings = []
        total = len(image_paths)
        
        # バッチ処理
        for i in tqdm(range(0, total, batch_size), desc="特徴抽出"):
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
                # L2正規化を適用
                batch_features = batch_features / torch.norm(batch_features, dim=1, keepdim=True)
            
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
def generate_descriptions_batch(image_paths, model, processor, batch_size=8, detected_classes=None):
    """バッチ処理で画像キャプションを生成（最適化版）"""
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
                # YOLOで検出されたクラス名を取得
                detected_class = detected_classes.get(path, "") if detected_classes else ""
                
                # ターゲットクラスに近いものは詳細なキャプションを生成
                target_lst = ["construction", "bicycle", "motorcycle", "trailer", "truck", "fence", "barrier", "traffic_cone", "traffic_light", "traffic_sign"]
                is_target_class = any(target.lower() in detected_class.lower() for target in target_lst) #"construction_vehicle", "bicycle", "motorcycle", "trailer", "truck"])
                
                # ターゲットクラスでない場合は簡易キャプションのみ生成（処理時間短縮）
                if not is_target_class:
                    simple_caption = detected_class
                    results.append(simple_caption)
                    continue
                
                # ターゲットクラスの場合は詳細なキャプション生成
                base64_image = encode_image(path)
                if not base64_image:
                    results.append(None)
                    continue
                
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
                                        "Analyze the following image and describe notable objects/scenarios. "
                                        "Pay special attention to one of these nuScenes classes: car, truck, trailer, bus, construction_vehicle, bicycle, motorcycle, pedestrian, traffic_cone, barrier."
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
# 6. Outlier detection (LOF)
###############################################################################
def predict_outliers_lof(features, n_neighbors=50, contamination=0.1):
    """
    Local Outlier Factorを使用してアウトライアを検出
    """
    if len(features) == 0:
        return []
    
    # LOFモデルを初期化
    lof = LocalOutlierFactor(
        n_neighbors=min(n_neighbors, len(features) - 1),
        contamination=contamination,
        novelty=False
    )
    
    # 予測実行 (inlier=1, outlier=-1)
    labels = lof.fit_predict(features)
    
    # アウトライア判定結果を返す (1: outlier, 0: inlier)
    return (labels == -1).astype(int)

# クラス認識型LOFを実装
def detect_class_aware_outliers(features, class_names, contamination=0.1, min_samples=10):
    """
    クラスごとにアウトライア検出を行う関数
    """
    # クラスごとのインデックスを取得
    class_indices = {}
    for i, cls in enumerate(class_names):
        if cls not in class_indices:
            class_indices[cls] = []
        class_indices[cls].append(i)
    
    # 結果を格納する配列
    outlier_flags = np.zeros(len(features), dtype=int)
    class_outlier_info = {}
    
    # クラスごとにアウトライア検出
    for cls, indices in class_indices.items():
        if len(indices) < min_samples:
            continue
            
        # クラス内の特徴ベクトルを抽出
        cls_features = features[indices]
        
        # 最適なneighborsサイズを計算
        optimal_n_neighbors = min(50, max(5, len(indices) // 10))
        
        # LOFモデルでアウトライア検出
        lof = LocalOutlierFactor(
            n_neighbors=optimal_n_neighbors, 
            contamination=min(contamination, 0.5),
            novelty=False
        )
        cls_outlier_labels = lof.fit_predict(cls_features)
        
        # アウトライアのインデックスを取得
        cls_outlier_indices = [indices[i] for i, label in enumerate(cls_outlier_labels) if label == -1]
        
        # 結果を格納
        for idx in cls_outlier_indices:
            outlier_flags[idx] = 1
            
        # クラス情報を保存
        class_outlier_info[cls] = {
            "total_samples": len(indices),
            "outlier_count": len(cls_outlier_indices),
            "outlier_ratio": len(cls_outlier_indices) / len(indices),
            "outlier_indices": cls_outlier_indices
        }
    
    return outlier_flags, class_outlier_info



def normalize_class_name(class_name):
    """クラス名を正規化する補助関数"""
    if not class_name:
        return ""
    normalized = class_name.lower().strip()
    # スペルミスの修正
    normalized = normalized.replace('vechicle', 'vehicle')
    # 区切り文字の正規化
    normalized = normalized.replace('_', ' ')
    return normalized

def is_target_class_match(class_name, target, near_classes):
    """ターゲットクラスとのマッチングをチェックする補助関数"""
    if not class_name or not target:
        return False
        
    normalized_class = normalize_class_name(class_name)
    normalized_target = normalize_class_name(target)
    
    # 完全一致チェック
    if normalized_class == normalized_target:
        return True
    
    # 部分一致チェック
    if normalized_target in normalized_class:
        return True
    
    # 類似クラスチェック
    if near_classes and target in near_classes:
        return any(
            normalize_class_name(alt) in normalized_class
            for alt in near_classes[target]
        )
    
    return False

###############################################################################
# メイン関数: 単一フォルダでのアウトライア検出
###############################################################################
def detect_outliers_single_folder(images_folder, output_dir="outlier_detection_results", 
                                 qwen_model_size="2B", contamination=0.1, 
                                 target_classes=None, common_classes=None,
                                 save_crops=True, save_descriptions=True, 
                                 save_probability_plots=True, cleanup_temp_files=False,
                                 max_images=None, seed=42):
    """
    単一フォルダ内の画像からアウトライアを検出する関数
    """
    # 出力ディレクトリの設定
    timestamp = datetime.datetime.now().strftime('%m%d_%H-%M-%S_JST')
    main_out_dir = os.path.join(output_dir, f"detection_{timestamp}")
    os.makedirs(main_out_dir, exist_ok=True)
    
    # サブディレクトリの作成
    cropped_dir = os.path.join(main_out_dir, "cropped_objects")
    outlier_dir = os.path.join(main_out_dir, "outliers")
    inlier_dir = os.path.join(main_out_dir, "inliers")
    final_outlier_dir = os.path.join(main_out_dir, "final_outliers")
    
    # ディレクトリ作成
    os.makedirs(cropped_dir, exist_ok=True)
    os.makedirs(outlier_dir, exist_ok=True)
    os.makedirs(inlier_dir, exist_ok=True)
    os.makedirs(final_outlier_dir, exist_ok=True)
    
    # ターゲットクラス用のディレクトリ作成
    targets_base_dir = os.path.join(main_out_dir, "targets")
    os.makedirs(targets_base_dir, exist_ok=True)
    
    target_dirs = {}
    for target in target_classes:
        target_dir = os.path.join(targets_base_dir, target)
        os.makedirs(target_dir, exist_ok=True)
        target_dirs[target] = target_dir
    
    # シグナルハンドラを設定
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 日本語フォントの設定
    setup_japanese_font()
    
    # シード値を設定
    set_seed(seed)
    
    # 開始時間を記録
    start_time_overall = time.time()
    
    # 設定情報を保存
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
        "device": get_device()
    }
    
    config_path = os.path.join(main_out_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"結果は {main_out_dir} に保存されます")
    print(f"設定情報: {config}")
    
    # 画像ファイルのリストを取得
    all_files = [f for f in os.listdir(images_folder) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif"))]
    random.shuffle(all_files)
    
    # 最大画像数を制限
    if max_images is not None and max_images < len(all_files):
        all_files = all_files[:max_images]
        print(f"処理対象: {len(all_files)}枚の画像（最大数制限）")
    else:
        print(f"処理対象: {len(all_files)}枚の画像")
    
    # YOLOモデルを初期化
    detector = DetectorYOLO(model_path="yolov8x.pt")
    
    # 1. オブジェクト検出と切り抜き
    print("オブジェクト検出と切り抜きを実行中...")
    all_cropped_info = []
    for idx, file_name in enumerate(all_files):
        try:
            # 進捗表示
            if idx % 100 == 0 or idx == len(all_files) - 1:
                elapsed = time.time() - start_time_overall
                progress = (idx + 1) / len(all_files) * 100
                eta = (elapsed / (idx + 1)) * (len(all_files) - (idx + 1)) if idx > 0 else 0
                print(f"オブジェクト検出進捗: {progress:.1f}% ({idx+1}/{len(all_files)}) | 経過: {elapsed:.1f}秒 | 残り: {eta:.1f}秒")
            
            image_path = os.path.join(images_folder, file_name)
            cropped_info = detector.detect_and_crop(
                image_path, out_dir=cropped_dir, conf_thres=0.25
            )
            all_cropped_info.extend(cropped_info)
        except Exception as e:
            print(f"画像処理エラー {file_name}: {e}")
            continue
    
    print(f"検出されたオブジェクト数: {len(all_cropped_info)}")
    
    # 検出されたオブジェクトがない場合は終了
    if len(all_cropped_info) == 0:
        print("検出されたオブジェクトがありません。処理を終了します。")
        return
    
    # 2. CLIPモデルを初期化
    clip_extractor = CLIPFeatureExtractor(model_name=None)
    
    # 3. 特徴抽出
    print("特徴抽出を実行中...")
    obj_paths = [info[0] for info in all_cropped_info]
    features = clip_extractor.get_image_embeddings_batch(obj_paths, batch_size=32)
    
    # 特徴抽出に失敗したオブジェクトを除外
    valid_indices = [i for i, feat in enumerate(features) if feat is not None]
    valid_features = features[valid_indices]
    valid_cropped_info = [all_cropped_info[i] for i in valid_indices]
    
    print(f"有効な特徴ベクトル数: {len(valid_features)}/{len(all_cropped_info)}")
    
    # 4. t-SNEによる可視化
    print("t-SNEによる可視化を実行中...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(valid_features)-1))
    tsne_result = tsne.fit_transform(valid_features)
    
    # クラス別の可視化
    class_names = [info[1] for info in valid_cropped_info]
    class_counts = Counter(class_names)
    top_classes = [cls for cls, count in class_counts.most_common(20)]
    
    plt.figure(figsize=(15, 10))
    cmap = plt.get_cmap("tab20")
    
    # クラス別に色分けして表示
    for i, cls in enumerate(top_classes):
        indices = [j for j, name in enumerate(class_names) if name == cls]
        if indices:
            plt.scatter(tsne_result[indices, 0], tsne_result[indices, 1], 
                       label=f"{cls} ({len(indices)})", 
                       color=cmap(i/len(top_classes)), 
                       alpha=0.7)
    
    plt.title("t-SNE Visualization (By Class)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    tsne_class_path = os.path.join(main_out_dir, "tsne_visualization_class.png")
    plt.savefig(tsne_class_path, bbox_inches='tight')
    plt.close()
    
    # 5. LOFによるアウトライア検出
    print(f"LOFによるアウトライア検出を実行中 (contamination={contamination})...")
    
    # クラスごとにアウトライア検出を行う
    class_outlier_flags, class_outlier_info = detect_class_aware_outliers(valid_features, class_names, contamination=contamination)
    
    # 全体でのアウトライア検出も実行
    global_outlier_flags = predict_outliers_lof(valid_features, contamination=contamination)
    
    # クラス内アウトライアとグローバルアウトライアを組み合わせる
    outlier_flags = np.logical_or(class_outlier_flags, global_outlier_flags).astype(int)
    
    # アウトライア検出結果の可視化
    plt.figure(figsize=(15, 10))
    color_map = {
        0: ('blue', 'Normal Samples'),
        1: ('red', 'Anomaly Samples')
    }
    
    for code in [0, 1]:
        indices = np.where(outlier_flags == code)[0]
        if len(indices) > 0:
            plt.scatter(tsne_result[indices, 0], tsne_result[indices, 1], 
                       c=color_map[code][0],
                       label=f"{color_map[code][1]} ({len(indices)})", 
                       alpha=0.7)
    
    plt.title("Local Outlier Factor Anomaly Detection Results")
    plt.legend(loc='best')
    plt.tight_layout()
    outlier_viz_path = os.path.join(main_out_dir, "outlier_detection_result.png")
    plt.savefig(outlier_viz_path)
    plt.close()
    
    # 6. 特定クラスのオブジェクトを抽出
    if target_classes is None:
        # 実験結果から効果が高かったクラスを優先
        target_classes = ["construction_vehicle", "bicycle", "motorcycle", "trailer"]
    
    # 各クラスの類似クラスも含める
    target_near_classes = {
        # YOLOX の COCO モデルで出力されるクラス名に合わせて最小限修正
        "construction_vehicle": ["truck"],       # 建設車両は COCO では truck として扱われる
        "bicycle": ["bicycle"],                  # COCO クラス名そのまま
        "motorcycle": ["motorcycle"],            # COCO クラス名そのまま
        "trailer": ["truck"],                    # COCO には trailer カテゴリがないため truck で代用
    }
    
    # 全ての近接クラスをフラット化
    all_near_classes = []
    for target in target_classes:
        all_near_classes.append(target)
        if target in target_near_classes:
            all_near_classes.extend(target_near_classes[target])
    
    target_indices = []
    for i, info in enumerate(valid_cropped_info):
        class_name = info[1].lower()
        for target in all_near_classes:
            if target.lower() in class_name:
                target_indices.append(i)
                break
    
    print(f"特定クラスのオブジェクト数: {len(target_indices)}")
    
    # 処理対象のインデックスを決定（アウトライアと特定クラス）
    process_indices = list(set(np.where(outlier_flags == 1)[0]).union(set(target_indices)))
    process_indices.sort()  # ソートして順序を保つ
    
    # 画像の多様性を確保するために類似画像をフィルタリング
    if len(process_indices) > 50 and len(process_indices) < 100:  # 十分な数がある場合のみ適用
        print("類似画像のフィルタリングを適用中...")
        # 対象画像の特徴量を抽出
        process_features = valid_features[process_indices]
        
        # 特徴量の類似度に基づいて多様性のある画像を選択
        diversity_indices = avoid_similar_images(
            [valid_cropped_info[i][0] for i in process_indices],
            process_features,
            similarity_threshold=0.95,  # コサイン類似度閾値
            max_per_cluster=3  # 各クラスタから最大3枚を選択
        )
        
        # 選択された多様な画像のインデックスを取得
        process_indices = [process_indices[i] for i in diversity_indices]
        print(f"多様性フィルタリング適用後: {len(process_indices)}枚の画像を処理対象に選択")
    
    # 10. YOLOで検出されたクラス名を保持する辞書を作成
    detected_classes_dict = {
        valid_cropped_info[i][0]: valid_cropped_info[i][1]
        for i in process_indices
    }
    
    # 11. 処理対象のパスをリストアップ
    process_paths = [valid_cropped_info[i][0] for i in process_indices]
    model_name = f"Qwen/Qwen2-VL-{qwen_model_size}-Instruct"
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_name)
    
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
    # 12. バッチ処理でキャプション生成
    if process_paths:
        print(f"バッチ処理で{len(process_paths)}個のオブジェクトのキャプションを生成中...")
        
        # 最適なバッチサイズを設定
        caption_batch_size = get_optimal_batch_size()
        
        # サブバッチに分割して処理
        sub_batch_size = 100  # 一度に処理する画像数
        all_descriptions = []
        
        # 中間結果ファイルのパス
        interim_results_file = os.path.join(main_out_dir, "interim_results.json")
        final_results = []
        target_class_results = {target: [] for target in target_classes}
        
        # チェックポイントファイルの確認
        checkpoint_file = os.path.join(main_out_dir, "process_checkpoint.json")
        if os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, 'r') as f:
                    checkpoint_data = json.load(f)
                    processed_batches = checkpoint_data.get('processed_batches', 0)
                    final_results = checkpoint_data.get('final_results', [])
                    print(f"チェックポイントを読み込みました: {processed_batches}バッチ処理済み ({len(final_results)}件)")
            except Exception as e:
                print(f"チェックポイント読み込みエラー: {e}")
                processed_batches = 0
        else:
            processed_batches = 0
        
        for batch_idx in range(0, len(process_paths), sub_batch_size):
            # 既に処理済みのバッチはスキップ
            if batch_idx < processed_batches * sub_batch_size:
                print(f"バッチ {batch_idx//sub_batch_size + 1} は既に処理済みです。スキップします。")
                continue
            
            sub_batch_paths = process_paths[batch_idx:batch_idx+sub_batch_size]
            sub_batch_indices = process_indices[batch_idx:min(batch_idx+sub_batch_size, len(process_indices))]
            
            print(f"サブバッチ処理中: {batch_idx+1}〜{min(batch_idx+sub_batch_size, len(process_paths))}/{len(process_paths)}...")
            
            # サブバッチのキャプション生成
            sub_batch_descriptions = generate_descriptions_batch(
                sub_batch_paths,
                model,
                processor,
                batch_size=caption_batch_size,
                detected_classes=detected_classes_dict
            )
            
            # サブバッチの結果処理
            batch_results = []
            for i, desc in enumerate(sub_batch_descriptions):
                if batch_idx + i >= len(process_indices):
                    continue
                if desc in common_classes:
                    continue
                idx = sub_batch_indices[i]
                info = valid_cropped_info[idx]
                path, cls_name, conf, bbox, original_path = info
                is_outlier = outlier_flags[idx] == 1
                
                # サンプルトークンを抽出
                sample_token = os.path.basename(original_path).split('.')[0]
                
                # 特定クラスに属するかチェック
                target_matches = []
                for target in target_classes:
                    if target.lower() in cls_name.lower():
                        target_matches.append(target)
                
                # キャプション解析
                top_concept = None
                is_final_outlier = False
                
                if desc:
                    # キャプション判定ロジック
                    desc_lower = desc.lower()
                    is_failed_caption = False
                    
                    if any(phrase in desc_lower for phrase in ["i'm sorry", "unable to", "i'm unable to", "cannot process"]):
                        is_failed_caption = True
                    
                    if not is_failed_caption:
                        # 類似度計算
                        probs_dict = parse_text_with_probability(desc, concept_list, clip_extractor)
                        
                        if probs_dict:
                            top_concept = next(iter(probs_dict))
                            # final outlier の判定基準: top_concept が common_classes にないこと
                            if top_concept not in common_classes:
                                is_final_outlier = True
                
                # 結果の保存
                result_info = {
                    'sample_token': sample_token,
                    'outlier_code': int(is_outlier),
                    'yolo_class': cls_name,
                    'is_yolo_potential': any(target.lower() in cls_name.lower() for target in target_classes),
                    'original_path': original_path,
                    'confidence': float(conf),
                    'top_concept': top_concept,
                    'is_final_outlier': is_final_outlier
                }
                
                batch_results.append(result_info)
                final_results.append(result_info)
                
                # 特定クラスの結果を保存
                for target in target_matches:
                    target_class_results[target].append(result_info)
                
                # ファイルのコピーと情報保存
                outlier_dir = os.path.join(output_dir, "outliers")
                inlier_dir = os.path.join(output_dir, "inliers")
                
                os.makedirs(outlier_dir, exist_ok=True)
                os.makedirs(inlier_dir, exist_ok=True)
                
                dest_dir = outlier_dir if is_outlier else inlier_dir
                
                # 元画像のコピー
                try:
                    if os.path.exists(original_path):
                        shutil.copy(original_path, os.path.join(dest_dir, os.path.basename(original_path)))
                        if is_final_outlier:
                            final_outlier_path = os.path.join(final_outlier_dir, os.path.basename(original_path))
                            shutil.copy(original_path, final_outlier_path)
                except Exception as e:
                    print(f"ファイルコピーエラー {original_path}: {e}")
                
                # 切り抜き画像のコピー
                if save_crops and os.path.exists(path):
                    shutil.copy(path, os.path.join(dest_dir, os.path.basename(path)))
                
                # 特定クラスの場合は専用ディレクトリにもコピー
                for target in target_matches:
                    target_dir = target_dirs[target]
                    
                    # YOLOクラスとtop_conceptの両方でマッチングをチェック
                    is_target_match = (
                        is_target_class_match(cls_name, target, target_near_classes) or
                        is_target_class_match(top_concept, target, target_near_classes)
                    )
                    
                    if is_target_match and os.path.exists(original_path):
                        os.makedirs(target_dir, exist_ok=True)
                        shutil.copy(original_path, os.path.join(target_dir, os.path.basename(original_path)))
                        print(f"Copied to {target} dir: {os.path.basename(original_path)}")
                        print(f"  YOLO class: {cls_name}")
                        print(f"  Top concept: {top_concept}")
                
                # 説明ファイルの保存
                if save_descriptions:
                    txt_path = os.path.join(dest_dir, f"{os.path.splitext(os.path.basename(path))[0]}_desc.txt")
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write(f"Sample Token: {sample_token}\n")
                        f.write(f"Outlier code: {int(is_outlier)} (0:inlier, 1:outlier)\n")
                        f.write(f"YOLO Class: {cls_name}\n")
                        f.write(f"Is YOLO potential candidate: {any(target.lower() in cls_name.lower() for target in target_classes)}\n")
                        f.write(f"Final outlier (Top concept not common): {is_final_outlier}\n")
                        f.write(f"Original Image Path: {original_path}\n")
                        f.write(f"Confidence={conf:.2f}, bbox={bbox}\n\n")
                        
                        f.write("Generated caption:\n")
                        if desc is not None:
                            f.write(desc + "\n\n")
                        else:
                            f.write("No caption generated.\n\n")
                        
                        if top_concept:
                            f.write(f"Top concept: {top_concept} (in common classes: {top_concept in common_classes})\n")
                
                # 確率可視化
                if save_probability_plots and 'probs_dict' in locals() and probs_dict:
                    plt.figure(figsize=(8, 4))
                    sorted_probs = sorted(probs_dict.items(), key=lambda item: item[1], reverse=True)[:10]
                    sorted_probs_keys = [k for k, v in sorted_probs]
                    sorted_probs_values = [v for k, v in sorted_probs]
                    plt.bar(sorted_probs_keys, sorted_probs_values, color='skyblue')
                    plt.xticks(rotation=45, ha='right')
                    plt.title("Top 10 Concept Similarities")
                    plt.ylabel("Cosine Similarity")
                    plt.tight_layout()
                    prob_png = os.path.join(dest_dir, f"{os.path.splitext(os.path.basename(path))[0]}_probs.png")
                    plt.savefig(prob_png)
                    plt.close()
            
            # サブバッチ処理後にGPUメモリを解放
            torch.cuda.empty_cache()
            
            # 中間結果を保存
            with open(interim_results_file, 'w') as f:
                json.dump(numpy_to_python(final_results), f, indent=2)
            
            # チェックポイントを更新
            checkpoint_data = {
                'timestamp': datetime.datetime.now().isoformat(),
                'processed_count': batch_idx//sub_batch_size + 1,
                'final_outliers': [fo for fo in final_results if fo['is_final_outlier']],
                'non_final_outliers': [nfo for nfo in final_results if not nfo['is_final_outlier']]
            }
            
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            
            print(f"サブバッチ {batch_idx//sub_batch_size + 1} 完了。中間結果を保存しました ({len(final_results)}件)")
        
        # 15. 結果をJSONファイルに保存
        # Final outliers
        final_outliers = [fo for fo in final_results if fo['is_final_outlier']]
        if final_outliers:
            final_outlier_path = os.path.join(main_out_dir, "final_outlier_results.json")
            with open(final_outlier_path, 'w') as f:
                json.dump(final_outliers, f, indent=2)
            print(f"Final outlier information saved to {final_outlier_path}")

        # Non-final outliers
        non_final_outliers = [nfo for nfo in final_results if not nfo['is_final_outlier']]
        if non_final_outliers:
            non_final_outlier_path = os.path.join(main_out_dir, "processed_non_final_results.json")
            with open(non_final_outlier_path, 'w') as f:
                json.dump(non_final_outliers, f, indent=2)
            print(f"Processed non-final object information saved to {non_final_outlier_path}")
        
        # 特定クラスの結果を保存
        for target, results in target_class_results.items():
            if results:
                target_results_path = os.path.join(main_out_dir, f"{target}_results.json")
                with open(target_results_path, 'w') as f:
                    json.dump(results, f, indent=2)
        
        # 16. 統計情報の計算と保存
        outlier_count = sum(1 for r in final_results if r['is_outlier']) if 'is_outlier' in final_results[0] else 0
        final_outlier_count = sum(1 for r in final_results if r['is_final_outlier']) if 'is_final_outlier' in final_results[0] else 0
        target_counts = {target: len(results) for target, results in target_class_results.items()} 
        
        stats = {
            "total_objects": len(valid_cropped_info),
            "processed_objects": len(process_indices),
            "outlier_count": outlier_count,
            "final_outlier_count": final_outlier_count,
            "target_counts": target_counts,
            "class_distribution": {k: v for k, v in class_counts.most_common()}
        }
        
        stats_path = os.path.join(main_out_dir, "statistics.json")
        with open(stats_path, 'w') as f:
            json.dump(numpy_to_python(stats), f, indent=2)
        
        # 17. 一時ファイルのクリーンアップ
        if cleanup_temp_files:
            try:
                print("一時ファイルのクリーンアップを実行中...")
                if os.path.exists(cropped_dir):
                    shutil.rmtree(cropped_dir)
            except Exception as e:
                print(f"一時ファイル削除エラー: {e}")
    
    else:
        print("処理対象のオブジェクトがありません。")
    
    # 18. リソースの解放
    if 'model' in locals():
        del model
    if 'processor' in locals():
        del processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 19. 処理時間の記録
    end_time_overall = time.time()
    elapsed_time = end_time_overall - start_time_overall
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"処理完了。総処理時間: {int(hours)}時間{int(minutes)}分{seconds:.1f}秒")
    
    timing_path = os.path.join(main_out_dir, "timing.txt")
    with open(timing_path, 'w') as f:
        f.write(f"開始時刻: {timestamp}\n")
        f.write(f"終了時刻: {datetime.datetime.now().strftime('%m%d_%H-%M-%S_JST')}\n")
        f.write(f"総処理時間: {int(hours)}時間{int(minutes)}分{seconds:.1f}秒\n")
    
    return main_out_dir, final_results



def signal_handler(sig, frame):
    """
    シグナルハンドラ関数：Ctrl+Cなどの割り込み信号を処理
    """
    print("\n処理を中断します。安全に終了しています...")
    # GPUメモリを解放
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    sys.exit(0)

def avoid_similar_images(image_paths, features, similarity_threshold=0.95, max_per_cluster=3):
    """
    特徴量の類似度に基づいて、似たような画像を除外し、多様性を確保する
    
    Args:
        image_paths: 画像パスのリスト
        features: 対応する特徴量 (embeddingベクトル)
        similarity_threshold: この値以上の類似度なら同じクラスタとみなす
        max_per_cluster: 各クラスタから最大いくつの画像を選ぶか
        
    Returns:
        selected_indices: 選択された画像のインデックス
    """
    n = len(image_paths)
    if n <= max_per_cluster:
        return list(range(n))  # 少なすぎる場合は全て選択
    
    # コサイン類似度行列を計算
    similarity_matrix = cosine_similarity(features)
    
    # 各画像のクラスタ割り当て
    clusters = [-1] * n
    next_cluster_id = 0
    
    for i in range(n):
        if clusters[i] == -1:
            # 新しいクラスタを作成
            clusters[i] = next_cluster_id
            
            # 類似画像を同じクラスタに割り当て
            for j in range(i+1, n):
                if clusters[j] == -1 and similarity_matrix[i, j] >= similarity_threshold:
                    clusters[j] = next_cluster_id
            
            next_cluster_id += 1
    
    # 各クラスタから画像を選択
    selected_indices = []
    for cluster_id in range(next_cluster_id):
        # このクラスタに属する画像のインデックス
        cluster_indices = [i for i in range(n) if clusters[i] == cluster_id]
        
        # 最大max_per_cluster個まで選択
        selected_count = min(len(cluster_indices), max_per_cluster)
        selected_indices.extend(cluster_indices[:selected_count])
    
    return selected_indices

# 日本語フォント設定関数の修正
def setup_japanese_font():
    """
    プロット用のフォント設定を行う関数
    日本語表示の代わりに英語表示を使用
    """
    import matplotlib
    
    # 英語表示に設定
    matplotlib.rcParams['font.family'] = 'DejaVu Sans'
    
    # 警告を抑制
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
    
    print("プロットを英語表示に設定しました")

# 最適なバッチサイズを取得する関数の追加
def get_optimal_batch_size(initial_size=8):
    """
    利用可能なGPUメモリに基づいて最適なバッチサイズを推定する
    """
    if not torch.cuda.is_available():
        return initial_size
    
    try:
        # GPUメモリ情報を取得
        free_memory, total_memory = torch.cuda.mem_get_info(0)
        free_gb = free_memory / (1024**3)
        
        # 利用可能なメモリに基づいてバッチサイズを調整
        if free_gb > 20:  # 20GB以上空き
            return 16
        elif free_gb > 10:  # 10GB以上空き
            return 8
        
        elif free_gb > 5:  # 5GB以上空き
            return 4
        else:
            return 2
    except Exception:
        # メモリ情報取得に失敗した場合は初期値を返す
        return initial_size

# NumPy配列をJSON形式に変換するためのヘルパー関数を追加
def numpy_to_python(obj):
    """
    NumPy配列やその他のJSON非対応型をPythonネイティブ型に変換する
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, tuple) or isinstance(obj, list):
        return [numpy_to_python(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: numpy_to_python(value) for key, value in obj.items()}
    else:
        return obj


###############################################################################
# メイン関数
###############################################################################
def main():
    # 日本語フォントの設定
    setup_japanese_font()
    
    parser = argparse.ArgumentParser(description="単一フォルダ内の画像からアウトライアを検出するスクリプト")
    parser.add_argument("--images_folder", type=str, default="data_nuscenes/samples/CAM_FRONT", help="画像が格納されているフォルダのパス")
    parser.add_argument("--output_dir", type=str, default="outlier_detection_results", help="結果を保存するフォルダのパス")
    parser.add_argument("--qwen_model_size", type=str, default="2B", choices=["2B", "7B"], help="Qwenモデルのサイズ (2B or 7B)")
    parser.add_argument("--contamination", type=float, default=0.1, help="アウトライア検出のcontamination値（アウトライアの割合）")
    parser.add_argument("--target_classes", nargs='+', default=["construction_vehicle", "bicycle", "motorcycle", "trailer"], 
                        help="特に注目するクラス名のリスト")
    parser.add_argument("--common_classes", nargs='+', default=["car", "pedestrian", "traffic_light", "traffic_sign"], 
                        help="一般的なクラス名のリスト（これらはアウトライアとして扱わない）")
    parser.add_argument("--no_save_crops", action='store_true', help="切り抜き画像を保存しない")
    parser.add_argument("--no_save_descriptions", action='store_true', help="説明テキストを保存しない")
    parser.add_argument("--no_save_probability_plots", action='store_true', help="確率プロットを保存しない")
    parser.add_argument("--cleanup_temp", action='store_true', help="処理終了後に一時ファイルを削除する")
    parser.add_argument("--max_images", type=int, default=None, help="処理する最大画像数")
    parser.add_argument("--seed", type=int, default=42, help="乱数シード")
    parser.add_argument("--minimal_io", action='store_true', 
                        help="I/O操作を最小限にする（--no_save_crops --no_save_descriptions --no_save_probability_plots と同等）")
    
    args = parser.parse_args()
    # args.images_folder = "projects/detect_rare_images/data/test_data" #data/nuscenes_image"
    # args.output_dir = "projects/detect_rare_images/outlier_detection_results_tmp"
    # minimal_ioが指定されている場合は全てのI/O削減オプションを有効にする
    if args.minimal_io or not args.max_images:
        args.no_save_crops = True
        args.no_save_descriptions = True
        args.no_save_probability_plots = True
    
    # シグナルハンドラを設定
    signal.signal(signal.SIGINT, signal_handler)
    
    # 引数を関数に渡す
    detect_outliers_single_folder(
        images_folder=args.images_folder,
        output_dir=args.output_dir,
        qwen_model_size=args.qwen_model_size,
        contamination=args.contamination,
        target_classes=args.target_classes,
        common_classes=args.common_classes,
        save_crops=not args.no_save_crops,
        save_descriptions=not args.no_save_descriptions,
        save_probability_plots=not args.no_save_probability_plots,
        cleanup_temp_files=args.cleanup_temp,
        max_images=args.max_images,
        seed=args.seed
    )
if __name__ == "__main__":
    main() 

