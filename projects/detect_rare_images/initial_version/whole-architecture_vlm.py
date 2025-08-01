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
    def __init__(self, model_path="yolov8l.pt", device=None, max_size=1280):
        if device is None:
            device = get_device()
        self.device = device
        self.max_size = max_size
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
            if width > self.max_size or height > self.max_size:
                # アスペクト比を維持したまま縮小
                if width > height:
                    new_width = self.max_size
                    new_height = int(height * (self.max_size / width))
                else:
                    new_height = self.max_size
                    new_width = int(width * (self.max_size / height))
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
                
                # 修正: クラス名取得の方法を改善
                if isinstance(names, list):
                    class_name = names[cls_id] if 0 <= cls_id < len(names) else f"class_{cls_id}"
                else:
                    class_name = names.get(cls_id, f"class_{cls_id}")

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

class VLMFeatureExtractor:
    """
    Obtain feature vectors (embeddings) for images or text using a Vision-Language Model.
    """
    def __init__(self, model_name="Qwen/Qwen2-VL-2B-Instruct", device=None, embedding_dim=1024, feature_layer=-1):
        if device is None:
            device = get_device()
        self.device = device
        self.embedding_dim = embedding_dim
        self.feature_layer = feature_layer
        
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model.eval()

    def get_image_embedding(self, image_path):
        """
        Get the VLM image feature (np.array, shape=(dim,)) from the given image path.
        """
        try:
            # 画像から特徴量とキャプションを同時に抽出
            features, _ = self.extract_features_and_caption(image_path)
            if features is not None:
                # 抽出された特徴量の次元を調整して返す
                # 多次元の場合は平均を取るなどして1次元に変換
                if len(features.shape) > 1:
                    features = np.mean(features, axis=0)
                return features
            
            # フォールバック: 従来の方法でも試す
            pil_image = Image.open(image_path).convert("RGB")
            messages = [
                {"role": "user", "content": [{"type": "image", "image": pil_image}]}
            ]
            
            # Process image
            inputs = self.processor.process_images(pil_image)
            inputs = {k: torch.tensor(v).unsqueeze(0).to(self.device) if isinstance(v, np.ndarray) else v 
                        for k, v in inputs.items()}
            
            with torch.no_grad():
                image_features = self.model.get_vision_encoder_outputs(**inputs)
                # Use pooled output as embedding
                pooled_features = image_features.pooler_output
            
            return pooled_features.cpu().numpy().squeeze(0)
        except Exception as e:
            print(f"Error extracting VLM embedding for {image_path}: {e}")
            # Return zeros vector as fallback with proper dimension
            return np.zeros(self.embedding_dim)

    def get_text_embedding(self, text_list):
        """
        Get the VLM text feature vectors (np.array, shape=(len(text_list), dim))
        for a list of texts.
        """
        if not text_list:
            return np.array([])
            
        embeddings = []
        for text in text_list:
            try:
                # フックを使用して特徴量を抽出
                features = None
                
                def hook_fn(module, input, output):
                    nonlocal features
                    features = output[0].detach().cpu().numpy()
                
                # 特定の層に登録
                hook_handle = self.model.language_model.model.layers[self.feature_layer].register_forward_hook(hook_fn)
                
                # テキスト入力の処理
                inputs = self.processor(text=text, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # フォワードパスを実行
                with torch.no_grad():
                    _ = self.model(**inputs)
                
                # フックを解除
                hook_handle.remove()
                
                # 特徴量が抽出されていれば使用、そうでなければフォールバック
                if features is not None:
                    if len(features.shape) > 1:
                        # 最初のトークンの埋め込みを使用するか、平均を取る
                        features = np.mean(features, axis=0)
                    embeddings.append(features)
                else:
                    print(f"Warning: Failed to extract features for text: {text[:30]}...")
                    embeddings.append(np.zeros(self.embedding_dim))
                    
            except Exception as e:
                print(f"Error extracting text embedding: {e}")
                embeddings.append(np.zeros(self.embedding_dim))
                
        return np.array(embeddings)

    def extract_features_and_caption(self, image_path, max_tokens=150):
        """
        画像から特徴量とキャプションを同時に抽出する
        
        Args:
            image_path: 画像ファイルへのパス
            max_tokens: 生成するトークンの最大数
            
        Returns:
            tuple: (features, caption)
                - features: np.array, 特徴量ベクトル
                - caption: str, 生成されたキャプション
        """
        try:
            # 画像の読み込みとBase64エンコード
            base64_image = encode_image(image_path)
            if not base64_image:
                return None, None
                
            # プロンプトの設定
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
                            "text": "Describe this image focusing on unusual or safety-relevant objects."
                        },
                    ],
                }
            ]

            # 推論の準備
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 特徴量抽出のためのフック設定
            features = None
            
            def hook_fn(module, input, output):
                nonlocal features
                features = output[0].detach().cpu().numpy()
            
            # 特定の層に登録（モデルの構造によって調整が必要かもしれません）
            hook_handle = self.model.language_model.model.layers[self.feature_layer].register_forward_hook(hook_fn)
            
            # 推論と生成
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=max_tokens)
                
            # キャプション生成
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            caption = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            # フックを解除
            hook_handle.remove()
            
            return features, caption
        except Exception as e:
            print(f"Error in extract_features_and_caption for {image_path}: {e}")
            return None, None

def train_baseline_model(baseline_folder="projects/detect_rare_images/data/data/nuscenes_image", 
                        lim1=0.8, 
                        extractor_type="clip",
                        model=None,
                        processor=None):
    detector = DetectorYOLO(model_path="yolov8l.pt")
    
    # Feature extractor selection
    if extractor_type.lower() == "vlm":
        extractor = VLMFeatureExtractor(feature_layer=-2)  # 一般的には-2層が良い特徴量を持つことが多い
    else:  # Default is CLIP
        extractor = CLIPFeatureExtractor(model_name="openai/clip-vit-base-patch32")
    
    cropped_dir = "baseline_cropped_objects"
    os.makedirs(cropped_dir, exist_ok=True)

    # Process images for baseline
    all_files = [f for f in os.listdir(baseline_folder) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif"))]
    
    # Check if there are enough files
    if len(all_files) == 0:
        print(f"No image files found in {baseline_folder}")
        return None, None, None, None, None, None, None, None, None, None, None, None
    
    random.shuffle(all_files)
    num_baseline_files = max(1, int(lim1 * len(all_files)))  # Ensure at least one file
    baseline_files = all_files[:num_baseline_files]

    # Load VLM model for descriptions if not provided
    if model is None or processor is None:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct", torch_dtype="auto", device_map="auto"
        )
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

    # Detect and crop objects
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

    # Extract features and generate descriptions
    features = []
    class_names = []
    descriptions = []
    
    for (obj_path, cls_name, conf, bbox, _) in all_cropped_info:
        # Get image embedding
        emb = extractor.get_image_embedding(obj_path)
        features.append(emb)
        class_names.append(cls_name)
        
        # Generate description
        desc = generate_description(obj_path, model, processor)
        descriptions.append(desc)
        
        # Save description to file
        desc_path = f"{os.path.splitext(obj_path)[0]}_desc.txt"
        with open(desc_path, "w", encoding="utf-8") as f:
            f.write(f"Class={cls_name}, conf={conf:.2f}, bbox={bbox}\n\n")
            f.write("Generated caption:\n")
            f.write(desc + "\n\n")

    features = np.array(features)
    if len(features) == 0:
        print("No features extracted from baseline. Exiting...")
        return None, None, None, None, None, None, None, None, None, None, None, None

    # Create text embeddings from descriptions
    text_features = []
    # if extractor_type.lower() == "vlm":
        # VLMの場合は同じVLMで特徴量を抽出
    for desc in descriptions:
        if desc:
            text_emb = extractor.get_text_embedding([desc])[0]
            text_features.append(text_emb)
    
    text_features = np.array(text_features) if text_features else np.array([])

    # Train anomaly detection models
    iso_model = train_isolation_forest(features, contamination=0.2)
    iso_model_text = train_isolation_forest(text_features, contamination=0.2) if len(text_features) > 0 else None

    # Prepare class labels
    unique_labels = list(set(class_names))
    label2id = {lb: i for i, lb in enumerate(unique_labels)}
    label_ids = [label2id[lb] for lb in class_names]
    candidate_labels = unique_labels

    # Generate t-SNE visualizations
    # Image feature t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    baseline_tsne = tsne.fit_transform(features)
    
    # Text feature t-SNE
    baseline_text_tsne = None
    if len(text_features) > 0:
        try:
            # 修正: perplexityのエラー処理を追加
            perplexity = min(30, max(5, len(text_features) - 1))
            if perplexity <= 1:  # perplexityは1より大きい必要がある
                print("Warning: Not enough text features for t-SNE visualization")
            else:
                text_tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
                baseline_text_tsne = text_tsne.fit_transform(text_features)
                
                # Visualize text embeddings
                plt.figure(figsize=(10,7))
                plt.scatter(baseline_text_tsne[:,0], baseline_text_tsne[:,1], alpha=0.5)
                plt.title("Baseline Text Embeddings t-SNE Visualization")
                plt.savefig("baseline_text_tsne.png")
                plt.close()
        except Exception as e:
            print(f"Error generating t-SNE for text features: {e}")
            baseline_text_tsne = None
    
    # Visualize image embeddings
    plt.figure(figsize=(10,7))
    plt.scatter(baseline_tsne[:,0], baseline_tsne[:,1], alpha=0.5)
    plt.title("Baseline Image Embeddings t-SNE Visualization")
    plt.savefig("baseline_tsne.png")
    plt.close()

    # Visualize top classes
    from collections import Counter
    class_counts = Counter(class_names)
    top_classes = [cls for cls, count in class_counts.most_common(20)]
    print("Top 20 classes:", top_classes)
    
    plot_indices = [i for i, lb in enumerate(class_names) if lb in top_classes]
    filtered_tsne = baseline_tsne[plot_indices]
    filtered_labels = [class_names[i] for i in plot_indices]
    
    plt.figure(figsize=(15, 10))
    cmap = plt.get_cmap("tab20")
    for i, lb in enumerate(top_classes):
        indices = [j for j, x in enumerate(filtered_labels) if x == lb]
        plt.scatter(filtered_tsne[indices, 0], filtered_tsne[indices, 1], 
                   label=lb, color=cmap(i/len(top_classes)), alpha=0.7)
    plt.title("Baseline t-SNE Visualization (Top 20 Classes)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("baseline_tsne_class.png", bbox_inches='tight')
    plt.close()

    return (extractor, candidate_labels, features, baseline_tsne, class_names, 
            iso_model, text_features, baseline_text_tsne, iso_model_text, 
            descriptions, model, processor)

def detect_new_images(
    new_images_folder,
    extractor,
    candidate_labels,
    out_dir="new_cropped_objects",
    baseline_features=None,
    baseline_tsne=None,
    baseline_class_names=None,
    iso_model=None,
    iso_model_text=None,
    threshold_percentile=95,
    lim1=0.8,
    lim2=1.0,
    save_outlier_list=True,
    extractor_type="clip",
    model=None,
    processor=None
):
    detector = DetectorYOLO(model_path="yolov8l.pt")
    today = datetime.datetime.today().strftime("%m%d")
    out_dir = f"{out_dir}_{today}_{extractor_type}"
    save_outliers = save_outlier_list
    os.makedirs(out_dir, exist_ok=True)

    all_files = [f for f in os.listdir(new_images_folder) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif"))]
    
    # Check if there are enough files
    if len(all_files) == 0:
        print(f"No image files found in {new_images_folder}")
        return
        
    random.shuffle(all_files)
    
    # 修正: 範囲チェック (ファイルが十分にあるか確認)
    start_idx = int(lim1 * len(all_files))
    end_idx = int(lim2 * len(all_files))
    if start_idx >= len(all_files) or start_idx == end_idx:
        print(f"Warning: Invalid slice range ({start_idx}:{end_idx}) for {len(all_files)} files")
        if len(all_files) > 0:
            start_idx = 0
            end_idx = min(10, len(all_files))  # Default to first 10 files or less
        else:
            print("No files to process")
            return
    
    new_files = all_files[start_idx:end_idx]

    # Load VLM model for descriptions if not provided
    if model is None or processor is None:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct", torch_dtype="auto", device_map="auto"
        )
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

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
    class_names_new = []
    text_features_new = []
    descriptions_new = []
    
    for (obj_path, cls_name, conf, bbox, original_path) in all_cropped_info:
        # Extract image features
        emb = extractor.get_image_embedding(obj_path)
        features_new.append(emb)
        class_names_new.append(cls_name)
        
        # Generate and store descriptions
        desc = generate_description(obj_path, model, processor)
        descriptions_new.append(desc)
        
        # Extract text features if we have a description
        if desc and extractor_type.lower() != "vlm":  # VLM already combines image+text
            text_emb = extractor.get_text_embedding([desc])[0]
            text_features_new.append(text_emb)
    
    features_new = np.array(features_new)
    if len(features_new) == 0:
        print("No features extracted from new images. Exiting...")
        return
    
    # Process text features if available
    text_features_new = np.array(text_features_new) if text_features_new else np.array([])
    
    # Combine features for visualization
    combined_features = np.vstack([baseline_features, features_new])

    # Create t-SNE visualization
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    combined_tsne = tsne.fit_transform(combined_features)
    baseline_tsne_2d = combined_tsne[:len(baseline_features)]
    new_tsne = combined_tsne[len(baseline_features):]

    # Calculate outliers based on t-SNE neighbors
    nbrs = NearestNeighbors(n_neighbors=10).fit(baseline_tsne)
    distances, indices = nbrs.kneighbors(new_tsne)
    avg_distances = distances.mean(axis=1)
    threshold_tsne = np.percentile(avg_distances, threshold_percentile)
    tsne_outliers = (avg_distances > threshold_tsne).astype(int)  # 1: outlier, 0: inlier

    # Calculate outliers using Isolation Forest
    if_labels = predict_outliers(features_new, iso_model) # 1: inlier, -1: outlier
    if_outliers = ((if_labels==-1).astype(int))
    
    # Combine outlier detection methods
    combined_outlier_code = tsne_outliers*2 + if_outliers

    # Calculate text-based outliers if available
    if len(text_features_new) > 0 and iso_model_text is not None:
        text_if_labels = predict_outliers(text_features_new, iso_model_text)
        text_if_outliers = ((text_if_labels==-1).astype(int))
        print(f"Detected {sum(text_if_outliers)} text-based outliers")
        
        # Update combined outlier code to include text outliers
        # 4: text outlier only, 5: text + IF, 6: text + t-SNE, 7: all three
        for i, is_text_outlier in enumerate(text_if_outliers):
            if is_text_outlier:
                combined_outlier_code[i] += 4

    # Visualize results
    plt.figure(figsize=(18,12))
    plt.scatter(baseline_tsne_2d[:,0], baseline_tsne_2d[:,1],
                c='lightgray', alpha=0.4, label="Baseline")

    color_map = {
        0: ('blue', 'Inlier (all normal)'),
        1: ('red',  'IF outlier only'),
        2: ('orange','t-SNE outlier only'),
        3: ('purple','Both IF and t-SNE outlier'),
        4: ('cyan', 'Text outlier only'),
        5: ('magenta', 'Text + IF outlier'),
        6: ('yellow', 'Text + t-SNE outlier'),
        7: ('black', 'All three outlier methods')
    }
    
    for code in sorted(color_map.keys()):
        indices_code = np.where(combined_outlier_code == code)[0]
        if len(indices_code) == 0:
            continue
        x_sub = new_tsne[indices_code,0]
        y_sub = new_tsne[indices_code,1]
        plt.scatter(x_sub, y_sub, c=color_map[code][0],
                    label=f"{color_map[code][1]} ({len(indices_code)})", alpha=0.8)

    plt.title("Combined Outlier Detection Visualization")
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
    plt.tight_layout()
    plt.savefig("combined_outlier_detection.png", bbox_inches='tight')
    plt.close()

    # Create output folders for each outlier type
    out_folders = {
        0: "inlier_normal",
        1: "if_outlier_only",
        2: "tsne_outlier_only",
        3: "both_if_tsne_outlier",
        4: "text_outlier_only",
        5: "text_if_outlier",
        6: "text_tsne_outlier",
        7: "all_outliers"
    }
    
    for k, v in out_folders.items():
        folder_path = os.path.join(out_dir, v)
        os.makedirs(folder_path, exist_ok=True)

    # Save detected outliers
    detected_images_dir = "detected_images"
    outliner_dir = "outliner"
    os.makedirs(detected_images_dir, exist_ok=True)
    os.makedirs(outliner_dir, exist_ok=True)
    
    outlier_info = []
    
    for i, ((path, cls_name, conf, bbox, original_path), desc) in enumerate(zip(all_cropped_info, descriptions_new)):
        out_code = combined_outlier_code[i] if i < len(combined_outlier_code) else 0
        folder_name = out_folders.get(out_code, "others")
        save_folder = os.path.join(out_dir, folder_name)
        
        # Copy the cropped image
        base_name = os.path.basename(path)
        dst_path = os.path.join(save_folder, base_name)
        try:
            # 修正: ファイルコピー時のエラーハンドリング
            if os.path.exists(dst_path):
                timestamp = datetime.datetime.now().strftime("%H%M%S")
                dst_path = os.path.join(save_folder, f"{os.path.splitext(base_name)[0]}_{timestamp}.jpg")
            shutil.copy(path, dst_path)
        except Exception as e:
            print(f"Error copying file {path} to {dst_path}: {e}")
            continue
        
        # For outliers, also copy original image to detected_images
        if out_code != 0:
            try:
                # Copy original image to the folder corresponding to out_code
                orig_filename = os.path.basename(original_path)
                target_path = os.path.join(save_folder, orig_filename)
                if os.path.exists(target_path):
                    timestamp = datetime.datetime.now().strftime("%H%M%S")
                    target_path = os.path.join(save_folder, f"{os.path.splitext(orig_filename)[0]}_{timestamp}.jpg")
                shutil.copy(original_path, target_path)
                
                # Copy to detected_images folder
                save_img = os.path.join(detected_images_dir, os.path.basename(original_path))
                if os.path.exists(save_img):
                    timestamp = datetime.datetime.now().strftime("%H%M%S")
                    save_img = os.path.join(detected_images_dir, f"{os.path.splitext(os.path.basename(original_path))[0]}_{timestamp}.jpg")
                shutil.copy(original_path, save_img)
            except Exception as e:
                print(f"Error copying outlier image {original_path}: {e}")
                continue
            
            # Add to outlier info
            sample_token = os.path.splitext(os.path.basename(original_path))[0]
            outlier_info.append({
                'sample_token': sample_token,
                'outlier_code': int(out_code),
                'original_path': original_path,
                'class_name': cls_name,
                'confidence': float(conf),
                'description': desc
            })

        # Calculate text-based similarities if we have a description
        probs_dict = {}
        if desc and desc not in ["I am unable to", "I'm unable to analyze", "unable to process"]:
            probs_dict = parse_text_with_probability(desc, candidate_labels, extractor)

        # Write description to text file
        txt_path = os.path.join(save_folder, f"{os.path.splitext(base_name)[0]}_desc.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"Outlier code: {out_code} ({out_folders.get(out_code, 'unknown')})\n")
            f.write(f"Cropped Image Path: {path}\n")
            f.write(f"Original Image Path: {original_path}\n")
            f.write(f"Class={cls_name}, conf={conf:.2f}, bbox={bbox}\n\n")
            f.write("Generated caption:\n")
            f.write(desc + "\n\n")
            if probs_dict:
                f.write("Cosine Similarity w.r.t candidate labels:\n")
                for k_, v_ in probs_dict.items():
                    f.write(f"  {k_}: {v_:.4f}\n")
            else:
                f.write("No similarity calculations available.\n")

        # Plot similarity bar chart if available
        if probs_dict:
            plt.figure(figsize=(8, 4))
            plt.bar(list(probs_dict.keys())[:10], list(probs_dict.values())[:10], color='skyblue')
            plt.xticks(rotation=45)
            plt.title("Top 10 Cosine Similarities")
            plt.tight_layout()
            prob_png = os.path.join(save_folder, f"{os.path.splitext(base_name)[0]}_probs.png")
            plt.savefig(prob_png)
            plt.close()
    
    # Save outlier information to JSON if requested
    if save_outlier_list and outlier_info:
        outlier_path = os.path.join("outlier_detection_results.json")
        with open(outlier_path, 'w') as f:
            json.dump(outlier_info, f, indent=2)
        print(f"Outlier information saved to {outlier_path}")
    
    print(f"Analysis complete. Processed {len(all_cropped_info)} objects, found {len(outlier_info)} outliers.")

def generate_description(image_path, model, processor):
    """
    Generate a description for an image using a Vision-Language Model.
    """
    try:
        # Base64エンコード機能を使用して画像を処理
        base64_image = encode_image(image_path)
        if not base64_image:
            return "I am unable to describe this image."
            
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
                        "text": "Describe this image in detail, focusing on all visible objects and their characteristics."
                    },
                ],
            }
        ]
        
        # 推論の準備
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
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # 推論と生成
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)
            
        # キャプション生成
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        caption = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        return caption
    except Exception as e:
        print(f"Error generating description for {image_path}: {e}")
        return "I am unable to describe this image."

def train_isolation_forest(features, contamination=0.1):
    """
    Train an Isolation Forest model for anomaly detection.
    """
    model = IsolationForest(
        n_estimators=100,
        max_samples="auto",
        contamination=contamination,
        random_state=42
    )
    model.fit(features)
    return model

def predict_outliers(features, model):
    """
    Use a trained Isolation Forest model to predict outliers.
    Returns 1 for inliers and -1 for outliers.
    """
    return model.predict(features)

def parse_text_with_probability(text, candidate_labels, extractor):
    """
    Calculate cosine similarity between text embedding and candidate label embeddings.
    Returns a dictionary of label:similarity pairs sorted by similarity.
    """
    if not isinstance(text, str) or not text or not candidate_labels:
        return {}
    
    try:
        # Get text embedding
        text_embedding = extractor.get_text_embedding([text])[0]
        
        # Get candidate embeddings
        candidate_embeddings = extractor.get_text_embedding(candidate_labels)
        
        # Calculate cosine similarity
        norm_text = text_embedding / np.linalg.norm(text_embedding)
        norm_cands = candidate_embeddings / np.linalg.norm(candidate_embeddings, axis=1, keepdims=True)
        similarities = np.dot(norm_cands, norm_text)
        
        # Create sorted dictionary
        similarity_dict = {label: float(sim) for label, sim in zip(candidate_labels, similarities)}
        return dict(sorted(similarity_dict.items(), key=lambda x: x[1], reverse=True))
        
    except Exception as e:
        print(f"Error parsing text: {e}")
        return {}

def main():
    # 1) Load common models once to reuse
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct", torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    
    # 2) train baseline model
    baseline_folder = "data_nuscenes/samples/CAM_FRONT" 
    lim1 = 0.002
    lim2 = 0.003
    
    # Added extractor_type parameter to choose between CLIP and VLM
    extractor_type = "vlm" # "clip" or "vlm"
    results = train_baseline_model(
        baseline_folder, 
        lim1, 
        extractor_type=extractor_type,
        model=model,
        processor=processor
    )
    
    if results is None or len(results) < 6 or results[0] is None:
        return
    
    # Unpack all the returned values from train_baseline_model
    extractor, candidate_labels, baseline_features, baseline_tsne, class_names, \
    iso_model, text_features, baseline_text_tsne, iso_model_text, \
    descriptions, model, processor = results
    
    if baseline_features is None:
        return
    
    # 3) detect rare images in new folder
    new_images_folder = baseline_folder
    detect_new_images(
        new_images_folder=new_images_folder,
        extractor=extractor,
        candidate_labels=candidate_labels,
        baseline_features=baseline_features,
        baseline_tsne=baseline_tsne,
        baseline_class_names=class_names,
        iso_model=iso_model,
        iso_model_text=iso_model_text,
        threshold_percentile=80,
        lim1=lim1,
        lim2=lim2,
        extractor_type=extractor_type,
        model=model,
        processor=processor
    )

    print(f"Using device: {get_device()}")
    print("\nAll Done.")


if __name__ == "__main__":
    main()