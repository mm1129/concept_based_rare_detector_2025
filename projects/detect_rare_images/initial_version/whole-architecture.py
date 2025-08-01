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
from collections import Counter
import time
import openai

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
# 1. 2D Detection (YOLO) - REMOVED
###############################################################################
def get_device():
    if torch.cuda.is_available():
        return "cuda:0"
    else:
        return "cpu"
device = get_device()

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
        print(output_text)
        response = output_text[0]
        return response

    except Exception as e:
        print(f"Qwen model error generating caption for {image_path}: {e}")
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
def train_baseline_model(baseline_folder, qwen_model_size, use_concept_list, concept_list, main_out_dir, lim1=0.8, train_indices=None):
    clip_extractor = CLIPFeatureExtractor(model_name="openai/clip-vit-base-patch32")
    
    # Load Qwen2VL model for captioning
    model_name = f"Qwen/Qwen2-VL-{qwen_model_size}-Instruct"
    print(f"Loading {model_name} model for image captioning...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_name)

    all_files = [f for f in os.listdir(baseline_folder) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif"))]
    random.shuffle(all_files)
    
    if train_indices is not None:
        baseline_files = [all_files[i] for i in train_indices]
    else:
        baseline_files = all_files[:int(lim1 * len(all_files))]

    features = []
    image_captions = []
    image_paths = []

    for file_name in baseline_files:
        try:
            image_path = os.path.join(baseline_folder, file_name)
            
            # Extract CLIP features for the whole image
            emb = clip_extractor.get_image_embedding(image_path)
            features.append(emb)
            
            # Generate caption for the whole image
            caption = generate_description(image_path, model, processor)
            image_captions.append(caption)
            image_paths.append(image_path)
            
        except Exception as e:
            print(f"Error processing baseline image {file_name}: {e}")
            continue

    features = np.array(features)
    if len(features) == 0:
        print("No features extracted from baseline. Exiting...")
        return None, None, None, None, None, None

    iso_model = train_isolation_forest(features, contamination=0.2)

    if use_concept_list:
        print("Using predefined concept list for candidate labels.")
        candidate_labels = concept_list
    else:
        print("Generating candidate labels from captions.")
        # Extract unique words from captions to use as candidate labels
        unique_words = set()
        for caption in image_captions:
            if caption:
                words = caption.lower().replace('.', ' ').replace(',', ' ').split()
                unique_words.update(words)
        candidate_labels = list(unique_words)
        print(f"Generated {len(candidate_labels)} candidate labels from captions")

    # t-SNE visualization for baseline
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    baseline_tsne = tsne.fit_transform(features)
    
    plt.figure(figsize=(10,7))
    plt.scatter(baseline_tsne[:,0], baseline_tsne[:,1], alpha=0.5)
    plt.title("Baseline t-SNE Visualization")
    baseline_tsne_path = os.path.join(main_out_dir, "baseline_tsne.png")
    plt.savefig(baseline_tsne_path)
    plt.close()

    # Clean up
    del model
    del processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return clip_extractor, candidate_labels, features, baseline_tsne, image_captions, iso_model


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
    baseline_captions=None,
    iso_model=None,
    threshold_percentile=95,
    lim1=0.8,
    lim2=1.0,
    save_outliers=True,
    save_outlier_list=True,
    test_indices=None
):
    run_specific_out_dir = os.path.join(main_out_dir, f"detection_lim{lim1}-{lim2}")
    os.makedirs(run_specific_out_dir, exist_ok=True)

    # Load Qwen2VL model for captioning
    model_name = f"Qwen/Qwen2-VL-{qwen_model_size}-Instruct"
    print(f"Loading {model_name} model for image captioning...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_name)

    all_files = [f for f in os.listdir(new_images_folder) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif"))]
    random.shuffle(all_files)
    
    if test_indices is not None:
        new_files = [all_files[i] for i in test_indices]
    else:
        new_files = all_files[int(lim1 * len(all_files)): int(lim2 * len(all_files))]

    features_new = []
    image_captions = []
    image_paths = []
    
    for file_name in new_files:
        try:
            image_path = os.path.join(new_images_folder, file_name)
            
            # Extract CLIP features for the whole image
            emb = clip_extractor.get_image_embedding(image_path)
            features_new.append(emb)
            
            # Generate caption for the whole image
            caption = generate_description(image_path, model, processor)
            image_captions.append(caption)
            image_paths.append(image_path)
            
        except Exception as e:
            print(f"Error processing new image {file_name}: {e}")
            continue

    features_new = np.array(features_new)
    if len(features_new) == 0:
        print("No features extracted from new images. Exiting...")
        return

    combined_features = np.vstack([baseline_features, features_new])

    # t-SNE による次元削減と座標化
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    combined_tsne = tsne.fit_transform(combined_features)
    baseline_tsne_2d = combined_tsne[:len(baseline_features)]
    new_tsne = combined_tsne[len(baseline_features):]

    # t-SNE による異常検知
    nbrs = NearestNeighbors(n_neighbors=10).fit(baseline_tsne_2d)
    distances, indices = nbrs.kneighbors(new_tsne)
    avg_distances = distances.mean(axis=1)
    threshold_tsne = np.percentile(avg_distances, threshold_percentile)
    tsne_outliers = (avg_distances > threshold_tsne).astype(int)

    # IsolationForest による異常検知
    if_labels = predict_outliers(features_new, iso_model)
    if_outliers = ((if_labels==-1).astype(int))
    combined_outlier_code = tsne_outliers*2 + if_outliers

    plt.figure(figsize=(18,12))

    color_map = {
        0: ('blue', 'Inlier (both normal)'),
        1: ('red',  'IF outlier only'),
        2: ('orange','t-SNE outlier only'),
        3: ('purple','Both outlier'),
    }

    plt.scatter(baseline_tsne_2d[:,0], baseline_tsne_2d[:,1],
                c='lightgray', alpha=0.4, label="Baseline")

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

    detected_images_dir = os.path.join(main_out_dir, "detected_images_overall")
    os.makedirs(detected_images_dir, exist_ok=True)

    outlier_info = []
    final_folder = os.path.join(main_out_dir, "final_outliers")
    os.makedirs(final_folder, exist_ok=True)
    
    for i, (out_code, image_path, caption) in enumerate(zip(combined_outlier_code, image_paths, image_captions)):
        folder_name = out_folders.get(out_code, "others")
        save_folder = os.path.join(run_specific_out_dir, folder_name)
        
        # Initialize is_final_outlier as False
        is_final_outlier = False
        top_concept = None
        
        # Only process potential outliers
        if out_code != 0:
            if not caption or "unable to" in caption.lower() or "i'm unable to" in caption.lower() or "cannot process" in caption.lower():
                print(f"Caption generation failed or indicated inability for {image_path}. Skipping similarity calculation.")
                probs_dict = {}
            else:
                probs_dict = parse_text_with_probability(caption, candidate_labels, clip_extractor) 
                
                # Check if the top concept is not in common classes
                if probs_dict:
                    top_concept = next(iter(probs_dict))  # Get first key (highest similarity)
                    if top_concept not in common_classes:
                        is_final_outlier = True
        else:
            probs_dict = {}
        
        # Copy files based on final outlier status
        if is_final_outlier:
            save_img = os.path.join(detected_images_dir, os.path.basename(image_path))
            shutil.copy(image_path, save_img)
        
            # Always copy to appropriate outlier code folder for analysis
            shutil.copy(image_path, os.path.join(save_folder, os.path.basename(image_path)))
                
            # Create description file for all images
            txt_path = os.path.join(save_folder, f"{os.path.splitext(os.path.basename(image_path))[0]}_desc.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(f"Outlier code: {out_code} (0:inlier_inlier, 1:IFonly, 2:tSNEonly, 3:both)\n")
                f.write(f"Final outlier: {is_final_outlier}\n")
                f.write(f"Image Path: {image_path}\n\n")
                
                if out_code != 0:  # Only for potential outliers
                    f.write("Generated caption:\n")
                    f.write(caption + "\n\n" if caption else "No caption generated\n\n")
                    if probs_dict:
                        f.write("Cosine Similarity w.r.t candidate labels:\n")
                        for k_, v_ in probs_dict.items():
                            f.write(f"  {k_}: {v_:.4f}\n")
                        f.write(f"\nTop concept: {top_concept} (in common classes: {top_concept in common_classes})\n")
                    else:
                        f.write("Skipped similarity calculation due to caption generation failure.\n")

        # Create probability visualization for non-inliers with valid probs
        if out_code != 0 and probs_dict and len(probs_dict) > 0:
            plt.figure(figsize=(8, 4))
            plt.bar(list(probs_dict.keys())[:10], list(probs_dict.values())[:10], color='skyblue')
            plt.xticks(rotation=45)
            plt.title("Top 10 Concept Similarities")
            plt.tight_layout()
            prob_png = os.path.join(save_folder, f"{os.path.splitext(os.path.basename(image_path))[0]}_probs.png")
            plt.savefig(prob_png)
            plt.close()
        
        # Add to outlier info if it's a final outlier
        if is_final_outlier:
            sample_token = os.path.splitext(os.path.basename(image_path))[0]
            outlier_info.append({
                'sample_token': sample_token,
                'outlier_code': int(out_code),
                'original_path': image_path,
                'top_concept': top_concept
            })
    
    # Save outlier information if requested
    if save_outlier_list and outlier_info:
        outlier_path = os.path.join(main_out_dir, "outlier_detection_results.json")
        with open(outlier_path, 'w') as f:
            json.dump(outlier_info, f, indent=2)
        print(f"Outlier information saved to {outlier_path}")

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

def main():
    parser = argparse.ArgumentParser(description="Detect rare images using object detection, CLIP, and outlier detection.")
    parser.add_argument("--baseline_folder", type=str, default="data_nuscenes/samples/CAM_FRONT", help="Path to the baseline image folder.")
    parser.add_argument("--new_images_folder", type=str, default=None, help="Path to the new image folder (defaults to baseline_folder).")
    parser.add_argument("--output_base_dir", type=str, default="rare_whole_architecture", help="Base directory for saving results.")
    parser.add_argument("--lim1", type=float, default=0.5, help="Start fraction for baseline/new image split.")
    parser.add_argument("--lim2", type=float, default=1.0, help="End fraction for baseline/new image split.")
    parser.add_argument("--qwen_model_size", type=str, default="2B", choices=["2B", "7B"], help="Size of the Qwen2VL model to use (2B or 7B).")
    parser.add_argument("--no_concept_list", action='store_true', help="Don't use the predefined concept list, use dynamic labels instead.")
    parser.add_argument("--threshold_percentile", type=int, default=65, help="Percentile threshold for t-SNE outlier detection (lower values detect more outliers).")
    parser.add_argument("--no_save_outlier_list", action='store_true', help="Do not save the outlier detection results to a JSON file.")
    parser.add_argument("--common_classes", nargs='+', 
                        default=["car", "truck", "bus", "pedestrian", 
                                 "traffic light", "traffic sign"], 
                        help="List of common classes to filter out from outliers")
    parser.add_argument("--split_mode", type=str, choices=["standard", "10fold"], default="10fold", 
                        help="Data split mode: 'standard' for simple lim1/lim2 split or '10fold' for 10-part split with odd parts for baseline")
    parser.add_argument("--max_lim", type=float, default=1.0, 
                        help="Maximum fraction of images to use from dataset (default: 1.0)")

    args = parser.parse_args()

    if args.new_images_folder is None:
        args.new_images_folder = args.baseline_folder

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H-%M")
    main_out_dir = os.path.join(args.output_base_dir, f"run_{args.split_mode}_{timestamp}")
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
        train_indices = None
        test_indices = None
    else:  # "10fold"モード
        print(f"  10-fold Split - max_lim: {args.max_lim}")
        print(f"  Using odd parts (1,3,5,7,9) for baseline, even parts (0,2,4,6,8) for testing")
        lim1 = 0
        lim2 = args.max_lim
        train_indices = []
        test_indices = []
        
    print(f"  Qwen Model Size: {args.qwen_model_size}")
    print(f"  Use Concept List: {use_concept_list}")
    print(f"  t-SNE Threshold Percentile: {args.threshold_percentile}")
    print(f"  Save Outlier List: {not args.no_save_outlier_list}")
    print(f"  Common Classes: {args.common_classes}")

    # 標準モードと10foldモードで処理を分ける
    if args.split_mode == "standard":
        # ベースラインモデルを学習
        clip_extractor, candidate_labels, baseline_features, baseline_tsne, baseline_captions, iso_model = train_baseline_model(
            baseline_folder=args.baseline_folder,
            qwen_model_size=args.qwen_model_size,
            use_concept_list=use_concept_list,
            concept_list=concept_list,
            main_out_dir=main_out_dir,
            lim1=lim1,
            train_indices=None
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
            baseline_captions=baseline_captions,
            iso_model=iso_model,
            threshold_percentile=args.threshold_percentile, 
            lim1=lim1,
            lim2=lim2,
            save_outlier_list=(not args.no_save_outlier_list),
            test_indices=None
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
            
            if i % 2 == 1:  # 奇数パート (インデックスは0始まりなのでi%2==1が奇数)
                train_indices.extend(range(start_idx, end_idx))
            else:  # 偶数パート
                test_indices.extend(range(start_idx, end_idx))
        
        print(f"  Training on {len(train_indices)} images (odd parts)")
        print(f"  Testing on {len(test_indices)} images (even parts)")
        
        # 10fold方式でベースラインモデルを学習
        clip_extractor, candidate_labels, baseline_features, baseline_tsne, baseline_captions, iso_model = train_baseline_model(
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
            baseline_captions=baseline_captions,
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
