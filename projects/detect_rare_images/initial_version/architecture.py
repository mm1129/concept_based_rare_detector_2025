#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import base64
import mimetypes
from sklearn.manifold import TSNE
from sklearn.ensemble import IsolationForest

# YOLO (Ultralytics)
# Prepare appropriate version, e.g., `pip install ultralytics==8.0.20`
try:
    import ultralytics
    from ultralytics import YOLO
except ImportError:
    print("Ultralytics YOLO not found. Install via 'pip install ultralytics'.")

# transformers
from transformers import CLIPProcessor, CLIPModel

# Uncomment below to use spaCy for noun extraction after installation
# import spacy
# nlp = spacy.load("en_core_web_sm")

###############################################################################
# 1. 2D Detection
###############################################################################
class DetectorYOLO:
    """
    Class for object detection using YOLOv8
    """
    def __init__(self, model_path="yolov8n.pt", device="cuda" if torch.cuda.is_available() else "cpu"):
        # Example: 'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', etc.
        self.model = YOLO(model_path)
        self.device = device
        self.model.to(self.device)

    def detect_and_crop(self, image_path, out_dir="cropped_objects", conf_thres=0.3):
        """
        Perform object detection on the input image and crop objects with confidence above the threshold.
        Args:
          image_path: Path to the input image.
          out_dir: Directory to save cropped objects.
          conf_thres: Confidence threshold for detection.
        Returns:
          cropped_info_list: [(out_path, class_name, conf), ...]
        """
        os.makedirs(out_dir, exist_ok=True)
        img = Image.open(image_path).convert("RGB")
        # Inference
        results = self.model.predict(source=img, conf=conf_thres, device=self.device)

        if len(results) == 0:
            return []

        # For YOLOv8, results[0].boxes contains the Boxes object
        # Access boxes.xyxy, boxes.conf, boxes.cls, etc.
        boxes = results[0].boxes
        names = self.model.names  # Class id -> Class name
        cropped_info_list = []

        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
            conf = float(boxes.conf[i].cpu().numpy())
            cls_id = int(boxes.cls[i].cpu().numpy())
            class_name = names[cls_id] if cls_id in names else str(cls_id)

            x1, y1, x2, y2 = xyxy
            cropped = img.crop((x1, y1, x2, y2))
            out_path = os.path.join(
                out_dir, f"{os.path.basename(image_path).split('.')[0]}_{i}_{class_name}.jpg"
            )
            cropped.save(out_path)
            cropped_info_list.append((out_path, class_name, conf))

        return cropped_info_list

###############################################################################
# 2. CLIP Model (Vision-Language Model) for feature extraction
###############################################################################
class CLIPFeatureExtractor:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        # Load model (downloads on the first use)
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()
        if torch.cuda.is_available():
            self.model.to("cuda")

    def get_image_embedding(self, image_path):
        """
        Input image (path) -> CLIP image feature vector (numpy)
        """
        pil_image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=pil_image, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
        return image_features.cpu().numpy().squeeze(0)
    # text desctiption from chatGPT; no list of texts?

    def get_text_embedding(self, text_list):
        """
        Input list of text -> CLIP text feature vectors (numpy)
        """
        inputs = self.processor(text=text_list, return_tensors="pt", padding=True, truncation=True)
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
        return text_features.cpu().numpy()

###############################################################################
# 3. Parsing text (noun extraction, etc.) - if needed
###############################################################################
def extract_nouns(text):
    """
    Use spaCy or similar for noun extraction if necessary.
    Here, we demonstrate simple word splitting for simplicity.
    """
    # doc = nlp(text)
    # nouns = [token.text for token in doc if token.pos_ == "NOUN"]
    # return nouns
    return text.split()  # Simple splitting for demonstration

###############################################################################
# 4. t-SNE visualization
###############################################################################
def visualize_tsne(features, labels=None, title="t-SNE Visualization"):
    """
    features: shape = (N, D) ndarray
    labels:   shape = (N,) or None
    """
    if len(features) == 0:
        print("No features to visualize.")
        return

    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)-1), metric="cosine" if labels is None else "euclidean")
    reduced = tsne.fit_transform(features)

    plt.figure(figsize=(10, 7))
    if labels is not None:
        scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap="tab10", alpha=0.7)
        plt.legend(*scatter.legend_elements(), title="Classes")
    else:
        plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.7)
    plt.title(title)
    plt.xlabel("Dim1")
    plt.ylabel("Dim2")
    plt.show()

###############################################################################
# 5. Anomaly detection with IsolationForest
###############################################################################
def detect_outliers(features, contamination=0.1):
    """
    features: (N, D)
    contamination: Proportion of outliers in the data
    return: outlier_labels, shape=(N,), values are 1 or -1
    """
    if len(features) == 0:
        return []
    iso = IsolationForest(contamination=contamination, random_state=42)
    iso.fit(features)
    pred = iso.predict(features)  # 1: inlier, -1: outlier
    return pred

def judge_new_images(base_features, base_class_names, new_images_folder, detector, extractor, cropped_dir="cropped_new_objects"):
    """
    Detect objects, extract features, and identify rare images in the new_images_folder
    using an IsolationForest model trained on base_features.
    
    Args:
        base_features (numpy.ndarray): Feature vectors from the base dataset.
        base_class_names (list): Corresponding class names.
        new_images_folder (str): Path to the folder containing new images.
        detector (DetectorYOLO): YOLO object detector instance.
        extractor (CLIPFeatureExtractor): CLIP feature extractor instance.
        cropped_dir (str): Directory to save cropped objects from new images.
    
    Returns:
        List of (image_path, anomaly_score) tuples for new images.
    """
    os.makedirs(cropped_dir, exist_ok=True)
    
    # Train IsolationForest on base dataset
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    iso_forest.fit(base_features)
    
    # 1) Detect objects and crop in new images
    new_cropped_info = []
    for file_name in os.listdir(new_images_folder):
        if file_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            image_path = os.path.join(new_images_folder, file_name)
            print(f"Processing {image_path}")
            cropped_info = detector.detect_and_crop(image_path, out_dir=cropped_dir)
            new_cropped_info.extend(cropped_info)
    
    # 2) Extract features from cropped new objects
    new_features = []
    new_paths = []
    
    for (obj_path, _, _) in new_cropped_info:
        emb = extractor.get_image_embedding(obj_path)
        new_features.append(emb)
        new_paths.append(obj_path)
    
    if len(new_features) == 0:
        print("No new objects detected.")
        return []
    
    new_features = np.array(new_features)
    print("New features shape:", new_features.shape)
    
    # 3) Predict anomaly scores for new features
    anomaly_scores = iso_forest.decision_function(new_features)  # Higher = more normal, Lower = more anomalous
    predictions = iso_forest.predict(new_features)  # 1 (normal), -1 (anomalous)
    
    # 4) Collect rare images with anomaly scores
    rare_images = [(new_paths[i], anomaly_scores[i]) for i, pred in enumerate(predictions) if pred == -1]
    
    # Print details for debugging
    for img, score in rare_images:
        print(f"Rare image detected: {img} | Anomaly score: {score:.4f}")
    
    print(f"Detected {len(rare_images)} rare images.")
    return rare_images


###############################################################################
# MAIN pipeline
###############################################################################
def main():
    # Example: 1) Prepare YOLO model for detection
    yolo_detector = DetectorYOLO(model_path="yolov8n.pt")  # Switch to a different model if needed
    # Example: 2) Extract image features using CLIP
    clip_extractor = CLIPFeatureExtractor(model_name="openai/clip-vit-base-patch32")

    input_folder = "projects/detect_rare_images/data/data/nuscenes_image"
    new_images_folder = "projects/detect_rare_images/data/test_data"
    input_folder = new_images_folder

    cropped_dir = "cropped_objects"
    os.makedirs(cropped_dir, exist_ok=True)

    # 1) Object detection & cropping
    all_cropped_info = []
    for file_name in os.listdir(input_folder):
        if file_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            image_path = os.path.join(input_folder, file_name)
            print(f"Detecting objects in {image_path}")
            cropped_info = yolo_detector.detect_and_crop(image_path, out_dir=cropped_dir)
            all_cropped_info.extend(cropped_info)  # (path, class_name, conf)

    # 2) Extract feature vectors using CLIP
    #    cropped_info = [(cropped_img_path, class_name, conf), ...]
    features = []
    class_names = []
    for (obj_path, cls_name, conf) in all_cropped_info:
        emb = clip_extractor.get_image_embedding(obj_path)
        features.append(emb)
        class_names.append(cls_name)  # Here, simply retain class_name as label

    features = np.array(features)
    print("features shape:", features.shape)

    # 3) Visualize using t-SNE
    #   Convert class_names to category IDs for color mapping
    unique_labels = list(set(class_names))
    label_to_id = {lab: idx for idx, lab in enumerate(unique_labels)}
    color_ids = [label_to_id[lab] for lab in class_names]

    visualize_tsne(features, labels=color_ids, title="t-SNE of Object Embeddings")

    # # 4) Detect anomalies using IsolationForest
    outlier_labels = detect_outliers(features, contamination=0.1)
    # outlier_labels[i] == -1 indicates an outlier
    for i, label in enumerate(outlier_labels):
        if label == -1:
            path, cls_name, conf = all_cropped_info[i]
            print(f"Outlier detected: {path} (class={cls_name}, conf={conf})")

    # rare_images = judge_new_images()

    print("Done.")

###############################################################################
if __name__ == "__main__":
    main()
