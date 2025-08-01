import os
import base64
import openai
import numpy as np
import mimetypes
import torch
from io import BytesIO
from PIL import Image

# from sklearn.manifold import TSNE  # 必要なら可視化用に
from sklearn.ensemble import IsolationForest
# from sklearn.svm import OneClassSVM
# import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertModel
from transformers import CLIPProcessor, CLIPModel

from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')

class EmbeddingModelManager:
    def __init__(self, model_type="clip"):
        self.model_type = model_type.lower()

        if self.model_type == "clip":
            self.clip_model_name = "openai/clip-vit-base-patch32"
            self.clip_model = CLIPModel.from_pretrained(self.clip_model_name)
            self.clip_processor = CLIPProcessor.from_pretrained(self.clip_model_name)

        elif self.model_type == "sentence_transformer":
            self.model = SentenceTransformer('all-MiniLM-L6-v2')

        elif self.model_type == "bert":
            self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            self.bert_model = BertModel.from_pretrained("bert-base-uncased")
            self.bert_model.eval()

        else:
            raise ValueError("Unsupported model_type. Choose 'clip', 'sentence_transformer', or 'bert'.")

    def get_image_embedding(self, image_path):
        """
        CLIPを使って画像Embeddingを取得
        """
        if self.model_type != "clip":
            return None
        pil_image = Image.open(image_path).convert("RGB")
        inputs = self.clip_processor(images=pil_image, return_tensors="pt")
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs)
        return image_features.cpu().numpy()

    def get_text_embeddings(self, text_list):
        """
        ここでは使わないが、一応定義しておく
        """
        if self.model_type == "clip":
            inputs = self.clip_processor(text=text_list, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                text_features = self.clip_model.get_text_features(**inputs)
            return text_features.cpu().numpy()
        elif self.model_type == "sentence_transformer":
            return self.model.encode(text_list)
        elif self.model_type == "bert":
            all_embs = []
            for text in text_list:
                inputs = self.bert_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
                with torch.no_grad():
                    outputs = self.bert_model(**inputs)
                cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze(0)
                all_embs.append(cls_embedding.numpy())
            return np.array(all_embs)
        else:
            return None

def process_images_with_clip(folder_path, embed_manager):
    """
    フォルダ内の画像をCLIPの画像エンコーダでEmbeddingし、リストで返す
    """
    all_embeddings = []
    image_files = [f for f in os.listdir(folder_path)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        print(f"Processing (CLIP image) {image_file}...")

        emb = embed_manager.get_image_embedding(image_path)
        if emb is None:
            print(f"Skipping {image_file}: no embedding.")
            continue

        all_embeddings.append(emb[0])  # embは(1, 512)などなので、[0]で(512,)に
    return np.array(all_embeddings), image_files

def main():
    # モデルタイプをCLIPに
    chosen_model_type = "clip"
    embed_manager = EmbeddingModelManager(model_type=chosen_model_type)

    # フォルダ指定
    baseline_folder = "projects/detect_rare_images/data/data/nuscenes_image"
    new_images_folder = "projects/detect_rare_images/data/test_data"

    # --- 1) baseline画像から埋め込みを作成 ---
    print("Processing baseline images with CLIP image encoder...")
    baseline_embeddings, baseline_files = process_images_with_clip(baseline_folder, embed_manager)

    if baseline_embeddings.size == 0:
        print("No embeddings found in baseline images. Exiting.")
        return

    print(f"Baseline embeddings shape: {baseline_embeddings.shape}")

    # --- 2) IsolationForest学習 ---
    print("Training IsolationForest on baseline embeddings...")
    isolation_forest = IsolationForest(contamination=0.1, random_state=42)
    isolation_forest.fit(baseline_embeddings)

    # --- 3) 新規画像の埋め込みを作成＆判定 ---
    print("\nProcessing new images for rare object detection...")
    new_embeddings, new_files = process_images_with_clip(new_images_folder, embed_manager)

    if new_embeddings.size == 0:
        print("No new images to process.")
        return

    # レア判定
    preds = isolation_forest.predict(new_embeddings)  # -1: outlier
    rare_images = []
    for img_file, pred in zip(new_files, preds):
        if pred == -1:
            rare_images.append(img_file)

    print("\nRare images (based on CLIP image embeddings):")
    for r in rare_images:
        print("  ", r)

if __name__ == "__main__":
    main()
