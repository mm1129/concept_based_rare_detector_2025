import os
import base64
import openai
import numpy as np
import mimetypes
import torch
from io import BytesIO
from PIL import Image

# import spacy
from sklearn.manifold import TSNE
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertModel
from transformers import CLIPProcessor, CLIPModel

from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')

# -----------------------------
# Embeddingモデルを管理するクラス
# -----------------------------
class EmbeddingModelManager:
    def __init__(self, model_type="sentence_transformer"):
        """
        model_type: "sentence_transformer" | "bert" | "clip" など
        """
        self.model_type = model_type.lower()
        
        # spaCy (名詞抽出などに使う想定)
        # self.nlp = spacy.load("en_core_web_sm")

        if self.model_type == "sentence_transformer":
            self.model = SentenceTransformer('all-MiniLM-L6-v2')

        elif self.model_type == "bert":
            # BERTの初期化
            self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            self.bert_model = BertModel.from_pretrained("bert-base-uncased")
            self.bert_model.eval()

        elif self.model_type == "clip":
            # CLIPの初期化
            self.clip_model_name = "openai/clip-vit-base-patch32"
            self.clip_model = CLIPModel.from_pretrained(self.clip_model_name)
            self.clip_processor = CLIPProcessor.from_pretrained(self.clip_model_name)

        else:
            raise ValueError("Unsupported model_type. Choose 'sentence_transformer', 'bert', or 'clip'.")

    def get_text_embeddings(self, text_list):
        """
        text_list: ['some text', 'another text', ...]
        return: numpy array of embeddings, shape = (len(text_list), embedding_dim)
        """
        if self.model_type == "sentence_transformer":
            emb = self.model.encode(text_list)
            return emb

        elif self.model_type == "bert":
            # BERTで [CLS] の隠れ状態をEmbeddingとして使用
            all_embs = []
            for text in text_list:
                inputs = self.bert_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
                with torch.no_grad():
                    outputs = self.bert_model(**inputs)
                # outputs.last_hidden_state.shape = (batch_size, seq_len, hidden_size)
                # 通常 [CLS] は index=0 にある
                cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze(0)  # (hidden_size,)
                all_embs.append(cls_embedding.numpy())
            return np.array(all_embs)

        elif self.model_type == "clip":
            inputs = self.clip_processor(text=text_list, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                text_features = self.clip_model.get_text_features(**inputs)
            return text_features.cpu().numpy()
        else:
            return None

    def get_image_embedding(self, image_path):
        """
        """
        if self.model_type != "clip":
            return None
        pil_image = Image.open(image_path).convert("RGB")
        inputs = self.clip_processor(images=pil_image, return_tensors="pt")
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs)
        return image_features.cpu().numpy()

    def extract_nouns(self, text):
        """
        """
        # doc = self.nlp(text)
        # nouns = [token.text for token in doc if token.pos_ == "NOUN"]
        # return nouns
        return text

def combine_text_image_embeddings(text_emb, image_emb):
    # (1, d_text) 形式になっている場合があるので shape を合わせる
    if len(text_emb.shape) == 2:
        text_emb = text_emb[0]
    if len(image_emb.shape) == 2:
        image_emb = image_emb[0]

    combined = np.concatenate((text_emb, image_emb), axis=-1)
    return combined


# -----------------------------
# 既存の関数群（必要部分のみ再掲）
# -----------------------------
def resize_encode_image(image_path, max_size=512):
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'
    try:
        with open(image_path, "rb") as image_file:
            image = Image.open(image_file)
            image.thumbnail((max_size, max_size))
            if image.mode != "RGB":
                image = image.convert("RGB")
            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            base64_encoded_data = base64.b64encode(buffered.getvalue()).decode('utf-8')
    except FileNotFoundError:
        raise FileNotFoundError(f"The specified image file was not found: {image_path}")
    except Exception as e:
        raise Exception(f"An error occurred while encoding the image: {e}")

    return f"data:{mime_type};base64,{base64_encoded_data}"

def encode_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return encoded_string
    except FileNotFoundError:
        print(f"Error: The file {image_path} was not found.")
        return None
    except Exception as e:
        print(f"Unexpected error while encoding image: {e}")
        return None

def generate_description(image_path):
    base64_image = encode_image(image_path)
    if not base64_image:
        return None

    messages = [
        {
            "role": "system",
            "content": (
                "You are an AI vision system specialized in describing images "
                "with a focus on unusual or rare objects and scenarios relevant "
                "to autonomous driving. Provide clear, concise, numbered lists "
                "of objects or scenarios in the image."
            )
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Analyze the following image and list up to 10 notable objects/scenarios. "
                        "Please output each item in a concise numbered list (like '1. XXX', '2. XXX'). "
                        "Avoid any filler words like 'Here are' or 'From the image, I can see'. "
                        "Focus on relevant or interesting objects (e.g., baby stroller, group of pedestrians, "
                        "construction vehicles, etc.)."
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
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=150,
            temperature=0.5
        )
        description = response.choices[0].message['content'].strip()
        print(f"Generated description: {description}")
        return description
    except openai.error.OpenAIError as e:
        print(f'OpenAI API error: {e}')
        return None
    except Exception as e:
        print(f'Unexpected error: {e}')
        return None

def parse_gpt_description(description):
    lines = description.split('\n')
    items = []
    for line in lines:
        line_stripped = line.strip()
        if line_stripped and line_stripped[0].isdigit():
            items.append(line_stripped)
    return items

def process_images(folder_path, embed_manager):

    all_nouns = []
    all_embeddings = []
    image_files = [f for f in os.listdir(folder_path)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    image_files = image_files[:10]
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        print(f"Processing {image_file}...")
        description = generate_description(image_path)
        print(f"Description: {description}")
        if not description:
            print(f"No description for {image_file}.")
            continue

        # embed_managerのextract_nouns()などを使うのもあり
        nouns = description.split()
        print(f"Nouns: {nouns}")

        if nouns:
            embeddings = embed_manager.get_text_embeddings(nouns)
            all_nouns.extend(nouns)
            all_embeddings.extend(embeddings)
            print(f"Processed {image_file}: {nouns}")
        else:
            print(f"No nouns extracted for {image_file}.")
    return all_nouns, np.array(all_embeddings)

def visualize_embeddings(embeddings, labels=None, title="Latent Space Visualization with t-SNE"):
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
    reduced_embeddings = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        reduced_embeddings[:, 0],
        reduced_embeddings[:, 1],
        c=labels if labels is not None else 'blue',
        cmap="coolwarm", alpha=0.6
    )
    if labels is not None:
        plt.legend(*scatter.legend_elements(), title="Classes")
    plt.title(title)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.savefig("projects/detect_rare_images/data/plot.png")
    
    return reduced_embeddings

def train_outlier_detector(embeddings):
    isolation_forest = IsolationForest(contamination=0.1, random_state=42)
    isolation_forest.fit(embeddings)
    return isolation_forest

def detect_rare_objects(detector_model, embeddings):
    outlier_scores = detector_model.predict(embeddings)
    return outlier_scores

def judge_new_images(folder_path, outlier_detector, embed_manager):
    rare_images = []
    image_files = [f for f in os.listdir(folder_path) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        print(f"Processing {image_file}...")
        description = generate_description(image_path)
        if not description:
            print(f"No description for {image_file}. Skipping.")
            continue

        items = parse_gpt_description(description)
        print(f"Parsed items: {items}")

        image_is_rare = False
        for item_text in items:
            tokens = item_text.split()
            if not tokens:
                continue

            embeddings = embed_manager.get_text_embeddings(tokens)

            outlier_scores = detect_rare_objects(outlier_detector, embeddings)
            if any(score == -1 for score in outlier_scores):
                image_is_rare = True
                print(f"Detected rare objects in item: {item_text}")
                break

        if image_is_rare:
            rare_images.append(image_path)
    return rare_images
def process_images_with_text_and_image(folder_path, embed_manager):
    all_embeddings = []
    image_files = [f for f in os.listdir(folder_path)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        print(f"Processing {image_file}...")

        # 1) GPTなどで画像説明を生成
        description = generate_description(image_path)
        if not description:
            print(f"No description for {image_file}. Skipping.")
            continue
        
        # 2) テキストを分割してトークン化
        tokens = description.split()
        if not tokens:
            print(f"No tokens for {image_file}. Skipping.")
            continue

        # 3) テキストEmbedding
        text_embedding = embed_manager.get_text_embeddings(tokens)  # (n_tokens, d_text)
        # 複数トークンがあるので、平均をとるか、あるいは何らかの集約
        text_embedding_mean = np.mean(text_embedding, axis=0)  # (d_text,)

        # 4) 画像Embedding
        img_embedding = embed_manager.get_image_embedding(image_path)  # (1, d_img) or (d_img,)
        if img_embedding is None:
            print(f"No image embedding for {image_file}. Skipping.")
            continue

        # 5) 連結
        combined_embedding = combine_text_image_embeddings(text_embedding_mean.reshape(1, -1),
                                                           img_embedding.reshape(1, -1))
        # shape = (d_text + d_img, )

        # 6) リストに格納
        all_embeddings.append(combined_embedding)

    if len(all_embeddings) == 0:
        return np.array([])

    return np.vstack(all_embeddings)  # shape=(N, d_text + d_img)
def judge_new_images_with_both(folder_path, outlier_detector, embed_manager):
    rare_images = []
    image_files = [f for f in os.listdir(folder_path) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        print(f"Processing {image_file}...")

        description = generate_description(image_path)
        if not description:
            print(f"No description for {image_file}. Skipping.")
            continue

        tokens = description.split()
        if not tokens:
            continue

        # テキストEmbedding(平均)
        text_embedding = embed_manager.get_text_embeddings(tokens)
        text_embedding_mean = np.mean(text_embedding, axis=0)

        # 画像Embedding
        img_embedding = embed_manager.get_image_embedding(image_path)
        if img_embedding is None:
            print(f"No image embedding for {image_file}. Skipping.")
            continue

        # 連結
        combined_embedding = combine_text_image_embeddings(
            text_embedding_mean.reshape(1, -1),
            img_embedding.reshape(1, -1)
        ).reshape(1, -1)  # shape=(1, d_text + d_img)

        # Outlier判定
        pred = outlier_detector.predict(combined_embedding)  # array([-1]) or array([1])
        if pred[0] == -1:
            rare_images.append(image_file)
            print(f"Rare image detected: {image_file}")

    return rare_images
def main():
    # chosen_model_type = "sentence_transformer"  # 例: Sentence-BERT
    # chosen_model_type = "bert"                 # 例: BERT
    chosen_model_type = "clip"                 # 例: CLIP

    embed_manager = EmbeddingModelManager(model_type=chosen_model_type)

    baseline_folder = "projects/detect_rare_images/data/data/nuscenes_image"
    new_images_folder = "projects/detect_rare_images/data/test_data"

    print("Processing baseline images...")
    baseline_nouns, baseline_embeddings = process_images(baseline_folder, embed_manager)
    
    if baseline_embeddings.size == 0:
        print("No embeddings found in baseline images. Exiting.")
        return

    print(f"Total nouns in baseline: {len(baseline_nouns)}")
    print(f"Baseline embeddings shape: {baseline_embeddings.shape}")

    print("Visualizing baseline embeddings...")
    visualize_embeddings(baseline_embeddings, title="Baseline Latent Space with t-SNE")

    print("Training outlier detection model on baseline embeddings...")
    outlier_detector = train_outlier_detector(baseline_embeddings)

    print("\nProcessing new images for rare object detection...")
    rare_images = judge_new_images(new_images_folder, outlier_detector, embed_manager)
    print(f"Rare images: {rare_images}")
    # baseline_folder = "projects/detect_rare_images/data/data/nuscenes_image"
    # baseline_embeddings = process_images_with_text_and_image(baseline_folder, embed_manager)

    # if baseline_embeddings.shape[0] == 0:
    #     print("No baseline embeddings. Exiting.")
    #     return

    # # 学習
    # outlier_detector = IsolationForest(contamination=0.1, random_state=42)
    # outlier_detector.fit(baseline_embeddings)
    # rare_images = judge_new_images(new_images_folder, outlier_detector, embed_manager)
    # print(f"Rare images: {rare_images}")
if __name__ == "__main__":
    main()
