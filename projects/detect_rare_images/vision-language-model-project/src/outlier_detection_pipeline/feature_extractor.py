#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
from PIL import Image
import time
from .utils import get_device

try:
    from transformers import CLIPProcessor, CLIPModel
except ImportError:
    print("Transformers not found. Install via 'pip install transformers'.")

class CLIPFeatureExtractor:
    """
    CLIPを使用して画像やテキストの特徴ベクトル（埋め込み）を取得するクラス。
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
        指定された画像パスからCLIP画像特徴（np.array, shape=(dim,)）を取得します。
        """
        pil_image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
        return image_features.cpu().numpy().squeeze(0)

    def get_image_embeddings_batch(self, image_paths, batch_size=32):
        """複数の画像のCLIP特徴をバッチ処理で一度に取得"""
        all_embeddings = []
        total = len(image_paths)
        
        # 進捗表示を追加
        start_time = time.time()
        
        for i in range(0, total, batch_size):
            # 進捗表示（100バッチごとまたは最後）
            if i % (batch_size * 100) == 0 or i + batch_size >= total:
                elapsed = time.time() - start_time
                progress = min(i + batch_size, total) / total * 100
                eta = (elapsed / (i + batch_size)) * (total - (i + batch_size)) if i > 0 else 0
                print(f"特徴抽出進捗: {progress:.1f}% ({min(i + batch_size, total)}/{total}) | 経過: {elapsed:.1f}秒 | 残り: {eta:.1f}秒")
            
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
        テキストのリストに対するCLIPテキスト特徴ベクトル（np.array, shape=(len(text_list), dim)）を取得します。
        """
        inputs = self.processor(text=text_list, return_tensors="pt",
                                padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
        return text_features.cpu().numpy()

    def compute_image_text_similarity(self, image_embedding, text_embedding):
        """
        画像埋め込みとテキスト埋め込み間のコサイン類似度を計算
        
        Args:
            image_embedding: 単一の画像埋め込み (numpy array, shape=(dim,))
            text_embedding: 単一のテキスト埋め込み (numpy array, shape=(dim,))
            
        Returns:
            コサイン類似度 (float)
        """
        # 正規化
        image_emb_norm = image_embedding / np.linalg.norm(image_embedding)
        text_emb_norm = text_embedding / np.linalg.norm(text_embedding)
        
        # コサイン類似度を計算
        similarity = np.dot(image_emb_norm, text_emb_norm)
        return similarity


def parse_text_with_probability(base_text, candidate_labels, clip_extractor, image_embedding=None, weight_text=0.5):
    """
    CLIP埋め込みを使用してベーステキストと候補ラベルを比較し、
    コサイン類似度を計算して、ラベルから確率への辞書を返します。
    
    画像埋め込みが提供された場合、テキスト間類似度と画像-テキスト間類似度の
    加重平均を計算します。
    
    Args:
        base_text: 比較するベーステキスト
        candidate_labels: 候補ラベルのリスト
        clip_extractor: CLIPFeatureExtractorインスタンス
        image_embedding: 画像埋め込み (オプション)
        weight_text: テキスト類似度の重み (0.0〜1.0)
            
    Returns:
        ラベルから確率への辞書
    """
    if not base_text:
        return {}

    # テキスト埋め込みを取得
    text_emb = clip_extractor.get_text_embedding([base_text])
    labels_emb = clip_extractor.get_text_embedding(candidate_labels)

    # テキスト埋め込みを正規化
    text_emb_norm = text_emb / np.linalg.norm(text_emb, axis=1, keepdims=True)
    labels_emb_norm = labels_emb / np.linalg.norm(labels_emb, axis=1, keepdims=True)

    # テキスト間のコサイン類似度を計算
    text_text_sims = []
    for i in range(labels_emb_norm.shape[0]):
        sim = (text_emb_norm[0] * labels_emb_norm[i]).sum()
        text_text_sims.append(sim)
    text_text_sims = np.array(text_text_sims)
    
    # 画像埋め込みが提供された場合、画像-テキスト間の類似度も計算
    if image_embedding is not None:
        # 画像埋め込みを正規化
        image_emb_norm = image_embedding / np.linalg.norm(image_embedding)
        
        # 画像-ラベル間の類似度を計算
        image_text_sims = []
        for i in range(labels_emb_norm.shape[0]):
            sim = (image_emb_norm * labels_emb_norm[i]).sum()
            image_text_sims.append(sim)
        image_text_sims = np.array(image_text_sims)
        
        # 重み付き平均を計算
        weight_image = 1.0 - weight_text
        cos_sims = weight_text * text_text_sims + weight_image * image_text_sims
        
        # 計算過程をログに出力（デバッグ用）
        # top_idx = np.argsort(cos_sims)[-5:][::-1]  # 上位5つのインデックス
        # for idx in top_idx:
        #     label = candidate_labels[idx]
        #     print(f"Label: {label}, Text-Text: {text_text_sims[idx]:.4f}, Image-Text: {image_text_sims[idx]:.4f}, Combined: {cos_sims[idx]:.4f}")
    else:
        # 画像埋め込みがない場合はテキスト間の類似度のみを使用
        cos_sims = text_text_sims

    # 確率に変換して辞書を作成
    label_probs = {candidate_labels[i]: float(cos_sims[i]) for i in range(len(candidate_labels)) if cos_sims[i] >= 0}
    label_probs = {k: v for k, v in sorted(label_probs.items(), key=lambda x: x[1], reverse=True)}
    return label_probs 

def adaptive_concept_similarity(image_path, candidate_labels, clip_extractor, description=None):
    """
    説明文の品質に応じて重みを適応的に調整する類似度計算
    より洗練された画像-テキスト対応付けロジック
    
    Args:
        image_path: 画像ファイルのパス
        candidate_labels: 候補ラベルのリスト
        clip_extractor: CLIPFeatureExtractorインスタンス
        description: 画像の説明文（オプション）
        
    Returns:
        ラベルから確率への辞書
    """
    try:
        # 画像埋め込みを取得
        image_embedding = clip_extractor.get_image_embedding(image_path)
        
        # 候補ラベルの埋め込みを取得
        labels_embeddings = clip_extractor.get_text_embedding(candidate_labels)
        
        # 画像と各ラベル間の類似度を計算
        image_emb_norm = image_embedding / np.linalg.norm(image_embedding)
        labels_emb_norm = labels_embeddings / np.linalg.norm(labels_embeddings, axis=1, keepdims=True)
        
        image_label_sims = []
        for i in range(labels_emb_norm.shape[0]):
            sim = np.dot(image_emb_norm, labels_emb_norm[i])
            image_label_sims.append(sim)
        image_label_sims = np.array(image_label_sims)
        
        # 説明文が提供された場合
        if description and isinstance(description, str) and len(description.strip()) > 0:
            # 説明文のエンコード
            text_embedding = clip_extractor.get_text_embedding([description])
            text_emb_norm = text_embedding / np.linalg.norm(text_embedding, axis=1, keepdims=True)
            
            # 説明文と画像の埋め込み間の類似度を計算（説明文の信頼性指標として使用）
            desc_img_sim = np.dot(image_emb_norm, text_emb_norm[0])
            
            # 説明文が適切かどうかの評価（0.0〜1.0のスコア）
            # 経験則に基づくヒューリスティック - CLIPの類似度は通常-0.2〜0.5の範囲に分布
            # -0.2の場合は0.0、0.5の場合は1.0になるようにスケーリング
            desc_quality = max(0.0, min(1.0, (desc_img_sim + 0.2) / 0.7))
            
            # 説明文と各ラベル間の類似度計算
            text_label_sims = []
            for i in range(labels_emb_norm.shape[0]):
                sim = np.dot(text_emb_norm[0], labels_emb_norm[i])
                text_label_sims.append(sim)
            text_label_sims = np.array(text_label_sims)
            
            # 説明文の質に基づいて重みを動的に調整
            # 説明文の質が高いほど、説明文の類似度により大きな重みを与える
            # 反対に低い場合は画像-ラベル直接比較に重きを置く
            text_weight = min(0.8, max(0.1, 0.2 + desc_quality * 0.6))
            image_weight = 1.0 - text_weight
            
            # 重み付き平均計算とデバッグ情報
            combined_sims = image_weight * image_label_sims + text_weight * text_label_sims
            
            # デバッグ情報（上位候補のみ）
            top_idx = np.argsort(combined_sims)[-5:][::-1]
            for idx in top_idx:
                label = candidate_labels[idx]
                # print(f"Label: {label}, Image-Text: {image_label_sims[idx]:.4f}, Text-Text: {text_label_sims[idx]:.4f}, Combined: {combined_sims[idx]:.4f}, Weight(T/I): {text_weight:.2f}/{image_weight:.2f}")
        else:
            # 説明文がない場合は画像-ラベル類似度のみを使用
            combined_sims = image_label_sims
            # デバッグ情報
            top_idx = np.argsort(combined_sims)[-3:][::-1]
            for idx in top_idx:
                label = candidate_labels[idx]
                # print(f"[No Description] Label: {label}, Image-Text: {image_label_sims[idx]:.4f}")
        
        # 確率に変換して辞書を作成
        label_probs = {candidate_labels[i]: float(combined_sims[i]) for i in range(len(candidate_labels)) if combined_sims[i] > -1.0}  # 閾値を少し下げて多様性を確保
        label_probs = {k: v for k, v in sorted(label_probs.items(), key=lambda x: x[1], reverse=True)}
        
        return label_probs
        
    except Exception as e:
        print(f"類似度計算エラー: {e}")
        # エラー時は空の辞書を返す
        return {}

def enhanced_ensemble_similarity(image_path, candidate_labels, clip_extractor, description=None):
    """
    複数の手法を組み合わせたアンサンブル類似度計算
    
    Args:
        image_path: 画像ファイルのパス
        candidate_labels: 候補ラベルのリスト
        clip_extractor: CLIPFeatureExtractorインスタンス
        description: 画像の説明文（オプション）
        
    Returns:
        ラベルから確率への辞書
    """
    try:
        # 方法1: 画像-ラベル直接比較
        image_embedding = clip_extractor.get_image_embedding(image_path)
        image_emb_norm = image_embedding / np.linalg.norm(image_embedding)
        
        labels_embeddings = clip_extractor.get_text_embedding(candidate_labels)
        labels_emb_norm = labels_embeddings / np.linalg.norm(labels_embeddings, axis=1, keepdims=True)
        
        method1_sims = []
        for i in range(labels_emb_norm.shape[0]):
            sim = np.dot(image_emb_norm, labels_emb_norm[i])
            method1_sims.append(sim)
        method1_sims = np.array(method1_sims)
        
        # 方法2: テキスト説明を経由した比較（説明文がある場合）
        if description and isinstance(description, str) and len(description.strip()) > 0:
            text_embedding = clip_extractor.get_text_embedding([description])
            text_emb_norm = text_embedding / np.linalg.norm(text_embedding, axis=1, keepdims=True)
            
            method2_sims = []
            for i in range(labels_emb_norm.shape[0]):
                sim = np.dot(text_emb_norm[0], labels_emb_norm[i])
                method2_sims.append(sim)
            method2_sims = np.array(method2_sims)
            
            # 説明文と画像の類似度を評価
            desc_img_sim = np.dot(image_emb_norm, text_emb_norm[0])
            
            # 信頼度評価を改善（CLIPの類似度分布を考慮）
            confidence = max(0.1, min(0.8, (desc_img_sim + 0.2) / 0.7))
            
            # 方法3: 類似度の補強（画像と説明文の両方に似ているラベルを優先）
            # 両方のモダリティで高いスコアを持つラベルを重視
            method3_sims = np.minimum(method1_sims, method2_sims) + 0.5 * np.maximum(method1_sims, method2_sims)
            
            # 最終的な重み付けアンサンブル（信頼度に応じて各手法の重みを調整）
            ensemble_weights = [
                max(0.3, 1.0 - confidence),  # 方法1の重み（直接比較）
                max(0.1, confidence * 0.7),  # 方法2の重み（テキスト経由）
                max(0.1, confidence * 0.3)   # 方法3の重み（補強法）
            ]
            
            # 重みの正規化
            total_weight = sum(ensemble_weights)
            ensemble_weights = [w / total_weight for w in ensemble_weights]
            
            # 重み付けアンサンブル
            combined_sims = (
                ensemble_weights[0] * method1_sims + 
                ensemble_weights[1] * method2_sims + 
                ensemble_weights[2] * method3_sims
            )
            
        else:
            # 説明文がない場合は直接比較のみ
            combined_sims = method1_sims
        
        # 確率に変換して辞書を作成
        label_probs = {candidate_labels[i]: float(combined_sims[i]) for i in range(len(candidate_labels)) if combined_sims[i] > -1.0}
        label_probs = {k: v for k, v in sorted(label_probs.items(), key=lambda x: x[1], reverse=True)}
        
        return label_probs
        
    except Exception as e:
        print(f"アンサンブル類似度計算エラー: {e}")
        return {} 