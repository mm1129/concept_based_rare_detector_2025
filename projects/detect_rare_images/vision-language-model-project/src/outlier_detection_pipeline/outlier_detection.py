#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from collections import Counter
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


def train_isolation_forest(features, contamination=0.1, random_state=42):
    """IsolationForestモデルを訓練する関数"""
    print(f"IsolationForestモデルを訓練中 (contamination={contamination})...")
    model = IsolationForest(
        n_estimators=100,
        max_samples='auto',
        contamination=float(contamination),
        random_state=random_state
    )
    model.fit(features)
    return model


def predict_outliers(features, model):
    """IsolationForestモデルで予測を行う関数"""
    return model.predict(features)


def filter_rare_classes(valid_cropped_info, class_names):
    """希少クラスに関連する可能性が高いオブジェクトを優先的に抽出"""
    # 希少クラスのキーワード
    rare_keywords = [
        "construction_vehicle", "bicycle", "motorcycle", "trailer", "truck",
        "bulldozer", "excavator", "forklift", "cement_mixer", "road_roller",
        "backhoe", "cherry_picker", "unusual", "rare", "strange"
    ]
    
    # 各オブジェクトに希少度スコアを付与
    rare_scores = []
    for i, info in enumerate(valid_cropped_info):
        class_name = info[1].lower()
        score = 0
        
        # キーワードマッチングによるスコア付け
        for keyword in rare_keywords:
            if keyword in class_name:
                score += 5  # 希少クラスに該当する場合は高スコア
                break
        
        # クラスの出現頻度によるスコア付け
        class_count = class_names.count(info[1])
        if class_count < 5:
            score += 10  # 非常に希少
        elif class_count < 20:
            score += 5   # やや希少
            
        rare_scores.append((i, score))
    
    # スコア順にソート
    rare_scores.sort(key=lambda x: x[1], reverse=True)
    
    # 上位のインデックスを返す（全体の30%程度）
    top_count = max(int(len(valid_cropped_info) * 0.3), 50)  # 最低50個は確保
    return [idx for idx, _ in rare_scores[:top_count]]


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